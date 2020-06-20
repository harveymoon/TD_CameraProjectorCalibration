"""
Major props to Alvaro Cassinelli & Niklas BergstrÃ¶m
https://www.youtube.com/watch?v=pCq7u2TvlxU

Open Frameworks : Cyril Diagne
https://github.com/cyrildiagne/ofxCvCameraProjectorCalibration



"""

import numpy as np
import cv2 
aruco = cv2.aruco  # make sure you include the required contributed libraries for OpenCV 


## local helper function for projecting the 2d dots onto the grid's plane
def intersectCirclesRaysToBoard(circles, rvec, t, K, dist_coef):
    circles_normalized = cv2.convertPointsToHomogeneous(cv2.undistortPoints(circles, K, dist_coef))
    if not rvec.size:
        return None
    R, _ = cv2.Rodrigues(rvec)
 
    # https://stackoverflow.com/questions/5666222/3d-line-plane-intersection
 
    plane_normal = R[2,:] # last row of plane rotation matrix is normal to plane
    plane_point = t.T     # t is a point on the plane
 
    epsilon = 1e-06
 
    circles_3d = np.zeros((0,3), dtype=np.float32)
 
    for p in circles_normalized:
        ray_direction = p / np.linalg.norm(p)
        ray_point = p
 
        ndotu = plane_normal.dot(ray_direction.T)
 
        if abs(ndotu) < epsilon:
            print ("no intersection or line is within plane")
 
        w = ray_point - plane_point
        si = -plane_normal.dot(w.T) / ndotu
        Psi = w + si * ray_direction + plane_point
 
        circles_3d = np.append(circles_3d, Psi, axis = 0)
 
    return circles_3d


class StereoCalibrate:
	"""
	this process has three steps:
	1. Calibrate the camera based on a grid pattern
	2. Calibrate the projector based on the camera
	3. Define the stereo calibration between the two lenses considering all of the known information.
	"""
	def __init__(self, ownerComp):
		self.ownerComp = ownerComp
		print('___Camera Projector Calibration INIT___')
		
		self.inputTop = op('null_frame')

		self.sqWidth = 12
		self.sqHeight = 8

		self.CameraRes = (100,100) # is re-set when first frame is grabbed

		self.dictionary = aruco.getPredefinedDictionary (aruco.DICT_4X4_50) #aruco.DICT_6X6_250) #
		self.board = cv2.aruco.CharucoBoard_create(self.sqWidth,self.sqHeight,0.035,0.0175, self.dictionary)

	def captureFrame(self):
		print('capture frame')

	def SaveBoard(self):
		imboard = self.board.draw((2000, 1300))
		fileSave = 'CharucoBoard.jpg'
		cv2.imwrite(fileSave, imboard)

	def GrabTop(self):
		target_top = self.inputTop
		input_w = target_top.width
		input_h = target_top.height
		# convert input frame to a numpy Array from 0-255
		pixels = target_top.numpyArray()[:,:,:3] * 255.0
		# Convert the pixel data to a CV Mat object
		cv_img = pixels.astype(np.uint8)
		#need to flip y
		cv_img = cv2.flip(cv_img, 0)
		#change to grayscape
		cv_gray = cv2.cvtColor(cv_img, cv2.COLOR_RGB2GRAY)
		frame = cv_img.copy()
		self.CameraRes = frame.shape[:2]
		return frame
		
	def ClearSets(self):
		parent().store('charucoCornersAccum', [])
		parent().store('charucoIdsAccum', [] )
		parent().par.Capturedsets = 0
		
	def FindGrids(self):
		print('Finding Grids')
		frame = parent().GrabTop()
		
		idDat = op('base_aruco_view/table_ids')
		idDat.clear(keepFirstRow = True)
		quadDat = op('base_aruco_view/table_quads')
		quadDat.clear(keepFirstRow = False)
		cornerDat = op('base_aruco_view/table_corners')
		cornerDat.clear(keepFirstRow = True)

		corners, ids, rejectedImgPoints = aruco.detectMarkers (frame, self.dictionary)
		aruco.drawDetectedMarkers (frame, corners, ids, (0, 255, 0))

		corners, ids, rejected, recovered = cv2.aruco.refineDetectedMarkers(frame, self.board, corners, ids, rejectedImgPoints)  #, cameraMatrix=K, distCoeffs=dist_coef)

		#fileSave = project.folder+'/testOutput.jpg'
		#cv2.imwrite(fileSave, quad_image)

		pCount = 1
		for ID in corners:
			quad = ''
			for vertex in ID[0]:
				idDat.appendRow(vertex)
				quad = quad+ ' ' + str(pCount)
				pCount += 1
			quadDat.appendRow([quad,1])

		# --------- detect ChAruco board -----------
 
		if corners == None or len(corners) == 0:
			print('no corners')
		else:
			ret, charucoCorners, charucoIds = cv2.aruco.interpolateCornersCharuco(corners, ids, frame, self.board)
			if(len(charucoCorners) == 0):
				print('NO CORNERS FOUND')
			else:
				print(ret)
				for corner in charucoCorners:
					cornerDat.appendRow(corner[0])

				cornerList = parent().fetch('charucoCornersAccum' , []) 
				cornerList += [charucoCorners]
				# parent().store('charucoCornersAccum', cornerList)

				idList = parent().fetch('charucoIdsAccum' , [])
				idList  += [charucoIds]
				# parent().store('charucoIdsAccum', idList )

				number_charuco_views = len(idList)
				parent().par.Capturedsets = number_charuco_views
				print(idList)


	def CalibrateCam(self):
		print('Calibrate Camera')

		cornerList = parent().fetch('charucoCornersAccum' , []) 
		idList = parent().fetch('charucoIdsAccum' , [])

		ret, K, dist_coef, rvecs, tvecs = cv2.aruco.calibrateCameraCharuco(cornerList, idList, self.board ,(self.CameraRes[0], self.CameraRes[1]),None, None) #K,dist_coef,flags = cv2.CALIB_USE_INTRINSIC_GUESS)
		
		parent().store('ret', ret)
		parent().store('K', K)
		parent().store('dist_coef', dist_coef)
		parent().store('rvecs', rvecs)
		parent().store('tvecs', tvecs)
	
		print("camera calib mat after\n%s"%K)
		print("camera dist_coef %s"%dist_coef.T)
		print("calibration reproj err %s"%ret)
		#op('text_circle_grid').run()

	def FindPose(self):
		print("Find Pose")

