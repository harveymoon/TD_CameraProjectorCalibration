"""
Major props to Alvaro Cassinelli & Niklas BergstrÃ¶m
https://www.youtube.com/watch?v=pCq7u2TvlxU

Open Frameworks : Cyril Diagne
https://github.com/cyrildiagne/ofxCvCameraProjectorCalibration


Made by Harvey Moon, Satoru Higa, Michael Walczyk

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

		# self.CameraCalibrationError = -1
		# self.ProjectorCalibrationError = -1

		self.ownerComp = ownerComp
		print('___Camera Projector Calibration INIT___')
		
		self.inputTop = op('null_frame')

		self.sqWidth = 12
		self.sqHeight = 8

		self.circleWidth = 7
		self.circleHeight = 6

		self.CameraRes = (100,100) # is re-set when first frame is grabbed

		self.dictionary = aruco.getPredefinedDictionary (aruco.DICT_4X4_50) #aruco.DICT_6X6_250) #
		self.board = cv2.aruco.CharucoBoard_create(self.sqWidth,self.sqHeight,0.035,0.0175, self.dictionary)


	def SaveBoard(self):
		"""
		This function will save a local jpg copy of the currently used Arcuo board. If you have not printed one out already, you will need to do so.
		"""
		imboard = self.board.draw((2000, 1300))
		fileSave = 'CharucoBoard.jpg'
		cv2.imwrite(fileSave, imboard)

	def GrabTop(self):
		"""
		This function grabs a frame from a TOP Operator within the network and returns a grayscale CV Frame output. 
		"""
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
		"""
		This function will clear out the saved camera features
		"""
		parent().store('charucoCornersAccum', [])
		parent().store('charucoIdsAccum', [] )
		parent().par.Capturedsets = 0

	def CaptureFrame(self):
		"""
		This function saves the current grid found into storage for calibration NOTE: maybe this should be renamed to 'StoreCorners'
		"""
		print('capture frame')
		cornerList = parent().fetch('charucoCornersAccum' , []) 
		idList = parent().fetch('charucoIdsAccum' , [])
		charucoCorners, charucoIds, corners = parent().FindGrids()
		cornerList += [charucoCorners]				
		idList  += [charucoIds]
		number_charuco_views = len(idList)
		parent().par.Capturedsets = number_charuco_views
		# print(idList)

				
	def FindGrids(self, frame = None):
		"""
		This function is will find grids in the input frame, it returns the arcuo shapes, id's found and the grid corners. 
		"""
		print('Finding Grids')

		if not frame:
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
			print(ret)
			if(len(charucoCorners) == 0):
				print('NO CORNERS FOUND')
			else: # has found corners
				for corner in charucoCorners:
					cornerDat.appendRow(corner[0])
				return charucoCorners, charucoIds, corners

			

		
	def FindPose(self, frame=None):  
		""" 
		this function should find a pose from the current frame and the detected charuco patterns
		returns :
			worldPos, retval, Srvec, Stvec
		"""
		# gridCorners, arucoIDs, arucoCorners = FindGrids()

		print("Find Pose")

		if frame is None:
			frame = parent().GrabTop()

		charucoCorners ,charucoIds , feducialQuads= parent().FindGrids()
		
		camera_intrinsics = parent().fetch('camera_intrinsics')
		# for fed in feducialQuads:
		# 	rvec, tvec, objPoints =  aruco.estimatePoseSingleMarkers(fed, .008, camera_intrinsics['K'], camera_intrinsics['dist_coef'])
		# 	aruco.drawAxis(frame, camera_intrinsics['K'], camera_intrinsics['dist_coef'], rvec, tvec, 0.01)  # Draw axis
		
		# fileSave = project.folder+'/poseDraw.jpg'
		# cv2.imwrite(fileSave, frame)

		# print('found all markers poses')
		

		retval, Srvec, Stvec = aruco.estimatePoseBoard(charucoCorners, charucoIds, self.board, camera_intrinsics['K'], camera_intrinsics['dist_coef'], None, None)
		Sdst, jacobian = cv2.Rodrigues(Srvec)
		extristics = np.matrix([[Sdst[0][0],Sdst[0][1],Sdst[0][2],Stvec[0][0]],
                             [Sdst[1][0],Sdst[1][1],Sdst[1][2],Stvec[1][0]],
                             [Sdst[2][0],Sdst[2][1],Sdst[2][2],Stvec[2][0]],
                             [0.0, 0.0, 0.0, 1.0]
                ])
		extristics_I = extristics.I  # inverse matrix
		worldPos = [extristics_I[0,3],extristics_I[1,3],extristics_I[2,3]]


		matrix_comp = op('base_camera_matrix')

		# Fill rotation table
		cv_rotate_table = matrix_comp.op('cv_rotate')
		for i, v in enumerate(Srvec):
		        for j, rv in enumerate(v):
		                cv_rotate_table[i, j] = rv

		# Fill translate vector
		cv_translate_table = matrix_comp.op('cv_translate')
		for i, v in enumerate(Stvec.T):
		        cv_translate_table[0, i] = v[0]

		projRez = (1920,1080)
		# Break appart camera matrix
		size = projRez
		# Computes useful camera characteristics from the camera matrix.
		fovx, fovy, focalLength, principalPoint, aspectRatio = cv2.calibrationMatrixValues(camera_intrinsics['K'], size, 1920, 1080)  # this is not right, last two arguments are supposed to be the aperature size in mm
		near = .1
		far = 2000

		return worldPos, retval, Srvec, Stvec


	def CalibrateCam(self):
		"""
		This function will take the saved camera checkerboard information and calculate an intrincis matrix for the camera and lens pair. 
		"""

		print('Calibrate Camera')

		cornerList = parent().fetch('charucoCornersAccum' , []) 
		idList = parent().fetch('charucoIdsAccum' , [])

		ret, K, dist_coef, rvecs, tvecs = cv2.aruco.calibrateCameraCharuco(cornerList, idList, self.board ,(self.CameraRes[0], self.CameraRes[1]),None, None) #K,dist_coef,flags = cv2.CALIB_USE_INTRINSIC_GUESS)

		parent().store('camera_intrinsics', {	
				'ret': ret,
			   	'K': K,
			   	'dist_coef': dist_coef,
			   	'rvecs': rvecs,
			   	'tvecs': tvecs
			   })	

	
		print("camera calib mat after\n%s"%K)
		print("camera dist_coef %s"%dist_coef.T)
		print("calibration reproj err %s"%ret)
		#op('text_circle_grid').run()



	def Create_undistorted_uv_map(self):
		'''Creates a UV map that can be used in combination with a remap TOP to 
		undistort the video feed later on.

		'''

		# If the UV map template does not already exist, create it
		# file_path = 'flat_map.exr'
		# if not os.path.isfile(file_path):
			# op('glsl_generate_uv_map').save(file_path)

		# Here, -1 lets you import floating-point images
		# cv_img = cv2.imread(file_path, -1) 

		target_top = op('glsl_generate_uv_map')
		input_w = target_top.width
		input_h = target_top.height
		# convert input frame to a numpy Array from 0-255
		# pixels = target_top.numpyArray()[:,:,:3] * 255.0
		pixels = target_top.numpyArray()[:,:,:3] 
		# Convert the pixel data to a CV Mat object
		# cv_img = pixels.astype(np.uint8)
		cv_img = pixels
		#need to flip y
		# cv_img = cv2.flip(cv_img, 0)
		#change to grayscape
		# cv_gray = cv2.cvtColor(cv_img, cv2.COLOR_RGB2GRAY)
		# cv_frame = cv_img.copy()


		# Compute the optimal new camera matrix based on the free scaling parameter, alpha - 
		# note that if alpha = 0, the image will be cropped
		alpha = 0
		h, w = cv_img.shape[:2]
		camera_intrinsics = parent().fetch('camera_intrinsics')
		undistorted_mtx, roi = cv2.getOptimalNewCameraMatrix(camera_intrinsics['K'], 
														  	 camera_intrinsics['dist_coef'], 
														  	 (w, h), alpha, (w, h))

		# Compute the undistortion and rectification transformation map
		mapx, mapy = cv2.initUndistortRectifyMap(camera_intrinsics['K'], 
												 camera_intrinsics['dist_coef'], 
												 None, undistorted_mtx, (w, h), 5)
		
		# Remap and (optionally) crop the image
		dst = cv2.remap(cv_img, mapx, mapy, cv2.INTER_LINEAR)
		x, y, w, h = roi
		dst = dst[y:y+h, x:x+w]
		
		# Save out the file
		save_path = 'undistortion_map.exr'
		cv2.imwrite(save_path, dst)
		
		# Update storage 
		camera_intrinsics['mtx_undistorted'] = undistorted_mtx
		#camera_intrinsics['mapping_undistorted'] = { 'x': mapx, 'y': mapy }
		#camera_intrinsics['uv_map_path_undistorted'] = save_path

		# parent().store('camera_intrinsics', camera_intrinsics)



	def FindCircleGrid(self):

		"""
		This function will discover the circle grids in camera space and then re-project the found locations back into 3d space using the calibrated camera. 
		"""

		camera_intrinsics = parent().fetch('camera_intrinsics')

		ret=camera_intrinsics['ret']
		K=camera_intrinsics['K']
		dist_coef=camera_intrinsics['dist_coef']


		frame = parent().GrabTop()

		circleGridScale = (self.circleWidth,self.circleHeight) 

		frame = cv2.bitwise_not(frame) #invert the frame to see the dots, NOTE: do I really need to do this?
		flags = cv2.CALIB_CB_SYMMETRIC_GRID
		# --------- detect circles -----------
		params = cv2.SimpleBlobDetector_Params()

		params.filterByArea = True
		params.maxArea = 10000
		params.minArea = 200

		params.minDistBetweenBlobs = 10
		params.minThreshold = 10
		params.maxThreshold = 220
		params.thresholdStep = 5
		
		params.filterByCircularity = True
		params.minCircularity = 0.8

		params.filterByInertia = False
		# params.minInertiaRatio = 0.01

		params.filterByConvexity = False
			# params.minConvexity = 0.5

		detector = cv2.SimpleBlobDetector_create(params)

		ret, circles = cv2.findCirclesGrid(frame, circleGridScale, flags=flags, blobDetector=detector)

		print('ret : ')
		print(ret)

		if(ret == False):
			print('NO CIRCLES FOUND, ABORT')
			return

		img = cv2.drawChessboardCorners(frame, circleGridScale, circles, ret)

		worldPos, retval, Rrvec, Rtvec = parent().FindPose(frame)

		rvec = Rrvec
		tvec = Rtvec


		print('Matrix K : ')
		print(K)

		cv2.undistortPoints(circles, K, dist_coef)

		# ray-plane intersection: circle-center to chessboard-plane
		circles3D = intersectCirclesRaysToBoard(circles, rvec, tvec, K, dist_coef)
		 
		# re-project on camera for verification
		circles3D_reprojected, _ = cv2.projectPoints(circles3D, (0,0,0), (0,0,0), K, dist_coef)
		
		for c in circles3D_reprojected:
		    cv2.circle(frame, tuple(c.astype(np.int32)[0]), 3, (255,255,0), cv2.FILLED)
		    

		circles3D = circles3D.astype('float32')
		####
		circle3DList = parent().fetch('3dCirclesFound' , []) 
		circle2DList = parent().fetch('2dCirclesFound' , [])
		circle3DList += [circles3D]				
		circle2DList  += [circles]
		number_circleGrid_views = len(circle2DList)
		parent().par.Capturedcirclesets = number_circleGrid_views
		###


		frame = cv2.bitwise_not(frame) #inverter because we inverted before, NOTE: is this necessary?
		fileSave = 'circleGridFound.jpg'
		cv2.imwrite(fileSave, frame)


		circle3Dat = op('geo_found_circle_grid/table_3d_circles')
		circle3Dat.clear()

		circle2Dat = op('geo_found_circle_grid/table_2d_circles')
		circle2Dat.clear()

		for rr in range(0, len(circles3D)):
			circle3Dat.appendRow(circles3D[rr])
			
		# for c in circles3D_reprojected:
		#     cv2.circle(frame, tuple(c.astype(np.int32)[0]), 3, (255,255,0), cv2.FILLED)
		
		for rr in range(0, len(circles)):
			circle2Dat.appendRow(circles[rr][0])


	def CalibrateProjector(self):

		"""
		this function should calibrate the projector given the known points based on the cameras calibration. Then it does a stereo calibration between the camera and the projector together. 
		"""

		print('projector calibration')

		projRez = (1920,1080)

		objectPointsAccum = parent().fetch('3dCirclesFound')

		circleDat = op('null_circle_centers')

		projCirclePoints = np.zeros((circleDat.numRows, 2), np.float32)
	
		for rr in range(0,circleDat.numRows):
			projCirclePoints[rr] = ( float(circleDat[rr,0]), float(circleDat[rr,1]) ) 

		projCirclePoints  = projCirclePoints.astype('float32')
		# objectPointsAccum = objectPointsAccum.astype('float32')
		objectPointsAccum =  np.asarray(objectPointsAccum, dtype=np.float32)
		# print(objectPointsAccum)

		print(len(objectPointsAccum))
		# print(projCirclePoints)
		projCirlcleList = []

		for ix in range(0,len(objectPointsAccum)):
			projCirlcleList.append(projCirclePoints)

		# This can be omitted and can be substituted with None below.
		K_proj = cv2.initCameraMatrix2D( objectPointsAccum,projCirlcleList , projRez)

		#K_proj = None
		dist_coef_proj = None

	    flags = 0
	    #flags |= cv2.CALIB_FIX_INTRINSIC
	    flags |= cv2.CALIB_USE_INTRINSIC_GUESS
	    #flags |= cv2.CALIB_FIX_PRINCIPAL_POINT
	    #flags |= cv2.CALIB_FIX_FOCAL_LENGTH
	    # flags |= cv2.CALIB_FIX_ASPECT_RATIO
	    # flags |= cv2.CALIB_ZERO_TANGENT_DIST
	    # flags |= cv2.CALIB_SAME_FOCAL_LENGTH
	    # flags |= cv2.CALIB_RATIONAL_MODEL
	    # flags |= cv2.CALIB_FIX_K3
	    # flags |= cv2.CALIB_FIX_K4
	    # flags |= cv2.CALIB_FIX_K5



		# the actual function that figures out the projectors projection matrix
		ret, K_proj, dist_coef_proj, rvecs, tvecs = cv2.calibrateCamera(objectPointsAccum,
		                                                                projCirlcleList,
		                                                                projRez,
		                                                                K_proj,
		                                                                dist_coef_proj,flags = flags)
		print("proj calib mat after\n%s"%K_proj)
		print("proj dist_coef %s"%dist_coef_proj.T)
		print("calibration reproj err %s"%ret)
		
		cameraCirclePoints = parent().fetch('2dCirclesFound')
		camera_intrinsics = parent().fetch('camera_intrinsics')
		K=camera_intrinsics['K']
		dist_coef=camera_intrinsics['dist_coef']
		# rvecs=camera_intrinsics['rvecs']
		# tvecs=camera_intrinsics['tvecs']
		 
		print("stereo calibration")
		ret, K, dist_coef, K_proj, dist_coef_proj, proj_R, proj_T, _, _ = cv2.stereoCalibrate(
		        objectPointsAccum,
		        cameraCirclePoints,
		        projCirlcleList,
		        K,
		        dist_coef,
		        K_proj,
		        dist_coef_proj,
		        projRez,
		        flags = cv2.CALIB_USE_INTRINSIC_GUESS
		        )
		proj_rvec, _ = cv2.Rodrigues(proj_R)
		 
		print("R \n%s"%proj_R)
		print("T %s"%proj_T.T)
		print("proj calib mat after\n%s"%K_proj)
		print("proj dist_coef %s"       %dist_coef_proj.T)
		print("cam calib mat after\n%s" %K)
		print("cam dist_coef %s"        %dist_coef.T)
		print("reproj err %f"%ret)


		parent().store('proj_intrinsics', {	
		'ret': ret,
		'K': K_proj,
		'dist_coef': dist_coef_proj,
		'rvecs': rvecs,
		'tvecs': tvecs
		})	

		camera_intrinsics['K'] = K
		camera_intrinsics['dist_coef'] = dist_coef

		####  below are two tests to try and get pose information back into touchdesigner camera components

		matrix_comp = op('base_camera_matrix')


		# Fill rotation table
		cv_rotate_table = matrix_comp.op('cv_rotate')
		for i, v in enumerate(proj_R):
		        for j, rv in enumerate(v):
		                cv_rotate_table[i, j] = rv

		# Fill translate vector
		cv_translate_table = matrix_comp.op('cv_translate')
		for i, v in enumerate(proj_T.T):
		        cv_translate_table[0, i] = v[0]

		# Break appart camera matrix
		size = projRez
		# Computes useful camera characteristics from the camera matrix.
		fovx, fovy, focalLength, principalPoint, aspectRatio = cv2.calibrationMatrixValues(K_proj, size, 1920, 1080)
		near = .1
		far = 2000


		####

		INTRINSIC = op('INTRINSIC')
		INTRINSIC.clear()
		INTRINSIC.appendRow(['K', 'dist', 'fovx', 'fovy', 'focal', 'image_width', 'image_height'])
		INTRINSIC.appendRow([
			K_proj.tolist(),
			dist_coef_proj.tolist(),
			fovx,
			fovy,
			focalLength,
			1920,
			1080,
		])



		# Fill values table
		cv_values = matrix_comp.op('cv_values')
		cv_values['focalX',1] = focalLength
		cv_values['focalY',1] = focalLength
		cv_values['principalX',1] = principalPoint[0]
		cv_values['principalY',1] = principalPoint[1]
		cv_values['width',1] = 1920
		cv_values['height',1] = 1080

		l = near * (-principalPoint[0]) / focalLength
		r = near * (1920 - principalPoint[0]) / focalLength
		b = near * (principalPoint[1] - 1080) / focalLength
		t = near * (principalPoint[1]) / focalLength

		A = (r + l) / (r - l)
		B = (t + b) / (t - b)
		C = (far + near) / (near - far)
		D = (2 * far * near) / (near - far)
		nrl = (2 * near) / (r - l)
		ntb = (2 * near) / (t - b)

		table = matrix_comp.op('table_camera_matrices')

		proj_mat = tdu.Matrix([nrl, 0, 0, 0], 
		                                          [0, ntb, 0, 0], 
		                                          [A, B, C, -1], 
		                                          [0, 0, D, 0])

		# Transformation matrix
		tran_mat = tdu.Matrix([ proj_R[0][0],  proj_R[0][1],   proj_R[0][2],  proj_T[0]], 
		                      [-proj_R[1][0], -proj_R[1][1],  -proj_R[1][2], -proj_T[1]],
		                      [-proj_R[2][0], -proj_R[2][1],  -proj_R[2][2], -proj_T[2]],
		                      [0,0,0,1])

		matrix_comp.op('table_camera_matrices').clear()
		matrix_comp.op('table_camera_matrices').appendRow(proj_mat) 
		matrix_comp.op('table_camera_matrices').appendRow(tran_mat) 