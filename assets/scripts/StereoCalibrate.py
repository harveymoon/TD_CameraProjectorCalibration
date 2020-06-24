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

def build_projection_matrix(focal_x, focal_y, principal_x, principal_y, near, far, width, height):
	fx = focal_x
	fy = focal_y

	if fx != 0.0:

		cx = principal_x
		cy = principal_y

		n = near
		f = far

		w = width
		h = height

		l = n * (-cx) / fx
		r = n * (w - cx) / fx
		b = n * (cy - h) / fy
		t = n * (cy) / fy

		A = (r + l) / (r - l)
		B = (t + b) / (t - b)
		C = (f + n) / (n - f)
		D = (2*f*n)/(n - f)

		nrl = (2*n)/(r-l)
		ntb = (2*n)/(t-b)

		return [[nrl,0,A,0],
				[0,ntb,B,0],
				[0,0,C,D],
				[0,0,-1,0]]

	else:
		return None 

class StereoCalibrate:
	"""
	Stereo calibration consists of 3 steps:

	1. Calibrate the camera based on a grid pattern
	2. Calibrate the projector based on the camera
	3. Define the stereo calibration between the two lenses considering all of the known information
	
	"""
	def __init__(self, ownerComp):

		# self.CameraCalibrationError = -1
		# self.ProjectorCalibrationError = -1

		self.ownerComp = ownerComp
		print('Initialized stereo calibration class')
		
		self.inputTop = op('null_frame')

		self.sqWidth = 12
		self.sqHeight = 8

		self.circleWidth = 7
		self.circleHeight = 6

		# This will be set whenever the first camera frame is captured
		self.CameraRes = (100, 100) 

		# Create the aruco board (CV Mat)
		self.dictionary = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
		self.board = cv2.aruco.CharucoBoard_create(self.sqWidth, self.sqHeight, 0.035, 0.0175, self.dictionary)

	def SaveBoard(self, file_name='charuco_board'):
		"""
		This function will save a local .jpg copy of the requested Arcuo board. 
		If you have not printed one out already, you will need to do so.

		"""
		board = self.board.draw((2000, 1300))
		file_name = '{}.jpg'.format(file_name)
		cv2.imwrite(file_name, board)

	def GrabTop(self, needs_grayscale_conversion=False):
		"""
		This function grabs a frame from a TOP Operator within the network and returns 
		a grayscale CV Mat output. 

		"""
		target_top = self.inputTop
		input_w = target_top.width
		input_h = target_top.height

		# Convert input frame to a numpy Array from 0-255
		pixels = target_top.numpyArray()[:,:,:3] * 255.0
		
		# Convert the pixel data to a CV Mat object, 0-255 range
		cv_img = pixels.astype(np.uint8)
		
		# Need to flip Y because of how TouchDesigner stores pixel data
		cv_img = cv2.flip(cv_img, 0)
		
		# Convert to grayscape (if Mono TOP isn't in the network?)
		if needs_grayscale_conversion:
			cv_img = cv2.cvtColor(cv_img, cv2.COLOR_RGB2GRAY)

		frame = cv_img.copy()

		# Set the camera resolution
		self.CameraRes = frame.shape[:2]

		return frame

	def ClearSets(self):
		"""
		This function will clear out the saved camera features.

		"""
		parent().store('charucoCornersAccum', [])
		parent().store('charucoIdsAccum', [] )
		parent().par.Capturedsets = 0

	def CaptureFrame(self):
		"""
		This function saves the current grid found into storage for calibration.

		NOTE: maybe this should be renamed to 'StoreCorners'
		
		"""
		print('Capturing camera frame...')

		# Fetch lists 
		corners_list = parent().fetch('charucoCornersAccum', []) 
		ids_list = parent().fetch('charucoIdsAccum', [])

		# Find the aruco corners (markers and chessboard)
		corners, ids, chessboard_corners, chessboard_ids, frame = parent().FindGrids()
		
		# Accumulate
		corners_list += [chessboard_corners]				
		ids_list += [chessboard_ids]

		# Set the custom par that shows the user how many views have been captured
		number_charuco_views = len(ids_list)
		parent().par.Capturedsets = number_charuco_views

	def FindGrids(self, frame=None):
		"""
		A charuco board consists of feducial markers overlaid on top of a chessboard
		grid. Per the OpenCV docs, the benefit of using these patterns is that they 
		provide the versatility of aruco boards + the precision of chessboards.

		This function finds such a grid in the input frame and returns 3 items:

		1. The detected marker corners
		2. The IDs of the marker corners that were found
		3. The chessboard corners (interpolated based on item 1)
		4. The IDs of the chessboard corners that were found
		5. The OpenCV Mat (frame) that was used for detection

		"""
		print('Finding charuco board...')
	
		# If a frame was not supplied, grab one from the TOP 
		if frame is None:
			frame = parent().GrabTop()
		
		DAT_ids = op('base_aruco_view/table_ids')
		DAT_ids.clear(keepFirstRow = True)
		DAT_quads = op('base_aruco_view/table_quads')
		DAT_quads.clear(keepFirstRow = False)
		DAT_corners = op('base_aruco_view/table_corners')
		DAT_corners.clear(keepFirstRow = True)

		# First, detect the markers - this function returns 3 things:
		# 1. corners: a vector of detected marker corners (for each marker,
		#    its four corners are provided)
		# 2. ids: a vector of identifiers of the detected markers (ints)
		# 3. rejected image points: the image points of those squares whose 
		#    inner code doesn't have a correct codification
		marker_corners, marker_ids, rejected_img_pts = aruco.detectMarkers(frame, self.dictionary)
		aruco.drawDetectedMarkers(frame, marker_corners, marker_ids, (0, 255, 0))
		
		# Next, refine the markers that weren't detected in the previous step based 
		# on the already detected markers and the known board layout
		# 
		# NOTE: you can optionally supply a camera matrix and distortion coefficients
		# to this function, which may help improve accuracy?
		marker_corners, marker_ids, rejected, recovered = cv2.aruco.refineDetectedMarkers(frame, self.board, marker_corners, marker_ids, rejected_img_pts)  
		print('\t{} marker corners found (after refinement)'.format(len(marker_corners)))

		# Maybe save the results to disk, if needed
		save_to_disk = False 
		if save_to_disk:
			file_name = project.folder + '/find_grids_output.jpg'
			cv2.imwrite(file_name, frame)

		# Put corner data into a DAT so that we can draw each chessboard element
		total_points = 1

		for corner in marker_corners:
			
			quad = ''
			#print('Corner:', corner)
			for vertex in corner[0]:
				DAT_ids.appendRow(vertex)
				quad = quad + ' ' + str(total_points)
				total_points += 1

			DAT_quads.appendRow([quad, 1])

		# Finally, find the chessboard corners based on the information gathered above
		if marker_corners == None or len(marker_corners) == 0:
			print('\tNo marker corners were detected - exiting')
		else:
			# Find the position of the chessboard corners based on the (now known)
			# marker positions using a local homography (you can optionally provide
			# a camera matrix if it is known)
			ret, chessboard_corners, chessboard_ids = cv2.aruco.interpolateCornersCharuco(marker_corners, marker_ids, frame, self.board)
			
			if len(chessboard_corners) == 0:
				print('\tNo chessboard corners were found (after interpolation) - exiting')
			else: 
				# Otherwise, some corners were found 
				for corner in chessboard_corners:
					DAT_corners.appendRow(corner[0])
				
				print('\t{} chessboard corners found (after interpolation)'.format(len(chessboard_corners)))

				return marker_corners, marker_ids, chessboard_corners, chessboard_ids, frame

	def CalibrateCam(self, file_name='camera_intrinsics'):
		"""
		This function will take the saved board information and calculate 
		an intrincis matrix for the camera. 
		
		"""
		print('Calibrating the camera...')
		corners_list = parent().fetch('charucoCornersAccum', []) 
		ids_list = parent().fetch('charucoIdsAccum', [])

		if not corners_list or not ids_list:
			print('No corner points have been saved: these are necessary to calibrate the camera - exiting')
			return 

		# Calibrate the camera using the data we have gathered thus far
		# 
		# You can optionally set `flags=cv2.CALIB_USE_INTRINSIC_GUESS` and pass a camera matrix
		#
		# This function will return:
		# 1. The reprojection error
		# 2. The camera intrinsic matrix
		# 3. The camera distortion coefficients
		# 4. The rotation vector (axis+angle) of each of the poses that was used
		# 5. The translation vector of each of the poses that was used
		ret, K, dist_coeff, rvecs, tvecs = cv2.aruco.calibrateCameraCharuco(corners_list, 
																		    ids_list, self.board,
																		    (self.CameraRes[0], self.CameraRes[1]),
																		    None, 
																		    None) 
		parent().store('camera_intrinsics', {	
				'ret': ret,
				'K': K,
				'dist_coef': dist_coeff,
				'rvecs': rvecs,
				'tvecs': tvecs
		})	

		# Export .npy
		np.save(file_name, K)

		print('\tCamera calibration matrix:\n', K)
		print('\tCamera distortion coefficients:\n', dist_coeff.T)
		print('\tReprojection error:', ret)

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
		pixels = target_top.numpyArray()[:, :, :3] 
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

	def Display_pose(self, frame=None, file_name='display_pose'):
		"""
		Given the pose estimation of a marker or board, this function 
		draws the axis of the world coordinate system, i.e. the system 
		centered on the marker/board. Useful for debugging purposes.

		NOTE: requires a calibrated camera

		"""

		marker_corners, marker_ids, chessboard_corners, chessboard_id, frame = self.FindGrids(frame)
		
		camera_intrinsics = parent().fetch('camera_intrinsics', None)
		if not camera_intrinsics:
			print('Displaying a board pose requires a calibrated camera - exiting')
			return
		
		ret, rvec, tvec = aruco.estimatePoseBoard(marker_corners, 
											      marker_ids, 
											      self.board, 
											      camera_intrinsics['K'], 
											      camera_intrinsics['dist_coef'], 
											      None, # rvec initial guess: optional 
											      None) # tvec initial guess: optional
		aruco_marker_length_meters = 0.032
		frame = aruco.drawAxis( frame, camera_intrinsics['K'], camera_intrinsics['dist_coef'], rvec, tvec, aruco_marker_length_meters )
		
		# Save an image that shows the board's coordinate frame
		full_path = '{}/{}.png'.format(project.folder, file_name)
		cv2.imwrite(full_path, frame)


	def FindPose(self, frame=None):  
		""" 
		This function finds the camera pose based on the charuco pattern detected in the 
		current frame. 

		NOTE: This function requires a calibrated camera.

		returns :
			worldPos, retval, Srvec, Stvec

		"""
		# gridCorners, arucoIDs, arucoCorners = FindGrids()

		print("Finding camera pose...")

		if frame is None:
			frame = parent().GrabTop()

		marker_corners, marker_ids, chessboard_corners, chessboard_id, frame = self.FindGrids(frame)
		
		camera_intrinsics = parent().fetch('camera_intrinsics', None)

		if not camera_intrinsics:
			print('Displaying a board pose requires a calibrated camera - exiting')
			return
		
		ret, rvec, tvec = aruco.estimatePoseBoard(marker_corners, 
											      marker_ids, 
											      self.board, 
											      camera_intrinsics['K'], 
											      camera_intrinsics['dist_coef'], 
											      None, # rvec initial guess: optional 
											      None) # tvec initial guess: optional

		# Convert the axis+angle formulation into a 3x3 rotation matrix (the Jacobian is optional
		# and not really used here)
		rotation_matrix, jacobian = cv2.Rodrigues(rvec)

		# A 4x4 transformation matrix that transforms points from the board coordinate system 
		# to the camera coordinate system
		#
		# Reference: https://stackoverflow.com/questions/52833322/using-aruco-to-estimate-the-world-position-of-camera
		board_to_camera = np.matrix([[rotation_matrix[0][0], rotation_matrix[0][1], rotation_matrix[0][2], tvec[0][0]],
							         [rotation_matrix[1][0], rotation_matrix[1][1], rotation_matrix[1][2], tvec[1][0]],
							         [rotation_matrix[2][0], rotation_matrix[2][1], rotation_matrix[2][2], tvec[2][0]],
							         [0.0, 0.0, 0.0, 1.0]])

		# Invert the matrix above: this is the extrinsic matrix of the camera, i.e. the camera's
		# pose in a coordinate system relative to the board's origin
		camera_to_board = board_to_camera.I  

		# Get the position of the camera, in world space (this should be the same as -tvec): the last column
		camera_position = [camera_to_board[0, 3], 
						   camera_to_board[1, 3], 
						   camera_to_board[2, 3]]
		print('\tCamera position (in meters):', camera_position)
		print('\tCamera extrinsic matrix (pose):\n', camera_to_board)
		extrinsic_flat = np.copy(camera_to_board).flatten()
		tdu_extrinsic = tdu.Matrix(extrinsic_flat.tolist())
		tdu_extrinsic.fillTable(op('table_cam_ext'))


		# Construct an OpenGL-style projection matrix from an OpenCV-style projection
		# matrix
		#
		# References: 
		# [0] https://strawlab.org/2011/11/05/augmented-reality-with-OpenGL/
		# [1] https://fruty.io/2019/08/29/augmented-reality-with-opencv-and-opengl-the-tricky-projection-matrix/
		x0 = 0
		y0 = 0
		znear = 0.1
		zfar = 1000
		width = 1280
		height = 720
		K = camera_intrinsics['K']

		intrinsics = np.array([
			[ 2*K[0,0]/width, -2*K[0,1]/width,   (width - 2*K[0,2] + 2*x0)/width,     0],
			[ 0,               2*K[1,1]/height,  (-height + 2*K[1,2] + 2*y0)/height,  0],
			[ 0,               0,                (-zfar - znear)/(zfar - znear),     -2*zfar*znear/(zfar - znear)],
			[ 0,               0,               -1,                                   0]
		])
	
		tdu_intrinsic = tdu.Matrix(intrinsics.flatten().tolist())
		tdu_intrinsic.fillTable(op('table_cam_int'))













		'''
		matrix_comp = op('base_camera_matrix')

		# Fill rotation table
		cv_rotate_table = matrix_comp.op('cv_rotate')
		for i, v in enumerate(rvec):
			for j, rv in enumerate(v):
					cv_rotate_table[i, j] = rv

		# Fill translate vector
		cv_translate_table = matrix_comp.op('cv_translate')
		for i, v in enumerate(tvec.T):
			cv_translate_table[0, i] = v[0]

		projRez = (1920,1080)
		# Break appart camera matrix
		size = projRez
		# Computes useful camera characteristics from the camera matrix.
		fovx, fovy, focalLength, principalPoint, aspectRatio = cv2.calibrationMatrixValues(camera_intrinsics['K'], size, 1920, 1080)  # this is not right, last two arguments are supposed to be the aperature size in mm
		near = .1
		far = 2000

		# Create a new identity matrix
		extrinsic_matrix = tdu.Matrix()
		extrinsic_matrix.identity()
		# Convert `rvec` (which is a rotation specified by an axis+angle, used by OpenCV internally)
		# into a 3x3 rotation matrix
		rotation_matrix = np.eye(3)
		cv2.Rodrigues(rvec, rotation_matrix)
		# ----------- ^src  ^dst          
		# Rotate the extrinsic matrix - NOTE you're going to have to do some stuff here to convert
		# the 3x3 numpy array (`rotation_matrix`) into a tdu.Matrix first
		print(rotation_matrix)
		print(rotation_matrix.flatten().tolist())


		rotMatrix = tdu.Matrix(rotation_matrix.flatten().tolist())
		# extMatrix = tdu.Matrix(extrinsic_matrix.flatten().tolist())

		extrinsic = rotMatrix * extrinsic_matrix
		# Translate the extrinsic matrix by `tvec`
		extrinsic.translate(tvec[0], tvec[1], tvec[2])

		extrinsic.fillTable(op('table_cam_pose_ext'))


		camInternal = tdu.Matrix(camera_intrinsics['K'].flatten().tolist())
		camInternal.fillTable(op('table_cam_int'))

		return camera_position, retval, rvec, tvec

		'''










	



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