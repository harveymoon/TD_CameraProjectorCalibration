'''
Major props to Alvaro Cassinelli & Niklas BergstrÃ¶m:
https://www.youtube.com/watch?v=pCq7u2TvlxU

Also, Cyril Diagne's openFrameworks toolkit:
https://github.com/cyrildiagne/ofxCvCameraProjectorCalibration

(Re)coded by Harvey Moon, Satoru Higa, Michael Walczyk for 
use within Derivative's TouchDesigner software

'''
import numpy as np
import cv2 
aruco = cv2.aruco 

def intersect_circle_rays_to_board(circles, rvec, t, K, dist_coef):
	'''
	A small helper function for projecting the 2D circles onto the grid's plane

	'''
	circles_normalized = cv2.convertPointsToHomogeneous(cv2.undistortPoints(circles, K, dist_coef))
	if not rvec.size:
		return None

	R, _ = cv2.Rodrigues(rvec)
 
	# https://stackoverflow.com/questions/5666222/3d-line-plane-intersection
	plane_normal = R[2,:] # Last row of plane rotation matrix is normal to plane
	plane_point = t.T     # `t` is a point on the plane
	epsilon = 1e-06
	circles_3d = np.zeros((0,3), dtype=np.float32)
 
	for p in circles_normalized:
		ray_direction = p / np.linalg.norm(p)
		ray_point = p
 
		ndotu = plane_normal.dot(ray_direction.T)
 
		if abs(ndotu) < epsilon:
			print('No intersection found (or the line is collinear with the plane)')
 
		w = ray_point - plane_point
		si = -plane_normal.dot(w.T) / ndotu
		psi = w + si * ray_direction + plane_point
		circles_3d = np.append(circles_3d, psi, axis = 0)
 
	return circles_3d

def cv_to_gl_projection_matrix(K, w, h, znear=0.1, zfar=1000.0):
	# Construct an OpenGL-style projection matrix from an OpenCV-style projection
	# matrix
	#
	# References: 
	# [0] https://strawlab.org/2011/11/05/augmented-reality-with-OpenGL/
	# [1] https://fruty.io/2019/08/29/augmented-reality-with-opencv-and-opengl-the-tricky-projection-matrix/
	x0 = 0
	y0 = 0

	i00 = 2 * K[0, 0] / w
	i01 = -2 * K[0, 1] / w
	i02 = (w - 2 * K[0, 2] + 2 * x0) / w
	i03 = 0

	i10 = 0
	i11 = 2 * K[1, 1] / h
	i12 = (-h + 2 * K[1, 2] + 2 * y0) / h
	i13 = 0

	i20 = 0
	i21 = 0
	i22 = (-zfar - znear) / (zfar - znear)
	i23 = -2 * zfar * znear / (zfar - znear)

	i30 = 0
	i31 = 0
	i32 = -1
	i33 = 0

	return np.array([
		[i00, i01, i02, i03],
		[i10, i11, i12, i13],
		[i20, i21, i22, i23],
		[i30, i31, i32, i33]
	])

class StereoCalibrate:
	'''
	Stereo calibration consists of 3 steps:

	1. Calibrate the camera based on a charuco board pattern
	2. Calibrate the projector based on a projected circle pattern (seen by the camera)
	3. Create a stereo calibration between the two lenses based on all of the information
	   gathered in the previous two steps

	Generally speaking, as a user of this class you will:
	
	1. Call `CaptureFrame()` multiple times (8 or so)
	2. Call `CalibrateCam()` to get the camera's intrinsic matrix
	3. Call `CreateUndistortionMap()` to get a UV un-distortion map (texture)
	4. Call `FindPose()` to find the camera's pose, given any new boards
	
	'''
	def __init__(self, COMP_owner):
		self.COMP_owner = COMP_owner
		print('Initialized the stereo calibration class')
		
		# The TOP that we will grab video frames from
		self.TOP_frame_input = op('null_frame')

		# The number of chessboard squares along each axis of the charuco board
		self.Chessboards_x = 12
		self.Chessboards_y = 8

		# The number of circles along each axis of the circle pattern
		self.Circles_x = 7
		self.Circles_y = 6

		# This will be set whenever the first camera frame is captured
		self.Camera_resolution = (1280, 720) 

		# The minimum number of captured board images required before we can calibrate the camera
		self.Minimum_boards_required = 8

		# Create the aruco board (CV Mat)
		self.Aruco_dictionary = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
		self.Aruco_board = cv2.aruco.CharucoBoard_create(self.Chessboards_x, self.Chessboards_y, 0.035, 0.0175, self.Aruco_dictionary)

	def SaveBoard(self, w=2000, h=1300, file_name='charuco_board'):
		'''
		This function will save a local .jpg copy of the requested Arcuo board. 
		If you have not printed one out already, you will need to do so.

		'''
		drawn_board = self.Aruco_board.draw((w, h))
		cv2.imwrite('{}.jpg'.format(file_name), drawn_board)

	def ClearSets(self):
		'''
		Clear out the saved camera features.

		'''
		parent().store('chessboard_corners_accum', [])
		parent().store('chessboard_ids_accum', [])
		parent().par.Capturedsets = 0
		parent().par.Calibratecam.enable = False
		parent().par.Findpose.enable = False

	def GrabTop(self, needs_grayscale_conversion=False):
		'''
		Grab a frame from a TOP within the network and return a grayscale CV Mat output. 

		'''
		# Convert input frame to a numpy array from 0-255
		pixels = self.TOP_frame_input.numpyArray()[:, :, :3] * 255.0
		
		# Convert the pixel data to a CV Mat object
		cv_img = pixels.astype(np.uint8)
		
		# Need to flip Y because of how TouchDesigner stores pixel data
		cv_img = cv2.flip(cv_img, 0)
		
		# Convert to grayscape (if Mono TOP isn't in the network?)
		if needs_grayscale_conversion:
			cv_img = cv2.cvtColor(cv_img, cv2.COLOR_RGB2GRAY)

		# Set the camera resolution
		self.Camera_resolution = cv_img.shape[:2]

		return cv_img

	def CaptureFrame(self):
		'''
		Attempts to find the charuco board pattern in the current camera frame - if found,
		the resulting chessboard corners and IDs are saved into storage for calibration.
		
		'''
		print('Capturing camera frame...')

		# Fetch lists 
		corners_list = parent().fetch('chessboard_corners_accum', []) 
		ids_list = parent().fetch('chessboard_ids_accum', [])

		# Find the aruco corners (markers and chessboard)
		found, corners, ids, chessboard_corners, chessboard_ids, frame = parent().FindGrids()
		
		if found:
			# Accumulate
			corners_list += [chessboard_corners]				
			ids_list += [chessboard_ids]

			# Set the custom par that shows the user how many views have been captured
			number_charuco_views = len(ids_list)
			parent().par.Capturedsets = number_charuco_views

			if number_charuco_views >= self.Minimum_boards_required:
				parent().par.Calibratecam.enable = True
				parent().par.Findpose.enable = False

	def FindGrids(self, frame=None):
		'''
		A charuco board consists of feducial markers overlaid on top of a chessboard
		grid. Per the OpenCV docs, the benefit of using these patterns is that they 
		provide the versatility of aruco boards + the precision of chessboards.

		This function finds such a grid in the input frame and returns 3 items:
		
		0. The status (whether or not any markers were actually detected)
		1. The detected marker corners
		2. The IDs of the marker corners that were found
		3. The chessboard corners (interpolated based on item 1)
		4. The IDs of the chessboard corners that were found
		5. The OpenCV Mat (frame) that was used for detection

		'''
		print('Finding charuco board...')
	
		# If a frame was not supplied, grab one from the TOP 
		if frame is None:
			frame = parent().GrabTop()
		
		DAT_ids = op('base_draw_found_grid/geo_aruco_view/table_ids')
		DAT_ids.clear(keepFirstRow = True)
		DAT_quads = op('base_draw_found_grid/geo_aruco_view/table_quads')
		DAT_quads.clear(keepFirstRow = False)
		DAT_corners = op('base_draw_found_grid/geo_aruco_view/table_corners')
		DAT_corners.clear(keepFirstRow = True)

		# First, detect the markers - this function returns 3 things:
		# 1. corners: a vector of detected marker corners (for each marker,
		#    its four corners are provided)
		# 2. ids: a vector of identifiers of the detected markers (ints)
		# 3. rejected image points: the image points of those squares whose 
		#    inner code doesn't have a correct codification
		marker_corners, marker_ids, rejected_img_pts = aruco.detectMarkers(frame, self.Aruco_dictionary)
		aruco.drawDetectedMarkers(frame, marker_corners, marker_ids, (0, 255, 0))
		
		# Next, refine the markers that weren't detected in the previous step based 
		# on the already detected markers and the known board layout
		# 
		# NOTE: you can optionally supply a camera matrix and distortion coefficients
		# to this function, which may help improve accuracy?
		marker_corners, marker_ids, rejected, recovered = cv2.aruco.refineDetectedMarkers(frame, self.Aruco_board, marker_corners, marker_ids, rejected_img_pts)  
		print('\t{} marker corners found (after refinement)'.format(len(marker_corners)))

		# Maybe save the results to disk, if needed
		save_to_disk = False 
		if save_to_disk:
			file_name = project.folder + '/find_grids_output.jpg'
			cv2.imwrite(file_name, frame)

		# Put corner data into a DAT so that we can draw each chessboard element
		total_points = 1

		for corner in marker_corners:
			# Quad indices
			quad = ''

			# Flattened 1D array of the corner vertex coordinates: we have to do this
			# because for some reason, OpenCV sometimes returns a list with different
			# numbers of nested levels
			coords = corner.flatten()

			for index in range(len(coords) // 2):
				DAT_ids.appendRow([coords[index*2 + 0], coords[index*2 + 1]])
				quad = quad + ' ' + str(total_points)
				total_points += 1

			DAT_quads.appendRow([quad, 1])

		# Finally, find the chessboard corners based on the information gathered above
		if marker_corners == None or len(marker_corners) == 0:
			print('\tNo marker corners were detected - exiting')
			return False, [], [], [], [], frame 

		else:
			# Find the position of the chessboard corners based on the (now known)
			# marker positions using a local homography (you can optionally provide
			# a camera matrix if it is known)
			ret, chessboard_corners, chessboard_ids = cv2.aruco.interpolateCornersCharuco(marker_corners, marker_ids, frame, self.Aruco_board)
			
			if len(chessboard_corners) == 0:
				print('\tNo chessboard corners were found (after interpolation) - exiting')
				return False, [], [], [], [], frame 

			else: 
				# Otherwise, some corners were found 
				for corner in chessboard_corners:
					DAT_corners.appendRow(corner[0])
				
				print('\t{} chessboard corners found (after interpolation)'.format(len(chessboard_corners)))

				return True, marker_corners, marker_ids, chessboard_corners, chessboard_ids, frame

	def CalibrateCam(self, file_name_intrinsics='camera_intrinsics', file_name_dist_coeffs='camera_dist_coeffs'):
		'''
		Based on the captured charuco boards, calculate the camera's intrinsic
		matrix and distortion coefficients.
		
		'''
		print('Calibrating the camera...')
		corners_list = parent().fetch('chessboard_corners_accum', []) 
		ids_list = parent().fetch('chessboard_ids_accum', [])

		if not corners_list or not ids_list:
			print('No corner points have been saved: these are necessary to calibrate the camera - exiting')
			return 

		if len(ids_list) < self.Minimum_boards_required:
			print('Need more boards before calibration - exiting')
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
																		    ids_list, self.Aruco_board,
																		    (self.Camera_resolution[0], self.Camera_resolution[1]),
																		    None, 
																		    None) 
		parent().store('camera_intrinsics', {	
				'ret': ret,
				'K': K,
				'dist_coeff': dist_coeff,
				'rvecs': rvecs,
				'tvecs': tvecs
		})	

		# Export .npy
		np.save(file_name_intrinsics, K)
		np.save(file_name_dist_coeffs, dist_coeff)
		print('\tSaved camera intrinsics:', file_name_intrinsics)
		print('\tSaved camera distortion coefficients:', file_name_dist_coeffs)
		print('\tCamera calibration matrix:\n', K)
		print('\tCamera distortion coefficients:\n', dist_coeff.T)
		print('\tReprojection error:', ret)

		parent().par.Calibratecam.enable = True
		parent().par.Findpose.enable = True

	def CreateUndistortionMap(self, file_name_undistortion_map='undistortion_map'):
		'''
		Creates a UV map that can be used in combination with a Remap TOP to 
		undistort subsequent camera images.

		NOTE: This function requires a calibrated camera.

		'''
		TOP_uv = op('glsl_generate_uv_map')
		pixels = TOP_uv.numpyArray()[:, :, :3]
		cv_img = pixels.astype(np.float32)
		cv_img = pixels
		cv_img = cv2.flip(cv_img, 0)
		cv_img = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB) # Convert BGR to RGB
		h, w = cv_img.shape[:2]

		if (w, h) != self.Camera_resolution:
			raise Exception('Attempting to create an undistortion map with different resolution than camera - exiting') 

	
		# Compute the optimal new camera matrix based on the free scaling parameter, alpha - 
		# note that if alpha = 0, the image will be cropped
		alpha = 0
		camera_intrinsics = parent().fetch('camera_intrinsics', None)
		if not camera_intrinsics:
			raise Exception('Creating an undistortion map requires a calibrated camera - exiting')
			
		K_optimal, valid_pixels_ROI = cv2.getOptimalNewCameraMatrix(camera_intrinsics['K'], 
																    camera_intrinsics['dist_coeff'], 
																    (w, h), 
																    alpha, 
																    (w, h))

		# Compute the undistortion and rectification transformation map
		mapx, mapy = cv2.initUndistortRectifyMap(camera_intrinsics['K'], 
												 camera_intrinsics['dist_coeff'], 
												 None, 
												 K_optimal, 
												 (w, h), 
												 cv2.CV_32FC1)
		
		# Remap and (optionally) crop the image
		x, y, w, h = valid_pixels_ROI
		remapped = cv2.remap(cv_img, mapx, mapy, cv2.INTER_LINEAR)
		remapped = remapped[y : y + h, x : x + w, :]
		
		# Save out the file
		cv2.imwrite('{}.exr'.format(file_name_undistortion_map), remapped)
		
		# Update storage 
		camera_intrinsics['K_optimal'] = K_optimal
		parent().store('camera_intrinsics', camera_intrinsics)

	def FindPose(self, frame=None, file_name_pose=None):  
		''' 
		Find the camera pose based on the charuco pattern detected in the current frame. 

		NOTE: This function requires a calibrated camera.

		'''
		print("Finding camera pose...")

		# Finding the camera pose requires a calibrated camera
		camera_intrinsics = parent().fetch('camera_intrinsics', None)
		if not camera_intrinsics:
			print('Finding the camera pose requires a calibrated camera - exiting')
			return

		if frame is None:
			frame = parent().GrabTop()

		# First, find the marker corners
		found, marker_corners, marker_ids, _, _, frame = self.FindGrids(frame)
		if not found:
			print('No board was found - cannot find pose')
			return 

		# Then, estimate the board pose
		ret, rvec, tvec = aruco.estimatePoseBoard(marker_corners, 
											      marker_ids, 
											      self.Aruco_board, 
											      camera_intrinsics['K'], 
											      camera_intrinsics['dist_coeff'], 
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

		# Convert the OpenCV projection matrix to one that can be used by OpenGL
		to_opengl = cv_to_gl_projection_matrix(camera_intrinsics['K'], self.Camera_resolution[1], self.Camera_resolution[0])
		tdu_intrinsic = tdu.Matrix(to_opengl.flatten().tolist())
		tdu_intrinsic.fillTable(op('table_cam_int'))

		# Save an image that shows the board's coordinate frame
		if file_name_pose:
			aruco_marker_length_meters = 0.032

			frame = aruco.drawAxis(frame, 
								   camera_intrinsics['K'], 
								   camera_intrinsics['dist_coeff'], 
								   rvec, 
								   tvec, 
								   aruco_marker_length_meters)
			
			cv2.imwrite('{}.png'.format(file_name_pose), frame)

		return camera_to_board, frame












	






	def FindCircleGrid(self):

		"""
		This function will discover the circle grids in camera space and then re-project the found locations back into 3d space using the calibrated camera. 
		"""

		camera_intrinsics = parent().fetch('camera_intrinsics')

		ret=camera_intrinsics['ret']
		K=camera_intrinsics['K']
		dist_coef=camera_intrinsics['dist_coeff']


		frame = parent().GrabTop()

		circleGridScale = (self.Circles_x, self.Circles_y) 

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
		circles3D = intersect_circle_rays_to_board(circles, rvec, tvec, K, dist_coef)
		 
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
		dist_coef=camera_intrinsics['dist_coeff']
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
		'dist_coeff': dist_coef_proj,
		'rvecs': rvecs,
		'tvecs': tvecs
		})	

		camera_intrinsics['K'] = K
		camera_intrinsics['dist_coeff'] = dist_coef

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