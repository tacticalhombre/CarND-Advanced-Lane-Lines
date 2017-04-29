import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import glob
import pickle
from process import ProcessStep

class Calibrate(ProcessStep):
	
	fn_draw_chessboard = './output_images/draw_chessboard.jpg' 
	
	def __init__(self):
		super().__init__()
		print('Calibrate object created ...')
		self.step_name = 'calibration'

		# 3D in the real world space
		self.objpoints = [] 

		# 2D in image plane
		self.imgpoints = []

		self.data_file = 'calibrate.p'

		self.image = None
		
		self.mtx = None
		self.dist = None

		self.gray = None
		self.data = None

	# Creates object points, and image points
	# nx - the number of inside corners in x
	# ny - the number of inside corners in y
	def get_image_points(self, nx, ny, file_pattern):

		image_files = glob.glob(file_pattern)

		for fn in image_files:

			# read image of chessboard used for calibration
			image = cv2.imread(fn)

			objp = np.zeros((nx*ny, 3), np.float32)
			objp[:,:2] = np.mgrid[0:nx, 0:ny].T.reshape(-1, 2)

			# convert to grayscale
			self.gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

			# Finding chessboard corners (for an nx x ny board):
			ret, corners = cv2.findChessboardCorners(self.gray, (nx,ny), None)
			#print('corners: ', corners)

			# If found, draw corners
			if ret == True:
				print('Found corners for image ', fn)

				self.imgpoints.append(corners)
				self.objpoints.append(objp)

			else:
				print('Corners not found for image ', fn)

			print()

		return self.objpoints, self.imgpoints


	def save(self):
		print('Saving data into file', self.data_file)
		data = {'objpoints': self.objpoints, 'imgpoints': self.imgpoints}
		pickle.dump( data, open( self.data_file, "wb" ) )

	def load(self):
		print('Loading data from', self.data_file)

		with open(self.data_file, mode='rb') as f:
			data = pickle.load(f)

			self.objpoints = data['objpoints']
			self.imgpoints = data['imgpoints']


	# takes an image
	# performs the camera calibration, image distortion correction and 
	# returns the undistorted image
	def cal_undistort(self, image):
		self.gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

		if (self.mtx == None):
			# Camera calibration, given object points, image points, and the shape of the grayscale image:
			ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(self.objpoints, self.imgpoints, self.gray.shape[::-1], None, None)
			self.mtx = mtx
			self.dist = dist

		# Undistorting an image:
		undistorted = cv2.undistort(image, self.mtx, self.dist, None, self.mtx)
	    
		return undistorted

	def process(self, data):
		self.data = data
		image = data['image']
		orig = image.copy()
		self.image = self.cal_undistort(image)
		data['image'] = self.image

		debug = data['debug']

		if (debug):
			f, (ax1, ax2) = plt.subplots(1, 2, figsize=(25, 9))
			f.tight_layout()
			ax1.imshow(orig)
			ax1.set_title('Distorted image', fontsize=25)
			ax2.imshow(self.image)
			ax2.set_title('Undistorted image', fontsize=25)
			plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
			f.savefig('./output_images/vis-' + type(self).__name__ + '-undistort.png')	
		return data

	

def main():

	cal = Calibrate()

	cal.get_image_points(9, 6, './camera_cal/calibration*.jpg')
	cal.get_image_points(9, 5, './camera_cal/calibration1.jpg')

	cal.get_image_points(8, 6, './camera_cal/calibration4.jpg')
	cal.get_image_points(7, 6, './camera_cal/calibration5.jpg')

	cal.save()

	newcal = Calibrate()
	newcal.load()


if __name__ == "__main__":
	main()