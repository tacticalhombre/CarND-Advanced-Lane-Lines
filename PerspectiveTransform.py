import numpy as np
import matplotlib.pyplot as plt
import cv2
from process import ProcessStep

class PerspectiveTransform(ProcessStep):

	def __init__(self, perspective=None):
		super().__init__()
		print('PerspectiveTransform object created ...')
		self.image = None

		if (perspective == None):
			"""
			self.src = np.float32([	[586, 456],
								[699, 456],
								[1029, 668],
								[290, 668]])

			self.dst = np.float32([	[300, 70],
								[1000, 70],
								[1000, 600],
								[300, 600]])
			"""
			"""
			self.src = np.float32([	[570, 456],
									[750, 456],
									[1200, 700],
									[200, 700]])

			self.dst = np.float32([	[400, 70],
									[1000, 70],
									[900, 600],
									[500, 600]])
			"""
			w,h = 1280,720
			x,y = 0.5*w, 0.8*h
			self.src = np.float32([[200./1280*w,720./720*h],
									[453./1280*w,547./720*h],
									[835./1280*w,547./720*h],
									[1100./1280*w,720./720*h]])
			self.dst = np.float32([[(w-x)/2.,h],
									[(w-x)/2.,0.82*h],
									[(w+x)/2.,0.82*h],
									[(w+x)/2.,h]])
			# Given src and dst points, calculate the perspective transform matrix
			self.M = cv2.getPerspectiveTransform(self.src, self.dst)

		else:
			# inverse transform
			self.src = perspective.dst
			self.dst = perspective.src
			self.M = cv2.getPerspectiveTransform(self.src, self.dst)
			self.set_suffix('_inv')

	def warp(self, image, src, dst):
		img_size = (image.shape[1], image.shape[0])
		
		# Warp the image using OpenCV warpPerspective()
		warped = cv2.warpPerspective(image, self.M, img_size, flags=cv2.INTER_LINEAR)

		return warped

	def process(self, data):
		image = data['image']
		self.image = self.warp(image, self.src, self.dst)
		data['image'] = self.image

		return data

	def visualize(self, data):
		super(PerspectiveTransform, self).visualize(data)
		# Create a black image
		img = np.zeros((1280,720,3), np.uint8)

		# Draw a polygon
		"""
		src = np.array([[570, 456],
						[750, 456],
						[1200, 700],
						[200, 700]], np.int32)
		"""

		#src = np.array([[540, 470],[750, 470],[1130, 690],[200, 690]], np.int32)
		"""
		dst = np.array([[400, 70],
						[1000, 70],
						[900, 600],
						[500, 600]], np.int32)
		"""
		w,h = 1280,720
		x,y = 0.5*w, 0.8*h
		"""
		src = np.array([[200./1280*w,720./720*h],
								[453./1280*w,547./720*h],
								[835./1280*w,547./720*h],
								[1100./1280*w,720./720*h]])
		dst = np.array([[(w-x)/2.,h],
								[(w-x)/2.,0.82*h],
								[(w+x)/2.,0.82*h],
								[(w+x)/2.,h]])
		"""
		src = np.array([[200, 720],
		 			[453, 547],
		 			[835, 547],
		 			[1100, 720]], np.int32)

		dst = np.array([[320, 720],
		 			[320, 590],
		 			[960, 590],
		 			[960, 720]], np.int32)		
		
		src = src.reshape((-1,1,2))
		dst = dst.reshape((-1,1,2))

		img = data['rawimage']

		img = img.copy()
		cv2.polylines(img,[src],True,(0,255,255), 2)
		
		#cv2.polylines(img,[dst],True,(0,0,255), 2)

		#f, ax = plt.subplots()
		#plt.imshow(img)
		

		f, (ax1, ax2) = plt.subplots(1, 2, figsize=(25, 9))
		f.tight_layout()
		ax1.imshow(img)
		ax1.set_title('Undistorted image', fontsize=25)
		ax2.imshow(data['image'])
		ax2.set_title('Warped image', fontsize=25)
		plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
		f.savefig('./output_images/vis-' + type(self).__name__ + self.get_suffix() + '-src.png')		
		

def main():
	pt = PerspectiveTransform()

if __name__ == "__main__":
	main()