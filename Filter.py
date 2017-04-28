import cv2
import numpy as np
import matplotlib.pyplot as plt

from Thresholds import Thresholds
from process import ProcessStep

class Filter(ProcessStep):
	def __init__(self):
		super().__init__()
		print('Filter created ...')
		self.threshold = Thresholds()
		self.image = None

	def apply_all(self, image):
		ksize = 3

		gradx = self.threshold.abs_sobel_thresh(image, orient='x', sobel_kernel=ksize, thresh=(20, 100))
		grady = self.threshold.abs_sobel_thresh(image, orient='y', sobel_kernel=ksize, thresh=(20, 100))
		mag_binary = self.threshold.mag_thresh(image, sobel_kernel=9, mag_thresh=(30, 100))
		dir_binary = self.threshold.dir_threshold(image, sobel_kernel=15, thresh=(0.7, 1.3))
		hls_t = self.threshold.hls_select(image, (90, 255))
		
		combined = np.zeros_like(gradx)
		combined[((gradx == 1) & (grady == 1)) | ((mag_binary == 1) & (dir_binary == 1)) | (hls_t == 1)] = 1

		return combined

	def apply(self, image):
		combined = self.apply_all(image)

		return combined

	def process(self, data):

		
		image = data['image']
		self.image = self.apply(image)
		data['image'] = self.image

		return data

def main():
	f = Filter()
	

if __name__ == "__main__":
	main()