import cv2
import numpy as np
import matplotlib.pyplot as plt
import math
import glob

from Thresholds import Thresholds
from process import ProcessStep


class Filter(ProcessStep):
	def __init__(self):
		super().__init__()
		print('Filter created ...')
		self.threshold = Thresholds()
		self.image = None
		self.data = None

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
		ksize = 3
		lab = self.threshold.cielab_select(image, (141, 255))
		luv = self.threshold.cieluv_select(image, (215, 255))

		hls_t = self.threshold.hls_select(image, (90, 255))

		gradx = self.threshold.abs_sobel_thresh(image, orient='x', sobel_kernel=ksize, thresh=(20, 100))
		grady = self.threshold.abs_sobel_thresh(image, orient='y', sobel_kernel=ksize, thresh=(20, 100))

		combined = np.zeros_like(lab)
		combined[(lab == 1) | (luv == 1) | ((gradx == 1) & (grady == 1))] = 1

		return combined

	def process(self, data):
		self.data = data
		
		image = data['image']
		self.image = self.apply(image)
		data['image'] = self.image

		return data


def main():
	myfilter = Filter()

	image_files = glob.glob('./test_images/*.jpg')

	ncol = 2
	n = len(image_files)
	r = math.ceil(n/ncol)

	fig, axarr = plt.subplots(r, ncol, figsize=(30, 35))
	plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
	fig.tight_layout()
	z = 0
	for i in range(r):
		for j in range(ncol):

			if (z>=n):
				break

			fn = image_files[z]

			image = cv2.imread(fn)
			image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

			new_image = myfilter.apply(image)

			axarr[i, j].imshow(new_image)
			axarr[i, j].set_title(fn, fontsize=30)
			axarr[i, j].set_xticks([])
			axarr[i, j].set_yticks([])
			
			z+=1


	fig.savefig('./examples/vis-'  + 'Filter.png')

if __name__ == "__main__":
	main()