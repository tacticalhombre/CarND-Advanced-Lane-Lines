import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

class ProcessStep():

	def __init__(self):
		self.suffix = ''

	def set_suffix(self, sfx):
		self.suffix = sfx

	def get_suffix(self):
		return self.suffix

	def process(self, data):
		raise NotImplementedError()

	def visualize(self, data):
		vis_image = data['image']

		plt.imsave("./output_images/vis-" + type(self).__name__ + self.get_suffix() + '-F' + str(self.data['frame_num']) + '.png', vis_image)

	
		
