import numpy as np
import cv2
import matplotlib.pyplot as plt
import sys

from Calibration import Calibrate
from Filter import Filter
from PerspectiveTransform import PerspectiveTransform
from Thresholds import Thresholds
from Lane import Lane
from Line import Line
from process import ProcessStep
from moviepy.editor import VideoFileClip
from SlidingWindow import SlidingWindow
from TargettedLineSearch import TargettedLineSearch

class Pipeline(ProcessStep):

	def __init__(self, config=None):
		super().__init__()
		
		self.data = config

		self.frame_num = 0
		#self.debug_frame = config['debug_frame']
		self.image = None

		# Camera calibration
		calibration = Calibrate()

		# Load calibration data
		calibration.load()

		pt1 = PerspectiveTransform()
		pt2 = PerspectiveTransform(pt1)
		sw = SlidingWindow()
		tls = TargettedLineSearch()
		lane = Lane()
		b_gradient = Filter()

		self.steps = [calibration, b_gradient, pt1, sw, tls, lane, pt2]

		self.debug = config['debug']

		print('Pipeline object created ...')

	def invoke_steps(self, image):

		p_image = image
		self.data['image'] = p_image
		self.data['rawimage'] = image

		debug = self.data['debug']

		for step in self.steps:
			self.data = step.process(self.data)

			if (debug):
				step.visualize(self.data)

		image = cv2.addWeighted(image, 1, self.data['image'], 0.3, 0)

		font = cv2.FONT_HERSHEY_SIMPLEX
		cv2.putText(image,'F: %d'%(self.frame_num),(50,50), font, 1, (255,255,255),1,cv2.LINE_AA)
		
		lineobj = self.data['line']
		cv2.putText(image,'Curvature: %.2f m'%(lineobj.radius_of_curvature),(50,75), font, 1,(255,255,255),1,cv2.LINE_AA)
		cv2.putText(image,'Vehicle center offset: %.2f m'%(lineobj.center_offset),(50,100), font, 1,(255,255,255),1,cv2.LINE_AA)


		self.image = image
		self.data['image'] = image

		if (debug):
			self.visualize(self.data)

		return image

	def process_image(self, image):

		if (self.debug):

			if (self.data['filetype'] == 'mp4'):
				debug_frame = self.data['debug_frame']

				if (self.frame_num == debug_frame):
					self.image = self.invoke_steps(image)
				else:
					self.image = image
			else:
				self.image = self.invoke_steps(image)

		else:
			self.image = self.invoke_steps(image)

		self.frame_num += 1

		return self.image


def create_video(input_video_file, output_video_file, pipeline=None, config=None):

	## To speed up the testing process you may want to try your pipeline on a shorter subclip of the video
	## To do so add .subclip(start_second,end_second) to the end of the line below
	## Where start_second and end_second are integer values representing the start and end of the subclip
	## You may also uncomment the following line for a subclip of the first 5 seconds
	keys = config.keys()
	if (('start_second' in keys) and ('end_second' in keys)):
		start = config['start_second']
		end = config['end_second']

		print('Start clip:', start, ' End clip:', end)
		clip1 = VideoFileClip(input_video_file).subclip(start, end)

	else:
		clip1 = VideoFileClip(input_video_file)

	proj_clip = clip1.fl_image(pipeline.process_image) #NOTE: this function expects color images!!
	
	proj_clip.write_videofile(output_video_file , audio=False)

# main entry point
def main(input_file, config=None):

	input = input_file.split('.')
	
	# add a Line object into pipeline data 
	config['line'] = Line()
	config['filetype'] = input[-1]

	# create a pipeline to process images
	pipeline = Pipeline(config)

	if (input[-1] == 'jpg'):
		print('Will process image ', input_file)
		image = cv2.imread(input_file)
		image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
		
		result = pipeline.process_image(image)
	
	elif (input[-1] == 'mp4'):
		print('Will process video ', input_file)
		input_video = input[-2].split('/')[1]
		output_video = 'my-' + input_video + '.mp4'
		print('input :', input_video)
		print('output:', output_video)

		create_video(input_file, output_video, pipeline, config=config)
	

if __name__ == "__main__":

	if (len(sys.argv) > 1):
		config = {}
		with open('run.conf','r') as inf:
			config = eval(inf.read())
			print('CONFIG:', config)

		if (config['debug']):
			print('In DEBUG mode')

		main(sys.argv[1], config=config)
		
	else:
		print('First argument should be the image file!')
