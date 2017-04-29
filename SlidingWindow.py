import numpy as np
import cv2
import matplotlib.pyplot as plt
from process import ProcessStep
from Line import Line
from TargettedLineSearch import TargettedLineSearch

class SlidingWindow(ProcessStep):
	
	def __init__(self):
		super().__init__()
		print('SlidingWindow object created ...')
		self.line = Line()
		self.binary_warped = None
		self.image = None
		self.out_img = None
		self.histogram = None
		self.debug = False
		self.data = None

	def get_line(self):
		return self.line

	def find_lines(self, binary_warped):
		self.binary_warped = binary_warped
		# Assuming you have created a warped binary image called "binary_warped"
		# Take a histogram of the bottom half of the image
		self.histogram = np.sum(binary_warped[binary_warped.shape[0]/2:,:], axis=0)

		# Create an output image to draw on and  visualize the result
		out_img = np.dstack((binary_warped, binary_warped, binary_warped))*255
		
		# Find the peak of the left and right halves of the histogram
		# These will be the starting point for the left and right lines
		midpoint = np.int(self.histogram.shape[0]/2)
		leftx_base = np.argmax(self.histogram[:midpoint])
		rightx_base = np.argmax(self.histogram[midpoint:]) + midpoint

		# Choose the number of sliding windows
		nwindows = 9
		
		# Set height of windows
		window_height = np.int(binary_warped.shape[0]/nwindows)
		
		# Identify the x and y positions of all nonzero pixels in the image
		nonzero = binary_warped.nonzero()
		nonzeroy = np.array(nonzero[0])
		nonzerox = np.array(nonzero[1])
		
		# Current positions to be updated for each window
		leftx_current = leftx_base
		rightx_current = rightx_base
		
		# Set the width of the windows +/- margin
		margin = 100
		
		# Set minimum number of pixels found to recenter window
		minpix = 50
		
		# Create empty lists to receive left and right lane pixel indices
		left_lane_inds = []
		right_lane_inds = []

		# Step through the windows one by one
		for window in range(nwindows):
		    # Identify window boundaries in x and y (and right and left)
		    win_y_low = binary_warped.shape[0] - (window+1)*window_height
		    win_y_high = binary_warped.shape[0] - window*window_height
		    win_xleft_low = leftx_current - margin
		    win_xleft_high = leftx_current + margin
		    win_xright_low = rightx_current - margin
		    win_xright_high = rightx_current + margin
		    
		    # Draw the windows on the visualization image
		    cv2.rectangle(out_img,(win_xleft_low,win_y_low),(win_xleft_high,win_y_high),(0,255,0), 2) 
		    cv2.rectangle(out_img,(win_xright_low,win_y_low),(win_xright_high,win_y_high),(0,255,0), 2) 
		    
		    # Identify the nonzero pixels in x and y within the window
		    good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
		    good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]
		    
		    # Append these indices to the lists
		    left_lane_inds.append(good_left_inds)
		    right_lane_inds.append(good_right_inds)
		    
		    # If you found > minpix pixels, recenter next window on their mean position
		    if len(good_left_inds) > minpix:
		        leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
		    if len(good_right_inds) > minpix:        
		        rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

		# Concatenate the arrays of indices
		left_lane_inds = np.concatenate(left_lane_inds)
		right_lane_inds = np.concatenate(right_lane_inds)

		# Extract left and right line pixel positions
		leftx = nonzerox[left_lane_inds]
		lefty = nonzeroy[left_lane_inds] 
		rightx = nonzerox[right_lane_inds]
		righty = nonzeroy[right_lane_inds] 

		#out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
		#out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]
		#plt.imshow(out_img)

		# Fit a second order polynomial to each
		left_fit = np.polyfit(lefty, leftx, 2)
		right_fit = np.polyfit(righty, rightx, 2)

		if (self.debug):
			self.leftx = leftx
			self.rightx = rightx

			# code for plotting
			# Generate x and y values for plotting
			
			ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0] )
			left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
			right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]

			#print('L Xs:', left_fitx)
			f, ax = plt.subplots()
			out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
			out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]
			plt.imshow(out_img)
			plt.plot(left_fitx, ploty, color='yellow')
			plt.plot(right_fitx, ploty, color='yellow')
			plt.xlim(0, 1280)
			plt.ylim(720, 0)
			f.savefig('./output_images/vis-' + type(self).__name__ + '-F' + str(self.data['frame_num']) + '-fitline.png')

		self.out_img = out_img
		
		return left_fit, right_fit

	
	def vis(self, lines):
		binary_warped = self.binary_warped

		f, (ax1, ax2) = plt.subplots(1, 2, figsize=(25, 9))
		f.tight_layout()
		ax1.imshow(binary_warped)
		ax1.set_title('Warped image', fontsize=25)
		plt.plot(self.histogram)
		ax2.set_title('Histogram', fontsize=25)
		plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
		f.savefig('./output_images/vis-' + type(self).__name__ + '-F' + str(self.data['frame_num']) + '-histogram.png')		

		left_fit = lines[0]
		right_fit = lines[1]

		# Generate x and y values for plotting
		ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0] )
		left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
		right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]

		# Identify the x and y positions of all nonzero pixels in the image
		nonzero = binary_warped.nonzero()
		nonzeroy = np.array(nonzero[0])
		nonzerox = np.array(nonzero[1])

		"""
		f, ax = plt.subplots()
		plt.plot(left_fitx, ploty, color='yellow')
		plt.plot(right_fitx, ploty, color='yellow')
		plt.xlim(0, 1280)
		plt.ylim(720, 0)
		f.savefig('./output_images/vis-' + type(self).__name__ + '-F' + str(self.data['frame_num']) + '-fit.png')	
		"""
	
	def visualize(self, data):
		super(SlidingWindow, self).visualize(data)

		lines = data['lines']
		self.vis(lines)


	def process(self, data):
		self.data = data
		self.debug = data['debug']

		if (not data['line'].detected):
			print('Finding windows ...')
			image = data['image']
			lines = self.find_lines(image)
			data['lines'] = lines
			#self.image = self.draw_lines(lines)
			#self.image = self.vis(lines)
			self.image = self.out_img
			
			data['line'].detected = True

			lineobj = data['line']

			#print('sw-lastn_left:', lineobj.lastn_left)
			#print('sw-lastn_right:', lineobj.lastn_right)

			
			x0 = np.concatenate((lineobj.lastn_left, [lines[0]]), axis=0)
			x1 = np.concatenate((lineobj.lastn_right, [lines[1]]), axis=0)

			lineobj.lastn_left = x0
			lineobj.lastn_right = x1

			print(type(self).__name__ + ' - ' + str(data['frame_num']), ' lines: ', data['lines'])
		
		return data

