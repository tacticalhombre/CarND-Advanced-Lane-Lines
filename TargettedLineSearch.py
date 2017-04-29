import numpy as np
import cv2
import matplotlib.pyplot as plt
from process import ProcessStep
from Line import Line

class TargettedLineSearch(ProcessStep):
	
	def __init__(self):
		super().__init__()
		print('TargettedLineSearch object created ...')
		
		self.binary_warped = None
		self.image = None
		self.out_img = None
		self.debug = False
		self.data = None

	def search(self, binary_warped, data):
		self.binary_warped = binary_warped
		
		lines = data['lines']
		left_fit = lines[0]
		right_fit = lines[1]

		# Assume you now have a new warped binary image 
		# from the next frame of video (also called "binary_warped")
		# It's now much easier to find line pixels!
		nonzero = binary_warped.nonzero()
		nonzeroy = np.array(nonzero[0])
		nonzerox = np.array(nonzero[1])
		margin = 100
		left_lane_inds = ((nonzerox > (left_fit[0]*(nonzeroy**2) + left_fit[1]*nonzeroy + left_fit[2] - margin)) & (nonzerox < (left_fit[0]*(nonzeroy**2) + left_fit[1]*nonzeroy + left_fit[2] + margin))) 
		right_lane_inds = ((nonzerox > (right_fit[0]*(nonzeroy**2) + right_fit[1]*nonzeroy + right_fit[2] - margin)) & (nonzerox < (right_fit[0]*(nonzeroy**2) + right_fit[1]*nonzeroy + right_fit[2] + margin)))  

		# Again, extract left and right line pixel positions
		leftx = nonzerox[left_lane_inds]
		lefty = nonzeroy[left_lane_inds] 
		rightx = nonzerox[right_lane_inds]
		righty = nonzeroy[right_lane_inds]

		lineobj = data['line']
		lineobj.leftx = leftx
		lineobj.rightx = rightx

		# Fit a second order polynomial to each
		left_fit = np.polyfit(lefty, leftx, 2)
		right_fit = np.polyfit(righty, rightx, 2)
		return left_fit, right_fit 	

	def vis(self, lines):
		binary_warped = self.binary_warped
		left_fit = lines[0]
		right_fit = lines[1]

		# Create an image to draw on and an image to show the selection window
		out_img = np.dstack((binary_warped, binary_warped, binary_warped))*255
		window_img = np.zeros_like(out_img)

		nonzero = binary_warped.nonzero()
		nonzeroy = np.array(nonzero[0])
		nonzerox = np.array(nonzero[1])
		margin = 100
		left_lane_inds = ((nonzerox > (left_fit[0]*(nonzeroy**2) + left_fit[1]*nonzeroy + left_fit[2] - margin)) & (nonzerox < (left_fit[0]*(nonzeroy**2) + left_fit[1]*nonzeroy + left_fit[2] + margin))) 
		right_lane_inds = ((nonzerox > (right_fit[0]*(nonzeroy**2) + right_fit[1]*nonzeroy + right_fit[2] - margin)) & (nonzerox < (right_fit[0]*(nonzeroy**2) + right_fit[1]*nonzeroy + right_fit[2] + margin)))  


		# Color in left and right line pixels
		out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
		out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]

		# Generate x and y values for plotting
		ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0] )
		left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
		right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
		# Generate a polygon to illustrate the search window area
		# And recast the x and y points into usable format for cv2.fillPoly()
		left_line_window1 = np.array([np.transpose(np.vstack([left_fitx-margin, ploty]))])
		left_line_window2 = np.array([np.flipud(np.transpose(np.vstack([left_fitx+margin, ploty])))])
		left_line_pts = np.hstack((left_line_window1, left_line_window2))
		right_line_window1 = np.array([np.transpose(np.vstack([right_fitx-margin, ploty]))])
		right_line_window2 = np.array([np.flipud(np.transpose(np.vstack([right_fitx+margin, ploty])))])
		right_line_pts = np.hstack((right_line_window1, right_line_window2))

		# Draw the lane onto the warped blank image
		cv2.fillPoly(window_img, np.int_([left_line_pts]), (0,255, 0))
		cv2.fillPoly(window_img, np.int_([right_line_pts]), (0,255, 0))

		result = cv2.addWeighted(out_img, 1, window_img, 0.3, 0)

		f, ax = plt.subplots()
		plt.imshow(result)
		plt.plot(left_fitx, ploty, color='yellow')
		plt.plot(right_fitx, ploty, color='yellow')
		plt.xlim(0, 1280)
		plt.ylim(720, 0)
		f.savefig('./output_images/vis-' + type(self).__name__ + '-F' + str(self.data['frame_num']) +'-fitline.png')

		return result

	def process(self, data):
		self.debug = data['debug']
		self.data = data

		if ('line' in data.keys()):
			#print('targetted line search')
			lines = data['lines']
			self.binary_warped = data['image']
			newlines = self.search(self.binary_warped, data)
			data['lines'] = newlines
			self.image = data['image']


			#print('new Lines:', newlines)

			lineobj = data['line']

			#print('lastN L:', lineobj.lastn_left)
			#print('lastN R:', lineobj.lastn_right)

			x0 = np.concatenate((lineobj.lastn_left, [newlines[0]]), axis=0)
			x1 = np.concatenate((lineobj.lastn_right, [newlines[1]]), axis=0)

			lineobj.lastn_left = x0
			lineobj.lastn_right = x1


			if (len(lineobj.lastn_left) > lineobj.N):
				new_l = lineobj.lastn_left[-lineobj.N:]
				new_r = lineobj.lastn_right[-lineobj.N:]
				lineobj.lastn_left = new_l
				lineobj.lastn_right = new_r


				lineobj.best_fit_left = np.mean(lineobj.lastn_left, axis=0)
				lineobj.best_fit_right = np.mean(lineobj.lastn_right, axis=0)

			
			else:
				lineobj.best_fit_left = newlines[0]
				lineobj.best_fit_right = newlines[1]


			if (self.debug):
				self.image = self.vis(newlines)

			data['lines'] = (lineobj.best_fit_left, lineobj.best_fit_right)

		print()
		print(type(self).__name__ + ' - ' + str(data['frame_num']), ' lines: ', data['lines'])
		
		return data