import numpy as np
import cv2
import matplotlib.pyplot as plt
from process import ProcessStep
from Line import Line

class Lane(ProcessStep):
	
	#self.leftx = None
	#self.lefty = None
	#self.ploty = None
	
	def __init__(self):
		super().__init__()
		print('Lane object created ...')
		self.line = Line()
		self.binary_warped = None
		self.image = None
	
	def get_curvature(self, data):

		lineobj = data['line']

		left_fit = lineobj.best_fit_left
		right_fit = lineobj.best_fit_right

		# Generate x and y values for plotting
		ploty = np.linspace(0, self.binary_warped.shape[0]-1, self.binary_warped.shape[0] )

		y_eval = np.max(ploty)
		
		leftx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
		rightx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]

		#Define conversions in x and y from pixels space to meters
		ym_per_pix = 30/720 # meters per pixel in y dimension
		xm_per_pix = 3.7/700 # meters per pixel in x dimension

		ploty = np.linspace(0, self.binary_warped.shape[0]-1, self.binary_warped.shape[0] )

		# Fit new polynomials to x,y in world space
		left_fit_cr = np.polyfit(ploty*ym_per_pix, leftx*xm_per_pix, 2)
		right_fit_cr = np.polyfit(ploty*ym_per_pix, rightx*xm_per_pix, 2)

		# Calculate the new radii of curvature
		left_curverad = ((1 + (2*left_fit_cr[0]*y_eval*ym_per_pix + left_fit_cr[1])**2)**1.5) / np.absolute(2*left_fit_cr[0])
		right_curverad = ((1 + (2*right_fit_cr[0]*y_eval*ym_per_pix + right_fit_cr[1])**2)**1.5) / np.absolute(2*right_fit_cr[0])
		
		lane_center = (leftx[y_eval] + rightx[y_eval])/2

		image = data['image']
		camera_position = image.shape[0]/2

		center_offset_pixels = abs(camera_position - lane_center)
		#print('center_offset_pixels:', center_offset_pixels, ' in meters:', center_offset_pixels*xm_per_pix)
		lineobj.center_offset = center_offset_pixels*xm_per_pix

		# Now our radius of curvature is in meters
		#print(left_curverad, 'm', right_curverad, 'm')

		return left_curverad, right_curverad


	def draw_lines(self, lines):

		left_fit = lines[0]
		right_fit = lines[1]

		#print('LEFT:', left_fit)
		# Generate x and y values for plotting
		ploty = np.linspace(0, self.binary_warped.shape[0]-1, self.binary_warped.shape[0] )
		left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
		right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]

		#out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
		#out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]

		# Create an image to draw the lines on
		warp_zero = np.zeros_like(self.binary_warped).astype(np.uint8)
		color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

		# Recast the x and y points into usable format for cv2.fillPoly()
		pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
		pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
		pts = np.hstack((pts_left, pts_right))

		# Draw the lane onto the warped blank image
		result = cv2.fillPoly(color_warp, np.int_([pts]), (0,255, 0))

		return result

	def process(self, data):
		lines = data['lines']
		self.binary_warped = data['image']

		lineobj = data['line']
		lines = (lineobj.best_fit_left, lineobj.best_fit_right)
		self.image = self.draw_lines(lines)

		data['image'] = self.image

		c = self.get_curvature(data)

		curvature = (c[0] + c[1]) / 2
		lineobj.radius_of_curvature = curvature
		#print('curvature:', curvature)
		#print('lineobj.best_fit_left')
		#print(lineobj.best_fit_left)

		return data

