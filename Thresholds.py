import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from skimage import io, color


class Thresholds():

    def __init__(self):
        print('Thresholds object created ...')

    def abs_sobel_thresh(self, image, orient='x', sobel_kernel=3, thresh=(0, 255)):

        thresh_min = thresh[0]
        thresh_max = thresh[1]

        # Convert to grayscale
        #gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        #bt = self.hls_select(image, (90, 255))
        #gray = cv2.bitwise_and(image, image, mask = bt).astype(np.uint8)    
        hls = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
        s_channel = hls[:,:,2]
        gray = s_channel

        # Apply x or y gradient with the OpenCV Sobel() function
        # and take the absolute value
        if orient == 'x':
            abs_sobel = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 1, 0))
        if orient == 'y':
            abs_sobel = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 0, 1))
        
        # Rescale back to 8 bit integer
        scaled_sobel = np.uint8(255*abs_sobel/np.max(abs_sobel))
        
        # Create a copy and apply the threshold
        binary_output = np.zeros_like(scaled_sobel)
        
        # Here I'm using inclusive (>=, <=) thresholds, but exclusive is ok too
        binary_output[(scaled_sobel >= thresh_min) & (scaled_sobel <= thresh_max)] = 1

        # Return the result
        return binary_output

    def mag_thresh(self, image, sobel_kernel=3, mag_thresh=(0, 255)):    # Convert to grayscale
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        # Take both Sobel x and y gradients
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
        
        # Calculate the gradient magnitude
        gradmag = np.sqrt(sobelx**2 + sobely**2)
        
        # Rescale to 8 bit
        scale_factor = np.max(gradmag)/255 
        gradmag = (gradmag/scale_factor).astype(np.uint8) 
        
        # Create a binary image of ones where threshold is met, zeros otherwise
        binary_output = np.zeros_like(gradmag)
        binary_output[(gradmag >= mag_thresh[0]) & (gradmag <= mag_thresh[1])] = 1

        # Return the binary image
        return binary_output

    def dir_threshold(self, image, sobel_kernel=3, thresh=(0, np.pi/2)):
        # Grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        # Calculate the x and y gradients
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
        
        # Take the absolute value of the gradient direction, 
        # apply a threshold, and create a binary image result
        absgraddir = np.arctan2(np.absolute(sobely), np.absolute(sobelx))
        binary_output =  np.zeros_like(absgraddir)
        binary_output[(absgraddir >= thresh[0]) & (absgraddir <= thresh[1])] = 1

        # Return the binary image
        return binary_output

    # Define a function that thresholds the S-channel of HLS
    def hls_select(self, image, thresh=(0, 255)):
        hls = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
        s_channel = hls[:,:,2]
        binary_output = np.zeros_like(s_channel)
        binary_output[(s_channel > thresh[0]) & (s_channel <= thresh[1])] = 1
        return binary_output

    def cielab_select(self, image, thresh=(0, 255)):
        lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
        b = lab[:,:,2]
        binary_output = np.zeros_like(b)
        binary_output[(b > thresh[0]) & (b <= thresh[1])] = 1

        return binary_output

    def cieluv_select(self, image, thresh=(0, 255)):
        luv = cv2.cvtColor(image, cv2.COLOR_RGB2LUV)
        l = luv[:,:,0]
        binary_output = np.zeros_like(l)
        binary_output[(l > thresh[0]) & (l <= thresh[1])] = 1

        return binary_output

    def h_select(self, image):

        hls = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
        
        # create NumPy arrays from the boundaries
        lower = np.array([ 20, 120, 80], dtype = "uint8")
        upper = np.array([ 45, 200, 255], dtype = "uint8")

        # find the colors within the specified boundaries and apply
        # the mask
        mask = cv2.inRange(hls, lower, upper)
        
        return mask

    def yellow_select(self, image, thresh=(0, 255)):
        hls = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
        h_channel = hls[:,:,0]
        binary_output = np.zeros_like(h_channel)
        binary_output[(h_channel > thresh[0]) & (h_channel <= thresh[1])] = 1
        return binary_output

def main():
    t = Thresholds()
    
    # Choose a Sobel kernel size
    ksize = 3 # Choose a larger odd number to smooth gradient measurements

    image = cv2.imread('./straight_lines1.jpg')
    # Apply each of the thresholding functions
    gradx = t.abs_sobel_thresh(image, orient='x', sobel_kernel=ksize, thresh=(0, 255))


if __name__ == "__main__":
    main()

"""
hls_binary = hls_select(image, thresh=(90, 255))


# Choose a Sobel kernel size
ksize = 3 # Choose a larger odd number to smooth gradient measurements

# Apply each of the thresholding functions
gradx = abs_sobel_thresh(image, orient='x', sobel_kernel=ksize, thresh=(0, 255))
grady = abs_sobel_thresh(image, orient='y', sobel_kernel=ksize, thresh=(0, 255))
mag_binary = mag_thresh(image, sobel_kernel=ksize, mag_thresh=(0, 255))
dir_binary = dir_threshold(image, sobel_kernel=ksize, thresh=(0, np.pi/2))


combined = np.zeros_like(dir_binary)
combined[((gradx == 1) & (grady == 1)) | ((mag_binary == 1) & (dir_binary == 1))] = 1
"""

# Plot the result
"""
f, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 9))
f.tight_layout()
ax1.imshow(image)
ax1.set_title('Original Image', fontsize=50)
ax2.imshow(hls_binary, cmap='gray')
ax2.set_title('Thresholded S', fontsize=50)
plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
"""
