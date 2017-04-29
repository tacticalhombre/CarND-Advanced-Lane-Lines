
---

**Advanced Lane Finding Project**

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # (Image References)

[image1]: ./examples/undistort_output.png "Undistorted"
[image2]: ./examples/vis-Calibrate-undistort.png "Road Transformed"
[image3]: ./examples/vis-Filter.png "Binary Example"
[image4]: ./examples/vis-PerspectiveTransform-src.png "Warp Example"
[image5]: ./examples/vis-SlidingWindow-histogram.png "Histogram peaks"
[image7]: ./examples/vis-SlidingWindow-fitline.png "Fitline"
[image8]: ./examples/vis-TargettedLineSearch-fitline.png "Search area"
[image6]: ./examples/vis-Pipeline.png "Output"
[video1]: ./project_video.mp4 "Video" 

## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points
###Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---
###Writeup / README

####1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Advanced-Lane-Lines/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.  

You're reading it!
###Camera Calibration

####1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

The code for this step is contained in the file Calibration.py.

I start by preparing "object points", which will be the (x, y, z) coordinates of the chessboard corners in the world. Here I am assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image.  Thus, `objp` is just a replicated array of coordinates, and `objpoints` will be appended with a copy of it every time I successfully detect all chessboard corners in a test image.  `imgpoints` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection.  

Since the chessboards corners are not all 9x6, I tried with other dimensions (9x5, 8x6, 7x6) as well to get as much datapoints as possible (main method lines 103-107). I then saved the list of objpoints and imgpoints into a pickled file calibrate.p. The save() method is at line 70.

I then used the `objpoints` and `imgpoints` to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function (line 88).    Later in the pipeline, I applied this distortion correction to the test image using the `cv2.undistort()` (line 88) function and obtained this result: 

![alt text][image1]

### Code structure ###

I created an abstract class ProcessStep (in process.py) that will be used as a superclass for classes that will encapsulate each of the steps below:

Camera calibration (Calibration.py)
Distortion correction (Calibration.py)
Color/gradient threshold (Filter.py - Filter class; Thresholds.py - Thresholds class)
Perspective transform (PerspectiveTransform.py - PerspectiveTransform class)
Detect lane lines and compute curvature (Lane.py - Lane class)

It has a process() method that subclasses implement to contain the meat of the logic for each of the steps in processing an image.  It takes a single argument 'data' that is a dictionary that will contain objects that are collected/computed as we progress thru the pipeline.   

In the main pipeline method (main.py, Pipeline.invoke_steps() ), we just iterate through the list of ProcessStep objects and invoke the process() method.

###Pipeline (single images)

####1. Provide an example of a distortion-corrected image.
To demonstrate this step, I will describe how I apply the distortion correction to one of the test images. Below is a sample of a distorted and undistorted image.
![alt text][image2]

The undistorted image is produced using cv2.undistort().  (Calibration.py. Calibration.cal_undistort() method line 88)

####2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.
At first, I used a combination of Sobel gradient with magnitude and direction thresholds combined with thresholding the S-channel for the image's HLS colorspace.  This did not turn out satisfactory results.   
As suggested by my project reviewer, I explored other colorspaces and rely less on Sobel for this particular application.  I ended up thresholding the L channel of CIELuv for isolating white lines and thresholding the b channel in CIELab to isolate yellow lines.  The methods for this are contained in Thresholds.py - 

methods cielab_select() and cieluv_select().

I still used Sobel gradients but instead of using a grayscale image as input I used the S channel from the image's HLS colorspace - Line 23-32 in method abs_sobel_thresh() in Thresholds.py.

Finally, I combined these to create a binary image in file Filter.py in method apply() - line 33-46.  The apply() method is used in process() (line 48) and eventually invoked in the pipeline.
Below are output when applied to the test images.

![alt text][image3]

####3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

The code for my perspective transform includes a function called `warp()` in class PerspectiveTransform, which appears in lines 37 through 46 in the file `PerspectiveTransform.py`.  The `warp()` function takes as inputs an image (`img`), as well as source (`src`) and destination (`dst`) points.  I chose the source and destination points in the following manner:

```
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


```


I verified that my perspective transform was working as expected by drawing the `src` and `dst` points onto a test image and its warped counterpart to verify that the lines appear parallel in the warped image.

![alt text][image4]

####4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?
After applying calibration, thresholding, and a perspective transform to a road image, I need to determine the lane lines from the binary image. In order to do this, we first need to take the histogram of pixel counts along the x-axis (SlidingWindow.find_lines() line 28).  It should give us a good indication of where the lanes lines are by looking at the peaks in the histogram.

![alt text][image5]

I used the sliding windows technique to determine the location of the lane lines. 
The two most prominent peaks in the histogram will be good indicators of the x-position of the base of the lane lines. Once we know this, we can use a sliding window, placed around the line centers, to find and follow the lines up to the top of the frame (SlidingWindow.find_lines() method, line 24-107).
![alt text][image7]

For processing video frames, we do not have to do this for every frame.  Once we determine the lane lines for one frame, we can search within the same vicinity in the next frame.  This procedure is done in TargettedLineSearch.py TargettedLineSearch.search() method, line 18. A sample image with the search area is shown below
![alt text][image8]

####5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

I did this in lines #21 through #62 in my code in `Lane.py`.  It is contained in the method get_curvature() and is called within the process() method.  The offset of the vehicle from center lane is computed in this method as well (line 53-58).

####6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

I implemented this step in lines #61 through #68 in my code in `main.py` in the function `invoke_steps()`.  This method is also responsible for invoking the process() method of each of the ProcessStep instances. Here is an example of my result on a test image:

![alt text][image6]

---

###Pipeline (video)

####1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's a [link to my video result](./my-project_video.mp4)

---

###Discussion

####1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

The one thing I would improve in this solution is finding a more robust combination of the thresholding mechanism.  The solution I used worked on the project video but did not perform well in the challenge videos.  I would explore more of the colorspace thresholding to better determine the lane lines in different conditions (fake lines due to shadows / road imperfections, glare, or reduced image clarity due to weather).  Another thing to improve is the smoothing of lane lines from one frame to the next.  I used an averaging approach but did not have time to explore alternatives like exponential smoothing.

