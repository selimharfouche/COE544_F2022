# Helpers
# Helper #1: takes as input two bounding rectangles and returns them merged
import cv2
import numpy as np
def union(a, b):
 x = min(a[0], b[0])
 y = min(a[1], b[1])
 w = max(a[0] + a[2], b[0] + b[2]) - x
 h = max(a[1] + a[3], b[1] + b[3]) - y
 return (x, y, w, h)

# Helper #2: converts image to black and white
def convert_BW(image):
    img1 = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    ret, thresh = cv2.threshold(img1, 0, 255, cv2.THRESH_BINARY_INV)
    return thresh
###########################################################################
###########################################################################
###########################################################################


###########################################################################
###########################################################################
###########################################################################
# Features
# Feature #1: Aspect Ratio
# Reminders:
# images are stored in 2D arrays
# image.shape is (n,m). So image.shape[0] is n.

###########################################################################
# Feature #1.a : aspect ratio in decimal format
def aspect_ratio(cropped_image):
    # number of columns / number of rows
    ar = ((float)(cropped_image.shape[1] / cropped_image.shape[0]))
    return ar
###########################################################################

###########################################################################
# Feature #1.b : aspect ratio in decimal format
def calculate_aspect(width: int, height: int) -> str:
    def gcd(a, b):
        return a if b == 0 else gcd(b, a % b)


    r = gcd(width, height)
    x = int(width / r)
    y = int(height / r)

    print(f"{x}:{y}")
    print()
    return
###########################################################################

###########################################################################
###########################################################################

###########################################################################
###########################################################################
# Feature 2: Pixel Count, Pixel Percent 
# 4 variables
# takes as input the cropped image
# returns the percentage of black pixels in it

###########################################################################
# Feature 2.a: Black pixel percent of top_half_img
def top_half_img(cropped_image):
    top_half = cropped_image[int(0):int(cropped_image.shape[0] / 2), int(0):int(cropped_image.shape[1])]

    # get all non black Pixels
    cntNotBlack = cv2.countNonZero(top_half)

    # get pixel count of image
    height, width = top_half.shape
    cntPixels = height * width

    # compute all black pixels
    cntBlack = cntPixels - cntNotBlack
    percent_black = int((cntBlack / cntPixels) * 100)

    return percent_black
###########################################################################

###########################################################################
# Feature 2.b: Black pixel percent of lower_half_img
def lower_half_img(cropped_image):
    lower_half = cropped_image[int(cropped_image.shape[0] / 2):int(cropped_image.shape[0]), int(0):int(cropped_image.shape[1])]

    # get all non black Pixels
    cntNotBlack = cv2.countNonZero(lower_half)

    # get pixel count of image
    height, width = lower_half.shape
    cntPixels = height * width

    # compute all black pixels
    cntBlack = cntPixels - cntNotBlack
    percent_black = (cntBlack / cntPixels) * 100

    return percent_black

###########################################################################
# Feature 2.c: Black pixel percent of right_half_img
def right_half_img(cropped_image):
    right_half = cropped_image[int(0):int(cropped_image.shape[0]), int(cropped_image.shape[1] / 2):int(cropped_image.shape[1])]

    # get all non black Pixels
    cntNotBlack = cv2.countNonZero(right_half)

    # get pixel count of image
    height, width = right_half.shape
    cntPixels = height * width

    # compute all black pixels
    cntBlack = cntPixels - cntNotBlack
    percent_black = (cntBlack / cntPixels) * 100

    return percent_black

###########################################################################
# Feature 2.d: Black pixel percent of left_half_img
def left_half_img(cropped_image):
    left_half = cropped_image[int(0):int(cropped_image.shape[0]), int(0):int(cropped_image.shape[1] / 2)]

    # get all non black Pixels
    cntNotBlack = cv2.countNonZero(left_half)

    # get pixel count of image
    height, width = left_half.shape
    cntPixels = height * width

    # compute all black pixels
    cntBlack = cntPixels - cntNotBlack
    percent_black = (cntBlack / cntPixels) * 100

    return percent_black
###########################################################################
###########################################################################
# Feature 3: Projection histogram

def histogram(cropped_image):

    cropped_image[cropped_image == 0] = 1
    cropped_image[cropped_image == 255] = 0

    # Calculate horizontal projection
    hor_proj = np.sum(cropped_image, axis=1)

    height, width = cropped_image.shape

    blankImage = np.zeros((height, width), np.uint8)

    # Draw a line for each row
    for idx, value in enumerate(hor_proj):
        cv2.line(blankImage, (0, idx), (width-int(value), idx), (255, 255, 255), 1)

    # Save result
    blankImage = cv2.resize(blankImage, (8, 8), interpolation=cv2.INTER_AREA)

    return blankImage
###########################################################################
###########################################################################

###########################################################################
###########################################################################
# Feature 4: Pixel intensity
def pixel_intensity(cropped_image):
    n_samples = len(cropped_image)
    cropped_image_reshaped = cropped_image.reshape((n_samples, -1))
    return cropped_image_reshaped
###########################################################################
###########################################################################

###########################################################################
###########################################################################
# Feature 5: Sobel edge

def sobel_edge(cropped_image):
    # Sobel Edge Detection
    # sobelx = cv2.Sobel(src=cropped_image, ddepth=cv2.CV_64F, dx=1, dy=0, ksize=5) # Sobel Edge Detection on the X axis
    # sobely = cv2.Sobel(src=cropped_image, ddepth=cv2.CV_64F, dx=0, dy=1, ksize=5) # Sobel Edge Detection on the Y axis
    sobelxy = cv2.Sobel(src=cropped_image, ddepth=cv2.CV_64F, dx=1, dy=1, ksize=5) # Combined X and Y Sobel Edge Detection
    return sobelxy
###########################################################################
###########################################################################

###########################################################################
###########################################################################
# Feature 6: Canny Edge
def canny_edge(cropped_image):
    edges = cv2.Canny(image=cropped_image, threshold1=100, threshold2=200) # Canny Edge Detection
    return edges

from skimage import feature
from skimage.feature import hog
#
# ###########################################################################
# ###########################################################################
#
# ###########################################################################
# ###########################################################################
#Feature 7: Local Binary Patterns
def LocalBinaryPatterns(numPoints, radius, image, eps=1e-7):

	lbp = feature.local_binary_pattern(image, numPoints, radius, method="uniform")
	(hist, _) = np.histogram(lbp.ravel(), bins=np.arange(0, numPoints + 3), range=(0, numPoints + 2))

	hist = hist.astype("float")
	hist /= (hist.sum() + eps)

	# return the histogram of Local Binary Patterns
	return hist
###########################################################################
###########################################################################


#Feature 7: HOG
def HOG(cropped_image):

    cv2.imwrite(("cropped_temp.png"), cropped_image)

    img1 = cv2.cvtColor(cv2.imread("cropped_temp.png"), cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(img1, 127, 255, cv2.THRESH_BINARY_INV)

    cv2.imwrite(("test.png"), thresh)

    # resizing image
    resized_img = resize(thresh, (8 * 4, 8 * 4))

    # creating hog features
    fd, hog_image = hog(cropped_image, orientations=9, pixels_per_cell=(8, 8),
                        cells_per_block=(2, 2), visualize=True, multichannel=False)

    hog_image = cv2.resize(cropped_image, (16, 16), interpolation=cv2.INTER_AREA)

    return hog_image

###########################################################################
###########################################################################
###########################################################################