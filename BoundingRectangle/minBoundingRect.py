import cv2
import numpy as np
import os
import glob
import math

# function that merges 2 rectangles
def union(a, b):
 x = min(a[0], b[0])
 y = min(a[1], b[1])
 w = max(a[0] + a[2], b[0] + b[2]) - x
 h = max(a[1] + a[3], b[1] + b[3]) - y
 return (x, y, w, h)


# read the input images
for image in glob.glob("Images/*.png"):
   if (image is not None):
    img0 = cv2.imread(image)

    # convert the image to grayscale
    img1 = cv2.cvtColor(img0, cv2.COLOR_BGR2GRAY)

    ret, thresh = cv2.threshold(img1, 0, 255, cv2.THRESH_BINARY_INV)
    contours,_ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)


    if(len(contours)>1):
     # get first 2 contour points (base cases)

     # compute rectangle (minimum area)
     x0, y0, w0, h0 = cv2.boundingRect(contours[0])
     rect0 = x0, y0, w0, h0

     x1, y1, w1, h1 = cv2.boundingRect(contours[1])
     rect1 = x1, y1, w1, h1


     # merge rectangles
     X, Y, W, H = union(rect0, rect1)
     rect_temp = X, Y, W, H

     for cnt0 in contours[2::1]:

         # get the rest of the contour points
         x0, y0, w0, h0 = cv2.boundingRect(cnt0)
         rect2 = x0, y0, w0, h0

         # merge rectangles
         X, Y, W, H = union(rect_temp, rect2)
         rect_temp = X, Y, W, H


    else:
     # get cooint (in case of 1 rectangle)
     cnt0 = contours[0]
     cnt1 = 0

     # compute rectangle (minimum area)
     X, Y, W, H = cv2.boundingRect(cnt0)

    # crop image following the rectangle
    cropped_image = img1[int(Y):int(Y + H), int(X):int(X + W)]

    # substring for image name
    newImage = image[7:]
    cv2.imwrite(os.path.join("CroppedImages" , newImage), cropped_image)

    # resizing images
    #resizedImg = cv2.resize(cropped_image, (668, 649), interpolation = cv2.INTER_AREA)
    #cv2.imwrite(os.path.join("test1", newImage), resizedImg)



    #cv2.waitKey(0)

