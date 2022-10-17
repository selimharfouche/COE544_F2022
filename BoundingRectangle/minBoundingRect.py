import cv2
import numpy as np
import os
import glob


# read the input images
for image in glob.glob("Images/img045-043.png"):
   if (image is not None):
    img0 = cv2.imread(image)

    # convert the image to grayscale
    img1 = cv2.cvtColor(img0, cv2.COLOR_BGR2GRAY)

    ret, thresh = cv2.threshold(img1, 0, 255, cv2.THRESH_BINARY_INV)
    contours,_ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # get countour points
    cnt = contours[0]

    # compute rotated rectangle (minimum area)
    rect = cv2.minAreaRect(cnt)

    [X, Y, W, H] = cv2.boundingRect(cnt)
    cropped_image = img1[Y:Y + H, X:X + W]

    #substring for image name
    newImage = image[7:]

    cv2.imwrite(os.path.join("CroppedImages" , newImage), cropped_image)

    resizedImg = cv2.resize(cropped_image, (668, 649), interpolation = cv2.INTER_AREA)

    cv2.imwrite(os.path.join("test2", newImage), resizedImg)

    cv2.waitKey(0)

