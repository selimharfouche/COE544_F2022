import random
import cv2
import numpy as np
import os
import glob


import pickle
from sklearn.model_selection import train_test_split



DIRECTORY_TRAINING_DATA = '/Users/a12/Downloads/archive(1)/IMAGEs'

#When iterating through the training data looking for images
Windows_Iteration ='\\*.png'
Mac_Iteration='//*.png'

# function that merges 2 rectangles
def union(a, b):
 x = min(a[0], b[0])
 y = min(a[1], b[1])
 w = max(a[0] + a[2], b[0] + b[2]) - x
 h = max(a[1] + a[3], b[1] + b[3]) - y
 return (x, y, w, h)

######################################################################
# Feature #1: Aspect Ratio
# aspect ratio (in decimal format)
def aspect_ratio(cropped_image):
    # number of columns / number of rows
    ar = ((float)(cropped_image.shape[1] / cropped_image.shape[0]))
    return ar

# aspect ration (in fraction format)
def calculate_aspect(width: int, height: int) -> str:
    def gcd(a, b):
        return a if b == 0 else gcd(b, a % b)

    r = gcd(width, height)
    x = int(width / r)
    y = int(height / r)

    print(f"{x}:{y}")
    print()
    return

######################################################################
# Feature 2: percent of pixels
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

    #print(cntBlack)
    #cv2.imwrite(os.path.join("test1", newImage), top_half)

    return percent_black


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

    #cv2.imwrite(os.path.join("test2", newImage), lower_half)

    return percent_black


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

    cv2.imwrite(os.path.join("test3", newImage), right_half)

    return percent_black


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

    cv2.imwrite(os.path.join("test4", newImage), left_half)

    return percent_black

######################################################################
# Feature 3: projection histogram
def histogram(cropped_image):
    cropped_image[cropped_image == 0] = 1
    cropped_image[cropped_image == 255] = 0

    # Calculate horizontal projection
    hor_proj = np.sum(cropped_image, axis=1)

    height, width = cropped_image.shape

    blankImage = np.zeros((height, width), np.uint8)

    # Draw a line for each row
    for row in range(height):
        cv2.line(blankImage, (0, row), (int(hor_proj[row] * width / height), row), (255, 255, 255), 1)

    # Save result
    blankImage = cv2.resize(blankImage, (128, 128), interpolation=cv2.INTER_AREA)
    imgName = "img" + str(counter) + ".png"
    cv2.imwrite(os.path.join("test5", imgName), blankImage)

    return blankImage

def pixel_intensity(cropped_image):
    n_samples = len(cropped_image)
    cropped_image_reshaped = cropped_image.reshape((n_samples, -1))
    return cropped_image_reshaped



# Categories labeling
# Each directory name represents the label of the data we are working on 
categories = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y','z']
data = []
counter = 0

# read the input images
for category in categories:
 path = os.path.join(DIRECTORY_TRAINING_DATA, category)
 label = categories.index(category)

 for img in glob.glob(path + Mac_Iteration):
  if (img is not None):
    img0 = cv2.imread(img)



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
     # get contour (in case of 1 rectangle)
     cnt0 = contours[0]
     cnt1 = 0

     # compute rectangle (minimum area)
     X, Y, W, H = cv2.boundingRect(cnt0)

    # crop image following the rectangle
    cropped_image = img1[int(Y):int(Y + H), int(X):int(X + W)]

    # resize image
    cropped_image = cv2.resize(cropped_image, (8,8), interpolation=cv2.INTER_AREA)

    # substring for image name
    newImage = img[9:]
    #cv2.imwrite(os.path.join(("newImages//"+str(category)), newImage), cropped_image)
    cv2.imwrite(os.path.join(("newImages//"), newImage), cropped_image)

    #data.append(['{0:.3g}'.format(top_half_img(cropped_image)/lower_half_img(cropped_image)), '{0:.3g}'.format(right_half_img(cropped_image)/left_half_img(cropped_image)), label])

    #hor_proj = histogram(cropped_image)
    #hor_proj_reshaped = hor_proj.reshape((len(hor_proj),-1))
    #data.append([hor_proj_reshaped, label])
    #data.append([pixel_intensity(cropped_image), label])

    data.append([pixel_intensity(cropped_image), label])
    counter = counter + 1
    #print(len(data))

    # aspect_ratio()
    # calculate_aspect(W, H)
    #top_half_img(cropped_image)
    #lower_half_img(cropped_image)
    #right_half_img(cropped_image)
    #left_half_img(cropped_image)
    #histogram(cropped_image)
    #print()
    #cv2.waitKey(0)

#print(data)

#write data into pickle file
pick_in = open('data.pickle','wb')
pickle.dump(data, pick_in)
pick_in.close()

pick_in = open('data.pickle', 'rb')
data = pickle.load(pick_in)
pick_in.close()

random.shuffle(data)
features = []
labels = []

for feature1, label in data:
    features.append(feature1.flatten())
    #features.append(feature2)
    labels.append(label)

# Separate the data into training and test data sets

X_train, X_test, Y_train, Y_test = train_test_split(features, labels, test_size=0.30)
