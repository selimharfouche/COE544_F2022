######################################################################
# imports
import random
import cv2
import numpy as np
import os
import glob
import pickle
from sklearn.model_selection import *
from helper_data_prep import *

######################################################################
# Constants
DIRECTORY_TRAINING_DATA = 'Images/'

#When iterating through the training data looking for images
Windows_Iteration ='\\*.png'
Mac_Iteration='//*.png'

######################################################################

# Categories labeling
# Each directory name represents the label of the data we are working on 
categories = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
data = []
counter = 0

######################################################################
#Append images to data
# read the input images
for category in categories:
 path = os.path.join(DIRECTORY_TRAINING_DATA, category)
 label = categories.index(category)

 for img in glob.glob(path + Mac_Iteration):
  if (img is not None):
    img0 = cv2.imread(img)
    # convert the image to BW
    thresh = convert_BW(img0)

    # Finding contours
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
    cropped_image = thresh[int(Y):int(Y + H), int(X):int(X + W)]

    # resize image
    cropped_image = cv2.resize(cropped_image, (8,8), interpolation=cv2.INTER_AREA)
    data.append([LocalBinaryPatterns(24,8,cropped_image), label])
    counter = counter + 1
   


#write data into pickle file
pick_in = open('data.pickle','wb')
pickle.dump(data, pick_in)
pick_in.close()

pick_in = open('data.pickle', 'rb')
data = pickle.load(pick_in)
pick_in.close()

#random.shuffle(data)
features = []
labels = []

for feature1, label in data:
    features.append(feature1.flatten())
    #features.append(feature2)
    labels.append(label)

# Separate the data into training and test data sets

X_train, X_test, Y_train, Y_test = train_test_split(features, labels, test_size=0.40)