######################################################################
######################## IMPORTS & Variables #########################
######################################################################


from pickle import dump

import cv2, glob, os, pickle, random

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.model_selection import *
from A_helper_data_prep import *



# Constants
DIRECTORY_TRAINING_DATA = 'datasets/B'

#When iterating through the training data looking for images
Windows_Iteration ='\\*.png'
Mac_Iteration='//*.png'

# Test size 
TEST_SIZE = 0.3


# Categories labeling
# Each directory name represents the label of the data we are working on 
categories = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
'a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z',
'a\'','b\'','c\'','d\'','e\'','f\'','g\'','h\'','i\'','j\'','k\'','l\'','m\'','n\'','o\'','p\'','q\'','r\'','s\'','t\'','u\'','v\'','w\'','x\'','y\'','z\'']
data = []
counter = 0
######################################################################


######################################################################
########################### IMAGE READING ############################
######################################################################


# Read the input images
for category in categories:
 path = os.path.join(DIRECTORY_TRAINING_DATA, category)
 label = categories.index(category)

 for img in glob.glob(path + Mac_Iteration):
  if (img is not None):
    img0 = cv2.imread(img) # Image read

######################################################################


######################################################################
########################## SKEW CORRECTION ###########################
######################################################################
    # # Skew correction

    # from A_skew_correction import correct_skew
    # img0 = correct_skew(img0)
######################################################################
   

######################################################################
############################ PLOTTING-1 ##############################
######################################################################
    ####### Plotting debug: Original Image
    # fig = plt.figure()
    # ax1 = fig.add_subplot(3,3,1)
    # ax1.set_title("original image")
    # ax1.imshow(img0)
######################################################################
    

   
######################################################################
############################### B&W ##################################
######################################################################
    # convert the image to BW
    thresh = convert_BW(img0)

######################################################################


######################################################################
############################ PLOTTING-2 ##############################
######################################################################
    ####### Plotting debug : Black and White Image
    # ax2 = fig.add_subplot(3,3,2)
    # ax2.set_title("BW")
    # ax2.imshow(thresh)
######################################################################

######################################################################
############### Minimum Bounding rectangle extraction ################
######################################################################   
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

######################################################################

######################################################################
############################ PLOTTING-3 ##############################
######################################################################
    ####### Plotting debug : Bounded Image
    # ax3 = fig.add_subplot(3,3,3)
    # ax3.set_title("Bounded image")
######################################################################
    
######################################################################
############################# RENDERING ##############################
######################################################################
    # resize image
    #cropped_image = cv2.resize(cropped_image, (10,10), interpolation=cv2.INTER_AREA)



######################################################################

######################################################################
########################### PLOTTING-4&5 #############################
######################################################################
    ####### Plotting debug
    # ax4 = fig.add_subplot(3,3,4)
    # ax4.set_title("cropeed image")
    # ax4.imshow(cropped_image)
    # ax5 = fig.add_subplot(3,3,5)
    # ax5.set_title("Skew correction")
    # ax5.imshow(cropped_image)
    # plt.show()
######################################################################



######################################################################
######################## FEATURES EXTRACTION #########################
######################################################################
    
    # Extracting first feature 
    feature1=pixel_intensity(cropped_image).flatten()

    # # Combining multiple features
    # feature2=histogram(cropped_image).flatten()
    # features_appended = np.append(feature1,feature2)
    
    # feature3 LocalBinaryPatterns(24,8,cropped_image)
    # To BE Completed

    
    data.append([feature1,label])

    # # OR if multiples features
    #  data.append([features_appended,label])

    # update the counter
    counter = counter + 1
   
######################################################################

######################################################################
######################## WHYYY?????????????? #########################
######################################################################
#write data into pickle file
pick_in = open('data.pickle','wb')
pickle.dump(data, pick_in)
pick_in.close()

pick_in = open('data.pickle', 'rb')
data = pickle.load(pick_in)
pick_in.close()
######################################################################

# DATA SHUFFLING
random.shuffle(data)
features = []
labels = []


    

######################################################################
########################## DATA SEPARATION ###########################
######################################################################

# for features1, label in data:
#     features.append(features1)
#     labels.append(label)

# Separate the data into training and test data sets

X_train, X_test, Y_train, Y_test = train_test_split(features, labels, test_size=TEST_SIZE)

# save the model
dump(X_train, open('data/X_train.pkl', 'wb'))
dump(X_test, open('data/X_test.pkl', 'wb'))
dump(Y_test, open('data/Y_test.pkl', 'wb'))
dump(Y_train, open('data/Y_train.pkl', 'wb'))






######################################################################