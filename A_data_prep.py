######################################################################
# imports
import random
import cv2
import numpy as np
import os
import glob
import pickle
from sklearn.model_selection import *
from A_helper_data_prep import *
import matplotlib.pyplot as plt
import pandas as pd

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



    ###### Skew correction

    # from A_skew_correction import correct_skew
    # img0 = correct_skew(img0)




    ####### Plotting debug
    # fig = plt.figure()
    # ax1 = fig.add_subplot(3,3,1)
    # ax1.set_title("original image")
    # ax1.imshow(img0)
    
   
    # convert the image to BW
    thresh = convert_BW(img0)



    ####### Plotting debug
    # ax2 = fig.add_subplot(3,3,2)
    # ax2.set_title("BW")
    # ax2.imshow(thresh)
    
   
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




    ####### Plotting debug
    # ax3 = fig.add_subplot(3,3,3)
    # ax3.set_title("Bounded image")
    # ax3.imshow(cropped_image)
    
    

    # resize image
    cropped_image = cv2.resize(cropped_image, (10,10), interpolation=cv2.INTER_AREA)




    ####### Plotting debug
    # ax4 = fig.add_subplot(3,3,4)
    # ax4.set_title("cropeed image")
    # ax4.imshow(cropped_image)

    ####### Plotting debug
    # ax5 = fig.add_subplot(3,3,5)
    # ax5.set_title("Skew correction")
    # ax5.imshow(cropped_image)
    # plt.show()
    #data.append([LocalBinaryPatterns(24,8,cropped_image), label])


    feature1=pixel_intensity(cropped_image).flatten()



    # feature2=histogram(cropped_image).flatten()
    # features_appended = np.append(feature1,feature2)



   # df = pd.DataFrame(np.vstack([feature1, feature2]).T, columns=['feature1', 'feature2'])
    
    data.append([feature1,label])
    counter = counter + 1
   


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

for features1, label in data:
    features.append(features1)
    #features.append(feature2)
    labels.append(label)
    


# Separate the data into training and test data sets

X_train, X_test, Y_train, Y_test = train_test_split(features, labels, test_size=0.8)