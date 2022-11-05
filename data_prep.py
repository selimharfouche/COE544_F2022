


from pickle import dump

import cv2, glob, os, pickle, random

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.model_selection import *
from A_helper_data_prep import *


class data_prep():
    print()
    DIRECTORY_TRAINING_DATA = 'datasets/A'
    Windows_Iteration ='\\*.png'
    Mac_Iteration='//*.png'
    TEST_SIZE = 0.3
    categories = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
    'a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z',
    'a\'','b\'','c\'','d\'','e\'','f\'','g\'','h\'','i\'','j\'','k\'','l\'','m\'','n\'','o\'','p\'','q\'','r\'','s\'','t\'','u\'','v\'','w\'','x\'','y\'','z\'']
    data = []
    counter = 0
    for category in categories:
        path = os.path.join(DIRECTORY_TRAINING_DATA, category)
        label = categories.index(category)
        for img in glob.glob(path + Mac_Iteration):
            if (img is not None):
                img0 = cv2.imread(img) # Image read
                thresh = convert_BW(img0)
                contours,_ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                if(len(contours)>1):
                    x0, y0, w0, h0 = cv2.boundingRect(contours[0])
                    rect0 = x0, y0, w0, h0
                    x1, y1, w1, h1 = cv2.boundingRect(contours[1])
                    rect1 = x1, y1, w1, h1
                    # merge 2 rectanlges
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
                    cnt0 = contours[0]
                    cnt1 = 0
                    # compute rectangle (minimum area)
                    X, Y, W, H = cv2.boundingRect(cnt0)
                # crop image following the rectangle
                cropped_image = thresh[int(Y):int(Y + H), int(X):int(X + W)]
                cropped_image = cv2.resize(cropped_image, (10,10), interpolation=cv2.INTER_AREA)
                feature1=pixel_intensity(cropped_image).flatten()
                data.append([feature1,label])
                counter = counter + 1

    print("AAAAAAA")
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
        labels.append(label)
    X_train, X_test, Y_train, Y_test = train_test_split(features, labels, test_size=TEST_SIZE)

    dump(X_train, open('data/X_train.pkl', 'wb'))
    dump(X_test, open('data/X_test.pkl', 'wb'))
    dump(Y_test, open('data/Y_test.pkl', 'wb'))
    dump(Y_train, open('data/Y_train.pkl', 'wb'))

