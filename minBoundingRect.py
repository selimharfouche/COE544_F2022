import random
import cv2
import numpy as np
import os
import glob
import math
from PIL import Image
import matplotlib.pyplot as plt
import pickle
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

# function that merges 2 rectangles
def union(a, b):
 x = min(a[0], b[0])
 y = min(a[1], b[1])
 w = max(a[0] + a[2], b[0] + b[2]) - x
 h = max(a[1] + a[3], b[1] + b[3]) - y
 return (x, y, w, h)


# aspect ratio (in decimal format)
def aspect_ratio(cropped_image):
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


def top_half_img(cropped_image):
    top_half = cropped_image[int(0):int(H / 2), int(0):int(W)]

    # get all non black Pixels
    cntNotBlack = cv2.countNonZero(top_half)

    # get pixel count of image
    height, width = top_half.shape
    cntPixels = height * width

    # compute all black pixels
    cntBlack = cntPixels - cntNotBlack
    percent_black = (cntBlack / cntPixels) * 100

    cv2.imwrite(os.path.join("test1", newImage), top_half)

    return percent_black


def lower_half_img(cropped_image):
    lower_half = cropped_image[int(H / 2):int(H), int(0):int(W)]

    # get all non black Pixels
    cntNotBlack = cv2.countNonZero(lower_half)

    # get pixel count of image
    height, width = lower_half.shape
    cntPixels = height * width

    # compute all black pixels
    cntBlack = cntPixels - cntNotBlack
    percent_black = (cntBlack / cntPixels) * 100

    cv2.imwrite(os.path.join("test2", newImage), lower_half)

    return percent_black


def right_half_img(cropped_image):
    right_half = cropped_image[int(0):int(H), int(W / 2):int(W)]

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
    left_half = cropped_image[int(0):int(H), int(0):int(W / 2)]

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


def histogram(cropped_image):
    cropped_image[cropped_image == 0] = 1
    cropped_image[cropped_image == 255] = 0

    # Calculate horizontal projection
    hor_proj = np.sum(cropped_image, axis=1)
    print(hor_proj)

    height, width = cropped_image.shape

    blankImage = np.zeros((height, width, 3), np.uint8)

    # Draw a line for each row
    for row in range(height):
        cv2.line(blankImage, (0, row), (int(hor_proj[row] * width / height), row), (255, 255, 255), 1)

    # Save result
    cv2.imwrite('test5/result.png', blankImage)

def pixel_intensity(cropped_image):
    n_samples = len(cropped_image)
    data = cropped_image.reshape((n_samples, -1))
    return data

categories = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
data = []

# read the input images
for category in categories:
 path = os.path.join("Images", category)

 label = categories.index(category)

 for img in glob.glob(path + "\\*.png"):
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
    cropped_image = cv2.resize(cropped_image, (8, 8), interpolation=cv2.INTER_AREA)
    # substring for image name
    newImage = img[9:]
    #cv2.imwrite(os.path.join(("newImages//"+str(category)), newImage), cropped_image)

    data.append([pixel_intensity(cropped_image), label])
print(len(data))

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
    labels.append(label)

# Separate the data into training and test data sets
X_train, X_test, Y_train, Y_test = train_test_split(features, labels, test_size=0.30)

# from sklearn.preprocessing import StandardScaler
# scaler = StandardScaler()
# X_train_scaled = scaler.fit_transform(X_train)
# # we must apply the scaling to the test set that we computed for the training set
# X_test_scaled = scaler.transform(X_test)
X_train_scaled = X_train
X_test_scaled = X_test

# Import Matplotlib for visualization
import matplotlib.pyplot as plt

acc = []
# Will take some time
from sklearn import metrics

# for i in range(1, 20):
#     neigh = KNeighborsClassifier(n_neighbors=i).fit(X_train_scaled, Y_train)
#     yhat = neigh.predict(X_test_scaled)
#     acc.append(metrics.accuracy_score(Y_test, yhat))
#
# plt.figure(figsize=(10, 6))
# plt.plot(range(1, 20), acc, color='blue', linestyle='dashed',
#          marker='o', markerfacecolor='red', markersize=10)
# plt.title('accuracy vs. K Value')
# plt.xlabel('K')
# plt.ylabel('Accuracy')
# print("Maximum accuracy:-", max(acc), "at K =", acc.index(max(acc)))
# knn = KNeighborsClassifier(n_neighbors = acc.index(max(acc)))
# knn.fit(X_train_scaled, Y_train)

#Checking performance on the training set

#Grid search best k-NN parameters
parameters = {'n_neighbors':list(range(1, 20)) ,'weights': ['uniform', 'distance']}
from sklearn.model_selection import GridSearchCV
# create an instance of the knn classifier
knn_grid_tuned = KNeighborsClassifier()
# create an instance of grid search with the above parameters
grid_search = GridSearchCV(knn_grid_tuned, parameters, cv=2, scoring='accuracy', return_train_score=True, verbose=10)
# fit the grid search with training set
grid_search.fit(X_train_scaled, Y_train)
# retrieve the best estimator
knn_tuned = grid_search.best_estimator_
print(knn_tuned)

from sklearn.metrics import accuracy_score
print(accuracy_score(Y_test, knn_tuned.predict(X_test_scaled)))