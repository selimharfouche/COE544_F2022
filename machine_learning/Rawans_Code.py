import random
import cv2
import numpy as np
import pandas as pd
import os
import glob
import math
from PIL import Image
import matplotlib.pyplot as plt
import pickle
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay, plot_confusion_matrix
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn import metrics, preprocessing
from sklearn.preprocessing import StandardScaler
import joblib as jl
from skimage.feature import hog
from skimage.transform import resize

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

    #cv2.imwrite(os.path.join("test3", newImage), right_half)

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

    #cv2.imwrite(os.path.join("test4", newImage), left_half)

    return percent_black

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
    # imgName = "img" + str(counter) + ".png"
    # cv2.imwrite(os.path.join("test5", imgName), blankImage)

    return blankImage

def histogram2(cropped_image):
    cropped_image[cropped_image == 0] = 1
    cropped_image[cropped_image == 255] = 0

    # Calculate vertical projection
    hor_proj = np.sum(cropped_image, axis=0)

    height, width = cropped_image.shape

    blankImage = np.zeros((height, width), np.uint8)

    # Draw a line for each column
    for idx, value in enumerate(hor_proj):
        cv2.line(blankImage, (idx, 0), (idx, height-int(value)), (255, 255, 255), 1)

    # Save result
    blankImage = cv2.resize(blankImage, (128, 128), interpolation=cv2.INTER_AREA)
    imgName = "img" + str(counter) + ".png"
    cv2.imwrite(os.path.join("test6", imgName), blankImage)

    return blankImage

def pixel_intensity(cropped_image):
    n_samples = len(cropped_image)
    cropped_image_reshaped = cropped_image.reshape((n_samples, -1))
    return cropped_image_reshaped

def sobel_edge(cropped_image):
    # Sobel Edge Detection
    sobelx = cv2.Sobel(src=cropped_image, ddepth=cv2.CV_64F, dx=1, dy=0, ksize=5) # Sobel Edge Detection on the X axis
    sobely = cv2.Sobel(src=cropped_image, ddepth=cv2.CV_64F, dx=0, dy=1, ksize=5) # Sobel Edge Detection on the Y axis
    sobelxy = cv2.Sobel(src=cropped_image, ddepth=cv2.CV_64F, dx=1, dy=1, ksize=5) # Combined X and Y Sobel Edge Detection
    return sobelxy

def canny_edge(cropped_image):
    edges = 255-cv2.Canny(image=cropped_image, threshold1=100, threshold2=200) # Canny Edge Detection
    return edges

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


def get_features(dataset_path, temp_mapping_labels, temp_feature_list):
    data = []

    counter = 0

    # read the input images
    for category in temp_mapping_labels.keys():
         path = os.path.join(dataset_path + "\\"+category)
         label = temp_mapping_labels[category]

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
                    #cropped_image = cv2.resize(cropped_image, (16,16), interpolation=cv2.INTER_AREA)

                    # substring for image name
                    newImage = "img"+str(counter)+".png"
                    #cv2.imwrite(os.path.join(("newImages//"+str(category)), newImage), cropped_image)
                    cv2.imwrite(os.path.join(("newImages//"), newImage), cropped_image)

                    #data.append(['{0:.3g}'.format(right_half_img(cropped_image)), label])

                    #data.append([canny_edge(cropped_image), sobel_edge(cropped_image), label])
                    # features = {'Aspect Ratio':aspect_ratio(cropped_image), 'Top Half':top_half_img(cropped_image), 'Lower Half':lower_half_img(cropped_image),
                    #             'Right Half':right_half_img(cropped_image), 'Left Half':left_half_img(cropped_image), 'Pixel Intensity':pixel_intensity(cropped_image),
                    #             'Histogram': histogram(cropped_image), 'Canny Edge Detector':canny_edge(cropped_image), 'Sobel Edge Detector':sobel_edge(cropped_image)}
                    # feat_ls = []
                    # for item in feature_list:
                    #     print(features[item])
                    #     feat_ls.append(features[item])
                    # feat_ls.append(label)
                    # data.append(feat_ls)
                    data.append([HOG(cropped_image), label])
                    counter = counter + 1

    # write data into pickle file
    pick_in = open('data.pickle', 'wb')
    pickle.dump(data, pick_in)
    pick_in.close()

    pick_in = open('data.pickle', 'rb')
    img_data = pickle.load(pick_in)
    pick_in.close()
    return img_data

def split_train_test_data(temp_data_train_test):
    random.shuffle(temp_data_train_test)
    features = []
    labels = []
    # for feature1, feature2, label in data:
    #     f1 = np.array(feature1.flatten())
    #     f2 = np.array(feature2.flatten())
    #     features.append(np.concatenate((f1,f2), axis=None))
    #     labels.append(label)

    for feature1, label in temp_data_train_test:
        features.append(feature1.flatten())
        # features.append(feature2)
        labels.append(label)

    #Separate the data into training and test datasets
    X_train, X_test, Y_train, Y_test = train_test_split(features, labels, test_size=0.3)

    # Checking performance on the training set
    return X_train, X_test, Y_train, Y_test

def train_knn(temp_X_train, temp_X_test, temp_Y_train, temp_Y_test):

    #Grid search best k-NN parameters
    parameters = {'n_neighbors':list(range(1, 20)) ,'weights': ['uniform', 'distance']}
    # create an instance of the knn classifier
    knn_grid_tuned = KNeighborsClassifier()
    # create an instance of grid search with the above parameters
    grid_search = GridSearchCV(knn_grid_tuned, parameters, cv=2, scoring='accuracy', return_train_score=True, verbose=10)
    # fit the grid search with training set
    grid_search.fit(temp_X_train, temp_Y_train)
    # retrieve the best estimator
    knn_tuned = grid_search.best_estimator_
    print(knn_tuned)
    print(accuracy_score(temp_Y_test, knn_tuned.predict(temp_X_test)))

    #Saves the model in 'model.sav' folder
    pick = open('knn_model.sav', 'wb')
    pickle.dump(knn_tuned, pick)
    pick.close()

def train_svm(temp_X_train, temp_X_test, temp_Y_train, temp_Y_test):

    #Define a parameter grid for the SVM model
    param_grid = {'C': [0.1, 1, 10, 100, 1000],
                  'gamma': [0.01, 0.001, 0.0001],
                  'kernel': ['rbf', 'poly', 'linear', 'sigmoid']}

     #Define the SVM model
    svc = svm.SVC(probability=True)

    #Chooses the best parameters from param_grid for the SVM model
    grid_search = GridSearchCV(svc, param_grid, cv=3)
    #Trains the model on the specified training data
    grid_search.fit(temp_X_train, temp_Y_train)
    #Prints the best parameters that the model chose for the given data
    svm_tuned = grid_search.best_estimator_
    print(svm_tuned)

    #Saves the model in 'model.sav' folder
    pick = open('svm_model.sav', 'wb')
    pickle.dump(svm_tuned, pick)
    pick.close()
    pick = open('svm_model.sav', 'rb')
    model = pickle.load(pick)
    pick.close()

      #Testing phase: predict and store the predictions of the testing data in model_predictions
    model_predictions = model.predict(temp_X_test)
      #Print out a classification report for the model that includes: precision, accuracy, f-value, and recall
    print(classification_report(temp_Y_test, model_predictions))

def get_confusion_matrix(model, temp_X_test, temp_Y_test):
    #  #Plot the confusion matrix that includes the correctly and wrongly classified data
    # # cm = confusion_matrix(Y_test, model_predictions, normalize='all')
    # # cmd = ConfusionMatrixDisplay(cm, display_labels=categories)
    # # cmd.plot()

    fig, ax = plt.subplots(figsize=(500, 500))
    ax.set_xticks(range(0,62,1))
    ax.set_xticklabels(categories)
    plot_confusion_matrix(model, temp_X_test, temp_Y_test,values_format='d', ax=ax)
    plt.show()

    #Another method used to calculate the accuarcy
    accuracy = np.mean(model_predictions==temp_Y_test)
    print('Accuracy: ', accuracy)


categories = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'aa', 'B', 'bb', 'C', 'cc', 'D', 'dd', 'E',
              'ee', 'F', 'ff', 'G', 'gg', 'H', 'hh', 'I', 'ii', 'J', 'jj', 'K', 'kk', 'L', 'll', 'M', 'mm', 'N',
              'nn', 'O', 'oo', 'P', 'pp', 'Q', 'qq', 'R', 'rr', 'S', 'ss', 'T', 'tt', 'U', 'uu', 'V', 'vv', 'W',
              'ww', 'X', 'xx', 'Y', 'yy', 'Z', 'zz']

# categories = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']

feature_list = ['Canny Edge Detector', 'Sobel Edge Detector']

le = preprocessing.LabelEncoder()
le.fit_transform(categories)
mapping_labels = dict(zip(le.classes_, range(len(le.classes_))))
print(mapping_labels)

data = get_features("kNN_Dataset", mapping_labels, feature_list)

X_train, X_test, Y_train, Y_test = split_train_test_data(data)
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
jl.dump(sc, 'std_scaler.bin', compress=True)
train_knn(X_train, X_test, Y_train, Y_test)

# #For SVM
# print("Started training")
# train_svm(X_train,X_test, Y_train, Y_test)
# print("Done training")
# get_confusion_matrix(model, X_test, Y_test)


