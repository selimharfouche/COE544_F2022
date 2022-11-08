##################### Path configuration start #####################
import sys, os
from pathlib import Path # if you haven't already done so
file = Path(__file__).resolve()
parent, root = file.parent, file.parents[1]
sys.path.append(str(root))

# Additionally remove the current file's directory from sys.path
try:
    sys.path.remove(str(parent))
except ValueError: # Already removed
    pass
def path_corrector(path):
    if not os.path.isdir(path):
        path ='/machine_learning'+str(path)
        path=path.replace(".", "")
        path=".."+path
        print("NEW PATH")
    return path
##################### Path configuration end #####################



import cv2, glob, os, random
from sklearn import preprocessing
from sklearn.model_selection import *
from pickle import dump
from sklearn import preprocessing

#relative import
from data_processing.helper_data_prep import *


class data_prep:
    def __init__(
        self,
        microsoft_windows=False,
        test_size=0.3,
        #relative path
        directory_training_data="../datasets/A",
        directory_processed_data="../processed_data/",
        directory_uploaded_image="../datasets/uploaded_images/",
        numerical=False,
        alphabetical=False,
        lower_case=False,
        upper_case=False,
        all_features=False,
        features_array=[],
        uploaded=False,
    ):
        self.uploaded=uploaded
        print(os.getcwd())

        # setting the iteration parameter depending on the operating system
        if microsoft_windows:
            self.iteration = "\\*.png"
        else:
            self.iteration = "//*.png"

        # test size to be taken
        self.test_size = test_size

        # directory containing the training data
        self.directory_training_data = path_corrector(directory_training_data)
        


        # list of feature, label
        self.data = []

        # selecting characters to train on
        # relative import
        if numerical:
            from data_processing.characters_categories import numerical_category
            self.categories = numerical_category
                

        # needs further implementation with the label encoder feature
        # elif alphabetical:
        #     from data_processing.characters_categories import alphabetical_category
        #     self.categories = alphabetical_category
        # elif lower_case:
        #     from data_processing.characters_categories import lower_case_category
        #     self.categories = lower_case_category

        # elif upper_case:
        #     from data_processing.characters_categories import upper_case_category
        #     self.categories = upper_case_category
           
        else:  # by default, alphanumerical
            from data_processing.characters_categories import alphanumerical_category
            self.categories = alphanumerical_category

        if all_features:
            # setting up the features to be used
            self.selected_features = [
                "aspect_ratio",
                "top_half_img",
                "lower_half_img",
                "right_half_img",
                "left_half_img",
                "histogram",
                "pixel_intensity",
                "sobel_edge",
                "canny_edge",
                "LocalBinaryPatterns",
                "HOG",
            ]
        else:
            self.selected_features=features_array

        print("features to be used:")
        for feature in self.selected_features:
             print(feature)

        # Counter for total number of images
        self.counter = 0

        # Added LabelEncoder (encodes class labels into integers)
        le = preprocessing.LabelEncoder()
        le.fit_transform(self.categories)
        self.mapping_labels = dict(zip(le.classes_, range(len(le.classes_))))
        self.label = None
        self.directory_processed_data=path_corrector(directory_processed_data)
        self.directory_uploaded_image=path_corrector(directory_uploaded_image)
        

        

    def prep_data(self):
        if self.uploaded:
            # relative path
            img0 = cv2.imread(self.directory_uploaded_image+"website_upload.png")
            # print("image read success")
            self.label="+"
            # convert to black and white and identify contours
            self.process_image(img0)
            # compute bouding retangle
            self.compute_bounding_rect()
            # extract image from bonding rectangle
            self.crop_image()
            # extract features
            self.extract_features()
            features = []
            labels = []
            for features1, label in self.data:
                features.append(features1)
                labels.append(label)
            X_test=features
            #relative path
            dump(X_test, open(self.directory_processed_data+"X_test_uploaded.pkl", "wb"))
        else:
            
            for category in self.mapping_labels.keys():
                path = os.path.join(self.directory_training_data, category)

   
                # setting label to be examined
                self.label = self.mapping_labels[category]

                # reading images corresponding to the label
                for img in glob.glob(path + self.iteration):
                    if img is not None:
                        img0 = cv2.imread(img)
                        # convert to black and white and identify contours
                        self.process_image(img0)
                        # compute bouding retangle
                        self.compute_bounding_rect()
                        # extract image from bonding rectangle
                        self.crop_image()
                        # extract features
                        self.extract_features()
            self.save_data()

    def process_image(self, img0):

        # convert to black and white
        self.thresh = convert_BW(img0)

        # identify contours
        self.contours, _ = cv2.findContours(
            self.thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
        )



    def compute_bounding_rect(self):
        if len(self.contours) > 1:
            x0, y0, w0, h0 = cv2.boundingRect(self.contours[0])
            rect0 = x0, y0, w0, h0
            x1, y1, w1, h1 = cv2.boundingRect(self.contours[1])
            rect1 = x1, y1, w1, h1
            # merge 2 rectanlges
            self.X, self.Y, self.W, self.H = union(rect0, rect1)
            rect_temp = self.X, self.Y, self.W, self.H
            for cnt0 in self.contours[2::1]:
                # get the rest of the contour points
                x0, y0, w0, h0 = cv2.boundingRect(cnt0)
                rect2 = x0, y0, w0, h0
                # merge rectangles
                self.X, self.Y, self.W, self.H = union(rect_temp, rect2)
                rect_temp = self.X, self.Y, self.W, self.H
        else:
            cnt0 = self.contours[0]
            # compute rectangle (minimum area)
            self.X, self.Y, self.W, self.H = cv2.boundingRect(cnt0)

    def crop_image(self):
        # crop image following the rectangle
        cropped_image = self.thresh[
            int(self.Y) : int(self.Y + self.H), int(self.X) : int(self.X + self.W)
        ]
        cropped_image = cv2.resize(
            cropped_image, (32, 32), interpolation=cv2.INTER_AREA
        )
        self.cropped_image=cropped_image
        

    def extract_features(self):
        cropped_image=self.cropped_image
        features = []
        myFuncs = {
            "aspect_ratio": aspect_ratio,
            "top_half_img": top_half_img,
            "lower_half_img": lower_half_img,
            "right_half_img": right_half_img,
            "left_half_img": left_half_img,
            "histogram": histogram,
            "pixel_intensity": pixel_intensity,
            "sobel_edge": sobel_edge,
            "canny_edge": canny_edge,
            "HOG": HOG,
            "LocalBinaryPatterns": LocalBinaryPatterns,
        }

        for feature in self.selected_features:
            try:
                features = np.append(features, myFuncs[feature](cropped_image))
            except:
                try:
                    features = np.append(
                        features, myFuncs[feature](cropped_image).flatten()
                    )
                except:
                    try:
                        features = np.append(features, myFuncs[feature](24, 8, cropped_image))
                    except:
                        pass
                
            
            
                
            
            

        self.data.append([features, self.label])
        self.counter = self.counter + 1

    def save_data(self):
        print("Number of images processed: " + str(self.counter))
        data = self.data
        random.shuffle(data)
        features = []
        labels = []
        for features1, label in data:
            features.append(features1)
            labels.append(label)
        # print("length of features")
        # print(len(features[0]))

        X_train, X_test, Y_train, Y_test = train_test_split(
            features, labels, test_size=self.test_size
        )
        # print("length of X_test")
        # print(len(X_test[0]))

        # relative path
        
        dump(X_train, open(self.directory_processed_data+"X_train.pkl", "wb"))
        dump(X_test, open(self.directory_processed_data+"X_test.pkl", "wb"))
        dump(Y_test, open(self.directory_processed_data+"Y_test.pkl", "wb"))
        dump(Y_train, open(self.directory_processed_data+"Y_train.pkl", "wb"))
