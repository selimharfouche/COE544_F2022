import cv2, glob, os, random
from sklearn import preprocessing
from sklearn.model_selection import *
from A_helper_data_prep import *
from pickle import dump
from sklearn import preprocessing


class data_prep:
    def __init__(
        self,
        microsoft_windows=False,
        test_size=0.3,
        directory_training_data="datasets/A",
        numerical=False,
        alphabetical=False,
        lower_case=False,
        upper_case=False,
        all_features=False,
        feature1="",
        feature2="",
        feature3="",
        feature4="",
        feature5="",
        feature6="",
        feature7="",
        feature8="",
        feature9="",
        feature10="",
        feature11="",
    ):

        # setting the iteration parameter depending on the operating system
        if microsoft_windows:
            self.iteration = "\\*.png"
        else:
            self.iteration = "//*.png"

        # test size to be taken
        self.test_size = test_size

        # directory containing the training data
        self.directory_training_data = directory_training_data

        # list of feature, label
        self.data = []

        # selecting characters to train on
        if numerical:
            from characters_categories import numerical_category
            self.categories = numerical_category
                
        elif alphabetical:
            from characters_categories import alphabetical_category
            self.categories = alphabetical_category
        elif lower_case:
            from characters_categories import lower_case_category
            self.categories = lower_case_category

        elif upper_case:
            from characters_categories import upper_case_category
            self.categories = upper_case_category
           
        else:  # by default, alphanumerical
            from characters_categories import alphanumerical_category
            self.categories = alphanumerical_category

        if all_features:
            feature1 = "aspect_ratio"
            feature2 = "top_half_img"
            feature3 = "lower_half_img"
            feature4 = "right_half_img"
            feature5 = "left_half_img"
            feature6 = "histogram"
            feature7 = "pixel_intensity"
            feature8 = "sobel_edge"
            feature9 = "canny_edge"
            feature10 = "LocalBinaryPatterns"
            feature11 = "HOG"

        # setting up the features to be used
        self.selected_features = [
            feature1,
            feature2,
            feature3,
            feature4,
            feature5,
            feature6,
            feature7,
            feature8,
            feature9,
            feature10,
            feature11,
        ]

        # Counter for total number of images
        self.counter = 0
        # Added LabelEncoder (encodes class labels into integers)
        le = preprocessing.LabelEncoder()
        le.fit_transform(self.categories)
        self.mapping_labels = dict(zip(le.classes_, range(len(le.classes_))))
        self.label = None

    def prep_data(self):
        for category in self.mapping_labels.keys():
            path = os.path.join(self.directory_training_data, category)

            # setting label to be examined
            self.label = self.mapping_labels[category]

            # reading images corresponding to the label
            for img in glob.glob(path + self.iteration):
                if img is not None:
                    img0 = cv2.imread(img)
                    self.process_image(img0)

        self.save_data()

    def process_image(self, img0):

        # convert to black and white
        self.thresh = convert_BW(img0)

        # identify contours
        self.contours, _ = cv2.findContours(
            self.thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
        )

        # compute bouding retangle
        self.compute_bounding_rect()

        # extract image from bouding rectangle
        self.crop_image()

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
            cropped_image, (10, 10), interpolation=cv2.INTER_AREA
        )
        self.extract_features(cropped_image)

    def extract_features(self, cropped_image):
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
            #"LocalBinaryPatterns": LocalBinaryPatterns,
        }

        for feature in self.selected_features:
            try:
                features = np.append(
                    features, myFuncs[feature](cropped_image).flatten()
                )
            except:
                pass
            try:
                features = np.append(features, myFuncs[feature](cropped_image))
            except:
                pass
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
        X_train, X_test, Y_train, Y_test = train_test_split(
            features, labels, test_size=self.test_size
        )

        dump(X_train, open("data/X_train.pkl", "wb"))
        dump(X_test, open("data/X_test.pkl", "wb"))
        dump(Y_test, open("data/Y_test.pkl", "wb"))
        dump(Y_train, open("data/Y_train.pkl", "wb"))
