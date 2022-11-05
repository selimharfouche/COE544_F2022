


from pickle import dump

import cv2, glob, os, pickle, random



from sklearn.model_selection import *
from A_helper_data_prep import *


class data_prep():

    def __init__(self):
        self.counter = 0
        self.data = []
        self.DIRECTORY_TRAINING_DATA = 'datasets/A'
        self.Windows_Iteration ='\\*.png'
        self.Mac_Iteration='//*.png'
        self.TEST_SIZE = 0.3
        self.categories = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
        'a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z',
        'a\'','b\'','c\'','d\'','e\'','f\'','g\'','h\'','i\'','j\'','k\'','l\'','m\'','n\'','o\'','p\'','q\'','r\'','s\'','t\'','u\'','v\'','w\'','x\'','y\'','z\'']
        


    def read_image(self):
        categories=self.categories
        DIRECTORY_TRAINING_DATA = self.DIRECTORY_TRAINING_DATA 
        DIRECTORY_TRAINING_DATA = self.DIRECTORY_TRAINING_DATA 
        Mac_Iteration=self.Mac_Iteration
        TEST_SIZE =self.TEST_SIZE
        categories=self.categories
        
        
        for category in categories:
            path = os.path.join(DIRECTORY_TRAINING_DATA, category)
            self.label = categories.index(category)
            for img in glob.glob(path + Mac_Iteration):
                if (img is not None):
                  
                    img0 = cv2.imread(img) # Image read
                    print("image read"+str(self.counter))
                    self.process_image(img0)

        self.save_data()
    def process_image(self, img0):
        
        
        
        self.thresh = convert_BW(img0)
        self.contours,_ = cv2.findContours(self.thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        self.compute_bounding_rect()
        self.crop_image()

    def compute_bounding_rect(self):
        if(len(self.contours)>1):
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
            cnt1 = 0
            # compute rectangle (minimum area)
            self.X, self.Y, self.W, self.H = cv2.boundingRect(cnt0)
        print("COMPUTER MIN BOUNDING RECT")



        
    def crop_image(self):
    # crop image following the rectangle
        cropped_image = self.thresh[int(self.Y):int(self.Y + self.H), int(self.X):int(self.X + self.W)]
        cropped_image = cv2.resize(cropped_image, (10,10), interpolation=cv2.INTER_AREA)
        feature1=pixel_intensity(cropped_image).flatten()
        self.data.append([feature1,self.label])
        self.counter = self.counter + 1
        



    def save_data(self):
            print("AAAAAAA")
            pick_in = open('data.pickle','wb')
            pickle.dump(self.data, pick_in)
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
            X_train, X_test, Y_train, Y_test = train_test_split(features, labels, test_size=self.TEST_SIZE)

            dump(X_train, open('data/X_train.pkl', 'wb'))
            dump(X_test, open('data/X_test.pkl', 'wb'))
            dump(Y_test, open('data/Y_test.pkl', 'wb'))
            dump(Y_train, open('data/Y_train.pkl', 'wb'))

