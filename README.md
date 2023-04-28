# Intelligent Engineering Algorithms, Fall 2022, Dr Joe Tekli

## Intelligent hand-written character recognizer agent

## Project Description - project 1
The first project is to realize an intelligent hand-written character recognizer agent, to recognize a person’s hand-written characters, as a first step toward recognizing full words or sentences later-on. 

## Features Exctracted
- aspect ratio
- pixel count (top half, lower half, right half and left half)
- projection histogram
- pixel intensity
- sobel edge
- canny edge
- local binary patterns 
- HOG



## Non-parametric learners used
- KNN
- SVM 
- Ensemble using KNN and SVM

## Datasets Used

## machine_learning - directories

This is the directory containing all the machine learning algorithms, including the datasets that we are using 

### 1. Best_estimators
Directory containing the saved best estimator as *.joblib. 
This estimator is obtained by performing a gridsearch for KNN and SVM.


### 2. Datasets
Directory containing the datasets used
Every dataset will contain directories named by the corresponding label. 
- uploaded_images will contain 1 image, website_upload.png corresponding to the image submitted by the user. It is created when the user submits it. If another image is submited, the original one will be replaced

### 3. Models 
Directory containing pre-trained models along with their scaler

### 4. Plots
Containg confusion matrix of SVM.
! don't work when running the code from backend/base.py 


### 5. Processed_data

Directory containg the processed data
- X_test_uploaded : features of the uploaded/drawn image by the user
- X_test
- X_train
- X_train
- Y_test
image drawn by user processed
- X_test_transformed_knn.pkl
- X_test_transformed_svm.pkl

and the scalers of the best learners received after gridsearch
- SVM_fit_transformed.pkl
- KNN_fit_transformed.pkl

### 6. Scripts
#### Data processing

#### characters_categories
Beautified version of the categories' list. 
Imported in data_prep to setup the categories
containes helper to setup the labels, along with mapping_label which maps labels to their corresponding value : ex: a is 11

#### data_prep
- Goal: pre-process the data and extract its features
- Requires: image dataset
- Setup using a class constructor
    - microsoft_windows: boolean, set to true, will allow data extraction on windows using the correct path
    - test_size: float, setting one parameter of train_test_split
    - directory_training_data: string, directory corrresponding to the training data
    - numerical: boolean, allowing data processing on digits only (dataset must constain directories named by digits)
    - alphabetical: boolean, allowing data processing on alphabetical charactes only (dataset must constain directories named by characters of the alphabet, upper case will correspond to the letter followed by ' i.e.: a' )
    - lower_case: boolean, allowing data processing on lower case alphabetical charactes only (dataset must constain directories named by characters of the alphabet) 
    - upper_case: allowing data processing on lower case alphabetical charactes only (dataset must constain directories named by characters of the alphabet, upper case will correspond to the letter followed by ' i.e.: a' )
    - all_features: boolean, set to true will extract all disponible features
    - features_array: string array, corresponds to the features to be extracted
    - uploaded: boolean, if set to true will allow feature extraction for the uploaded images
    - N.B: label encoder needs to be updated to process the lowercase, uppercase by adding another variable. A big issue arose and it was variables and code hierarchy in Python.

- output: the processed data in the data directory inside of machine_learning
    - X_train.pkl
    - X_test.pkl
    - Y_test.pkl
    - Y_train.pkl

#### helper_data_prep
Contains helpers to data_prep along with the features extraction methods

### Learners
#### knn_learner
- Goal: perfom knn on the prepared data
- Requires trained data in "/machine_learning/data/*.pkl"
    - X_train.pkl
    - X_test.pkl
    - Y_test.pkl
    - Y_train.pkl
- Setup using a class contructor
    - scaler: string, to either use SplineTransformer or StandardScaler
    - verbose: terminal output while performing gridsearch
KNN learner using sklearn


#### Main
- for the experienced user to extensively try the features in the repo. Most of them are unavaible to be set on the website. 

#### svm learner
- Goal: perfom svm on the prepared data
- Requires trained data in "/machine_learning/data/*.pkl"
    - X_train.pkl
    - X_test.pkl
    - Y_test.pkl
    - Y_train.pkl
- Setup using a class contructor
    - learner: string, allows SVC, NuSVC & LinearSVC
    - scaler: string, allows StandarScaler & SplineTransformer
    - confusion_matrix: boolean, to show the confusion matrix
    - classification_report: boolean, to show the classification report
    - minimal_grid_search: boolean, if set to false will lead to a relative extensive gridsearch
    - k_fold: int, for cross validation
    - verbose: terminal output while performing gridsearch


#### voting_classifier


### user processing
- Goal: combine learner and data prep script to predict the image uploaded by the user

- staticmethod: prepare_data
    - prepares the data depending on the features array given

- give_label: returns a string: the prediction of the of the image along with its probability

