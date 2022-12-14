from data_prep import data_prep
from joblib import load
import numpy as np

class user_processing:
    def __init__(self,learner=""):
        self.learner = learner
       

    @staticmethod
    def prepare_data(features=[]):
        # preparing the dataset using features selected by user
        # N.B.: WE ARE USING NUMERICAL FOR THE SPEED
        p1 = data_prep( features_array=features)
        p1.prep_data()

        # preparing the image uploaded by the user
        p1 = data_prep(numerical = True, features_array=features, uploaded=True)
        p1.prep_data()


    def give_label(self):

        if self.learner=="SVM":
            print("SVM IS SELECTED")
            from A_learner_svm import SVM_class
            svm = SVM_class(confusion_matrix=False,minimal_grid_search=False,scaler="te")
            svm.train()
            estimator = load("best_estimators/SVM_BEST.joblib")
            X_test = load(open('data/X_test_uploaded.pkl', 'rb'))
            sc = load(open("data/SVM_fit_transformed.pkl", "rb"))

            #transformation for svm
            X_test = sc.transform(X_test)

            print ("expected label")
            print(estimator.predict(X_test))
            print("probability")
            print(np.max(estimator.predict_proba(X_test)))
            
        
        if self.learner=="KNN":
            from A_learner_knn import KNN_class
            knn = KNN_class(scaler="te")
            knn.train()
            estimator = load("best_estimators/KNN_BEST.joblib")
            X_test = load(open('data/X_test_uploaded.pkl', 'rb'))
            sc = load(open("data/KNN_fit_transformed.pkl", "rb"))

            #transformation for knn
            X_test = sc.transform(X_test)

            print ("expected label")
            print(estimator.predict(X_test))
            print("probability")
            print(np.max(estimator.predict_proba(X_test)))
        
        if self.learner=="ensemble":
            from A_learner_voting_classifier import voting_classifier_class
            ensemble=voting_classifier_class()
            ensemble.train()
            estimator = load("best_estimators/ENSEMBLE.joblib")
            X_test = load(open('data/X_test_uploaded.pkl', 'rb'))
            print(estimator.predict(X_test))


            
        





