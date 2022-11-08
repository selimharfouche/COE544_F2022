##################### Path configuration start #####################
import sys
from pathlib import Path # if you haven't already done so
file = Path(__file__).resolve()
parent, root = file.parent, file.parents[1]
sys.path.append(str(root))

# Additionally remove the current file's directory from sys.path
try:
    sys.path.remove(str(parent))
except ValueError: # Already removed
    pass


##################### Path configuration end #####################


from joblib import load
import numpy as np

# relative import
try:
    # running from Main.py
    from data_processing.data_prep import data_prep
except:
    # running from flask, base.py
    from machine_learning.scripts.data_processing.data_prep import data_prep


class user_processing:
    def __init__(self,learner=""):
        self.learner = learner
       

    @staticmethod
    def prepare_data(features=[]):
        # preparing the dataset using features selected by user
        # N.B.: WE ARE USING NUMERICAL FOR THE SPEED
        p1 = data_prep( numerical=True,features_array=features)
        p1.prep_data()

        # preparing the image uploaded by the user
        p1 = data_prep(numerical = True, features_array=features, uploaded=True)
        p1.prep_data()


    def give_label(self):
        # relative path
        try:
            X_test = load(open('../processed_data/X_test_uploaded.pkl', 'rb'))
        except:
            X_test = load("../machine_learning/processed_data/X_test_uploaded.pkl", 'rb')

        if self.learner=="SVM":
            # relative import
            from learners.svm_learner import SVM_class
            svm = SVM_class(learner="SVC",minimal_grid_search=False,scaler="te")
            svm.train()
            # relative path
            try:
                estimator = load("../best_estimators/SVM_BEST.joblib")
                sc = load(open("../processed_data/SVM_fit_transformed.pkl", "rb"))
            except:
                estimator = load("../machine_learning/best_estimators/SVM_BEST.joblib")
                sc = load(open("../machine_learning/processed_data/SVM_fit_transformed.pkl", "rb"))
                

            #transformation for svm
            X_test = sc.transform(X_test)

            print ("SVM expected label")
            print(estimator.predict(X_test))
            print("SVM probability")
            print(np.max(estimator.predict_proba(X_test)))
            return [estimator.predict(X_test),np.max(estimator.predict_proba(X_test))]
            
        
        if self.learner=="KNN":
            # relative import
            from learners.knn_learner import KNN_class
            knn = KNN_class(scaler="te")
            knn.train()
            # relative path
            try:
                estimator = load("../best_estimators/KNN_BEST.joblib")
                sc = load(open("../processed_data/KNN_fit_transformed.pkl", "rb"))
            except:
                estimator = load("../machine_learning/best_estimators/KNN_BEST.joblib")
                sc = load(open("../machine_learning/processed_data/KNN_fit_transformed.pkl", "rb"))
        

            #transformation for knn
            X_test = sc.transform(X_test)

            print ("KNN expected label")
            print(estimator.predict(X_test))
            print("KNN probability")
            print(np.max(estimator.predict_proba(X_test)))
            try:
                return [estimator.predict(X_test),np.max(estimator.predict_proba(X_test))]
            # therefore it is a linear SV
            except:
                return estimator.predict(X_test)


        
        if self.learner=="ensemble":
            # relative import
            from learners.voting_classifier import voting_classifier_class
            ensemble=voting_classifier_class()
            ensemble.train()
            # relative path
            try:
                estimator = load("../best_estimators/ENSEMBLE.joblib")

            except:
                estimator = load("../machine_learning/best_estimators/ENSEMBLE.joblib")
                
            print("ensemble expected label")
            print(estimator.predict(X_test))
            print("ensemble probability")
            print(np.max(estimator.predict_proba(X_test)))

            return [estimator.predict(X_test),np.max(estimator.predict_proba(X_test))]


        if self.learner=="PreTrained":
            p1 = data_prep(numerical = True, features_array=["HOG"], uploaded=True)
            p1.prep_data()
            print("PREPPED HOG")

            # relative path
            try:
                estimator = load("../models/svm_HOG/svm_model.sav")
                sc = load(open("../models/svm_HOG/std_scaler.bin", "rb"))
            except:
                estimator = load("../machine_learning/models/svm_HOG/svm_model.sav")
                sc = load(open("../machine_learning/models/svm_HOG/std_scaler.bin", "rb"))
            
            print("SICCCESSFULL IMPORTED MODEL")
            X_test = sc.transform(X_test)
            print ("pre trained SVM expected label")
            print(estimator.predict(X_test))
            print("pre trained probability")
            print(np.max(estimator.predict_proba(X_test)))

                
            return [estimator.predict(X_test),np.max(estimator.predict_proba(X_test))]
            


            
        





