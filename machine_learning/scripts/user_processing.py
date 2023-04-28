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
    from data_processing.characters_categories import mapping_labels
except:
    # running from flask, base.py
    from machine_learning.scripts.data_processing.data_prep import data_prep
    from machine_learning.scripts.data_processing.characters_categories import mapping_labels


class user_processing:
    def __init__(self,learner=""):
        self.learner = learner
       

    @staticmethod
    def prepare_data(features=[]):
        #preparing the dataset using features selected by user
        # N.B.: CHANGHE THE mapping_labels (LABEL ENCDOER ) IF YOU ARE USING OTHER THAN ALPHANUMERIACL

        p1 = data_prep( features_array=features)
        p1.prep_data()

        ## preparing the image uploaded by the user
        p1 = data_prep( features_array=features, uploaded=True)
        p1.prep_data()

        ###########################################################################
        ###########################################################################
        ##################TUNING

        # p1 = data_prep(test_size=0.5, all_features=True)
        # p1.prep_data()

        
        
        # p1 = data_prep(test_size=0.5, all_features=True, uploaded=True)
        # p1.prep_data()
    

        ###########################################################################
        ###########################################################################


    def give_label(self):
        # relative path
        try:
            X_test = load(open('../processed_data/X_test_uploaded.pkl', 'rb'))
        except:
            try:
                X_test = load("../machine_learning/processed_data/X_test_uploaded.pkl", 'rb')
            except:
                print("IMAGE NOT PREPROCESSED")

        if self.learner=="SVM":
            # Setting up svm
            # relative import
            from learners.svm_learner import SVM_class
            svm = SVM_class(learner="SVC",minimal_grid_search=False,scaler="te")
            svm.train()
            # loading trained models and scaler
            # relative path
            try:
                estimator = load("../best_estimators/SVM_BEST.joblib")
                sc = load(open("../processed_data/SVM_fit_transformed.pkl", "rb"))
            except:
                estimator = load("../machine_learning/best_estimators/SVM_BEST.joblib")
                sc = load(open("../machine_learning/processed_data/SVM_fit_transformed.pkl", "rb"))
                        # relative path
            try:
                X_test = load(open('../processed_data/X_test_uploaded.pkl', 'rb'))
            except:
                try:
                    X_test = load("../machine_learning/processed_data/X_test_uploaded.pkl", 'rb')
                except:
                    print("IMAGE NOT PREPROCESSED")

                    
            sc1 = sc
            #transformation for X_test
            X_test = sc1.transform(X_test)

            # predicting data if everything but lowercase & uppercase
            mapping_labels2 = mapping_labels
            label = {i for i in mapping_labels2 if mapping_labels2[i]==model_prediction[0]}
            expected_label=str(label.pop())[0]
            print ("SVM expected")
            print(expected_label)

            probability=np.max(estimator.predict_proba(X_test))
            print("SVM probability")
            print(probability)

            return [expected_label,probability]
            
        
        if self.learner=="KNN":
            # setting up knn
            # relative import
            from learners.knn_learner import KNN_class
            knn = KNN_class(scaler="te")
            knn.train()
            # loading up trained model and scaler
            # relative path
            try:
                estimator = load("../best_estimators/KNN_BEST.joblib")
                sc = load(open("../processed_data/KNN_fit_transformed.pkl", "rb"))
            except:
                estimator = load("../machine_learning/best_estimators/KNN_BEST.joblib")
                sc = load(open("../machine_learning/processed_data/KNN_fit_transformed.pkl", "rb"))
                    # relative path
            try:
                X_test = load(open('../processed_data/X_test_uploaded.pkl', 'rb'))
            except:
                try:
                    X_test = load("../machine_learning/processed_data/X_test_uploaded.pkl", 'rb')
                except:
                    print("IMAGE NOT PREPROCESSED")

        
            sc1 = sc
            #transformation for X_Test
            X_test = sc1.transform(X_test)

            # predicting data if everything but lowercase & uppercase
            model_prediction = estimator.predict(X_test)

            print("PRDICIOT")
            print(model_prediction)

            label = {i for i in mapping_labels if mapping_labels[i]==model_prediction[0]}
            print("LABEL")
            print(label)
            expected_label=str(label.pop())[0]
            print("POPPED LABEL UPDATE")
            print(label)
            print ("KNN expected")
            print(expected_label)

            probability=np.max(estimator.predict_proba(X_test))
            print("KNN probability")
            print(probability)

            return [expected_label,probability]
            

        
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
                

            model_prediction = estimator.predict(X_test)
            label = {i for i in mapping_labels if mapping_labels[i]==model_prediction[0]}
            expected_label=str(label.pop())[0]
            print ("Ensemble expected")
            print(expected_label)

            probability=np.max(estimator.predict_proba(X_test))
            print("Ensemble probability")
            print(probability)

            return [expected_label,probability]

            
        if self.learner=="PreTrained":
            
            print("pre-trained ONNNNNN")
            
            #p1 = data_prep(numerical = True, features_array=["HOG"], uploaded=True)
            p1 = data_prep(numerical = True, features_array=["sobel_edge","HOG","pixel_intensity"], uploaded=True)
            p1.prep_data()
            # relative path
            try:
                X_test = load(open('../processed_data/X_test_uploaded.pkl', 'rb'))
            except:
                try:
                    X_test = load("../machine_learning/processed_data/X_test_uploaded.pkl", 'rb')
                except:
                    print("IMAGE NOT PREPROCESSED")

           
            try:
                estimator = load("../machine_learning/models/svm_HOG1/svm_model3.joblib")
                sc = load(open("../models/svm_HOG1/std_scaler3.pkl", "rb"))
            except:
                estimator = load("../machine_learning/models/svm_HOG1/svm_model3.joblib")
                sc = load(open("../machine_learning/models/svm_HOG1/std_scaler3.pkl", "rb"))


            sc1 = sc
        

            #transformation for knn
            X_test = sc1.transform(X_test)
            
            model_prediction = estimator.predict(X_test)
            label = {i for i in mapping_labels if mapping_labels[i]==model_prediction[0]}
            expected_label=str(label.pop())[0]
            print ("Ensemble expected")
            print(expected_label)

            probability=np.max(estimator.predict_proba(X_test))
            print("Ensemble probability")
            print(probability)

            return [expected_label,probability]






