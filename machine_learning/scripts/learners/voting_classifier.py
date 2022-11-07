

from sklearn.ensemble import VotingClassifier
from sklearn.metrics import accuracy_score
from joblib import load, dump



class voting_classifier_class():
    def __init__(self,voting="soft"):
        self.voting=voting

    def train(self):
        # relative path
        X_train = load(open('../processed_data/X_train.pkl', 'rb'))
        Y_train = load(open('../processed_data/Y_train.pkl', 'rb'))

        svm_best = load("../best_estimators/SVM_BEST.joblib")
        knn_best = load("../best_estimators/KNN_BEST.joblib")

        estimators=[("knn", knn_best), ("svm", svm_best)]#create our voting classifier, inputting our models

        ensemble = VotingClassifier(estimators, voting=self.voting)
        
        #fit model to training data
        ensemble.fit(X_train, Y_train)#test our model on the test data
        dump(ensemble, "../best_estimators/ENSEMBLE.joblib")
        
        #print(ensemble.score(X_test, Y_test))

# # Voting Classifier with hard voting
# vot_hard = VotingClassifier(estimators, voting ='hard')
# vot_hard.fit(X_train, Y_train)
# y_pred = vot_hard.predict(X_test)
  
# # using accuracy_score metric to predict accuracy
# score = accuracy_score(Y_test, y_pred)
# print("Hard Voting Score % d" % score)
  
# # Voting Classifier with soft voting
# vot_soft = VotingClassifier(estimators, voting ='soft')
# vot_soft.fit(X_train, Y_train)
# y_pred = vot_soft.predict(X_test)
  
# # using accuracy_score
# score = accuracy_score(Y_test, y_pred)
# print("Soft Voting Score % d" % score)

