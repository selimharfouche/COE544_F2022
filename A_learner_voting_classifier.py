from A_learner_svm import svm_best
from A_learner_rf import rf_best
from A_learner_knn import knn_best
from A_data_prep import X_test, X_train, Y_test, Y_train
from sklearn.ensemble import VotingClassifier#create a dictionary of our models
from sklearn.metrics import accuracy_score
estimators=[("knn", knn_best), ("rf", rf_best), ("svm", svm_best)]#create our voting classifier, inputting our models
ensemble = VotingClassifier(estimators, voting="hard")
#fit model to training data
ensemble.fit(X_train, Y_train)#test our model on the test data
print("AAAAAA")
print(ensemble.score(X_test, Y_test))
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

