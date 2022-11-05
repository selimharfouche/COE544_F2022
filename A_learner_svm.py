############################################################
# SVM Trainer
# Trains an SVM model (SVC, NuSVC or LinearSVC) 
# Does preprocessing SplineTransformer or StandardScaler
# Transform Data (A FAIRE PLUS DE RECHERCHE)
# Learn Digits on train subsets (A FAIRE PLUS DE RECHERCHE)
# Predict the value of the digit on the test subset (A FAIRE PLUS DE RECHERCHE)
# Perform GridSearchCV with predefined parameters
# save the best model to best_estimators

############################################################

######################################################################
############################## IMPORTS ###############################
######################################################################

from sklearn import svm, metrics
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler, SplineTransformer
from sklearn.model_selection import GridSearchCV
from joblib import load
import joblib as jl
import pickle

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

X_train = load(open('data/X_train.pkl', 'rb'))
X_test = load(open('data/X_test.pkl', 'rb'))
Y_test = load(open('data/Y_test.pkl', 'rb'))
Y_train = load(open('data/Y_train.pkl', 'rb'))

# from sklearn.preprocessing import SplineTransformer
scaler = SplineTransformer()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
#Save fitted scaler model to be used with new images or else model will not predict well
jl.dump(scaler, 'std_scaler_svm.bin', compress=True)


train_svm(X_train,X_test, Y_train, Y_test)