from sklearn.ensemble import RandomForestClassifier#create a new random forest classifier
from A_data_prep import X_train, X_test, Y_train, Y_test
from sklearn.model_selection import GridSearchCV

from sklearn.ensemble import RandomForestClassifier#create a new random forest classifier
rf = RandomForestClassifier()#create a dictionary of all values we want to test for n_estimators
params_rf = {"n_estimators": [50, 100, 200]}#use gridsearch to test all values for n_estimators
rf_gs = GridSearchCV(rf, params_rf, cv=5)#fit model to training data
rf_gs.fit(X_train, Y_train)
#save best model
rf_best = rf_gs.best_estimator_#check best n_estimators value
print(rf_gs.best_params_)