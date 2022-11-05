from sklearn.neighbors import KNeighborsClassifier
from pickle import load
import pickle
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import SplineTransformer
from sklearn.model_selection import train_test_split, GridSearchCV
import joblib as jl

def train_knn(temp_X_train, temp_X_test, temp_Y_train, temp_Y_test):

    #Grid search best k-NN parameters
    parameters = {'n_neighbors':list(range(1, 20)) ,'weights': ['uniform', 'distance']}
    # create an instance of the knn classifier
    knn_grid_tuned = KNeighborsClassifier()
    # create an instance of grid search with the above parameters
    grid_search = GridSearchCV(knn_grid_tuned, parameters, cv=2, scoring='accuracy', return_train_score=True, verbose=10)
    # fit the grid search with training set
    grid_search.fit(temp_X_train, temp_Y_train)
    # retrieve the best estimator
    knn_tuned = grid_search.best_estimator_
    print(knn_tuned)
    print(accuracy_score(temp_Y_test, knn_tuned.predict(temp_X_test)))

    #Saves the model in 'model.sav' folder
    pick = open('knn_model.sav', 'wb')
    pickle.dump(knn_tuned, pick)
    pick.close()

X_train = load(open('data/X_train.pkl', 'rb'))
X_test = load(open('data/X_test.pkl', 'rb'))
Y_test = load(open('data/Y_test.pkl', 'rb'))
Y_train = load(open('data/Y_train.pkl', 'rb'))

# from sklearn.preprocessing import SplineTransformer
scaler = SplineTransformer()
X_train_scaled = scaler.fit_transform(X_train)
#we must apply the scaling to the test set that we computed for the training set
X_test_scaled = scaler.transform(X_test)
#Save fitted scaler model to be used with new images or else model will not predict well
jl.dump(scaler, 'std_scaler_knn.bin', compress=True)
train_knn(X_train, X_test, Y_train, Y_test)
