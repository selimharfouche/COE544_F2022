from sklearn.neighbors import KNeighborsClassifier
from pickle import load
import pickle
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import SplineTransformer
from sklearn.model_selection import train_test_split, GridSearchCV
from joblib import load, dump



class KNN_class():
    def __init__(self) -> None:
        pass

    def train(self):
        X_train = load(open('data/X_train.pkl', 'rb'))
        X_test = load(open('data/X_test.pkl', 'rb'))
        Y_test = load(open('data/Y_test.pkl', 'rb'))
        Y_train = load(open('data/Y_train.pkl', 'rb'))

        #Grid search best k-NN parameters
        parameters = {'n_neighbors':list(range(1, 20)) ,'weights': ['uniform', 'distance']}

        # create an instance of the knn classifier
        knn_grid_tuned = KNeighborsClassifier()

        # create an instance of grid search with the above parameters
        grid = GridSearchCV(knn_grid_tuned, parameters, cv=2, scoring='accuracy', return_train_score=True, verbose=10)

        
        

   

        # from sklearn.preprocessing import SplineTransformer
        scaler = SplineTransformer()
        X_train = scaler.fit_transform(X_train)
        #we must apply the scaling to the test set that we computed for the training set
        X_test = scaler.transform(X_test)
        #Save fitted scaler model to be used with new images or else model will not predict well
        dump(scaler, open("data/KNN_fit_transformed.pkl", "wb"))

        # fit the grid search with training set
        grid.fit(X_train, Y_train)

        # retrieve the best estimator
        knn_best=grid.best_estimator_
        dump(knn_best, "best_estimators/KNN_BEST.joblib")
        
        print(accuracy_score(Y_test, knn_best.predict(X_test)))

        
