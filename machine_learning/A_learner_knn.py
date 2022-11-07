from sklearn.neighbors import KNeighborsClassifier
from pickle import load
from sklearn.metrics import accuracy_score

from sklearn.model_selection import GridSearchCV
from joblib import load, dump


class KNN_class():
    def __init__(self,scaler="ST"):
        self.scaler=scaler

    def train(self):
        scaler=self.scaler
        X_train = load(open('../machine_learning/data/X_train.pkl', 'rb'))
        X_test = load(open('../machine_learning/data/X_test.pkl', 'rb'))
        Y_test = load(open('../machine_learning/data/Y_test.pkl', 'rb'))
        Y_train = load(open('../machine_learning/data/Y_train.pkl', 'rb'))

        #Grid search best k-NN parameters
        parameters = {'n_neighbors':list(range(1, 20)) ,'weights': ['uniform', 'distance']}

        # create an instance of the knn classifier
        knn_grid_tuned = KNeighborsClassifier()

        # create an instance of grid search with the above parameters
        grid = GridSearchCV(knn_grid_tuned, parameters, cv=2, scoring='accuracy', return_train_score=True, verbose=10)


        if scaler=="ST":
            from sklearn.preprocessing import SplineTransformer
            scaler = SplineTransformer()
        else:
            from sklearn.preprocessing import StandardScaler
            scaler = StandardScaler()

        X_train = scaler.fit_transform(X_train)
        #we must apply the scaling to the test set that we computed for the training set
        X_test = scaler.transform(X_test)
        #Save fitted scaler model to be used with new images or else model will not predict well
        dump(scaler, open("../machine_learning/data/KNN_fit_transformed.pkl", "wb"))

        # fit the grid search with training set
        grid.fit(X_train, Y_train)

        # retrieve the best estimator
        knn_best=grid.best_estimator_
        dump(knn_best, "../machine_learning/best_estimators/KNN_BEST.joblib")

        print("Accuracy Score")
        print(accuracy_score(Y_test, knn_best.predict(X_test)))

        
