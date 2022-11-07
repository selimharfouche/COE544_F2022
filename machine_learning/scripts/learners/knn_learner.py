from sklearn.neighbors import KNeighborsClassifier
from pickle import load
from sklearn.metrics import accuracy_score

from sklearn.model_selection import GridSearchCV
from joblib import load, dump




class KNN_class():
    def __init__(self,scaler="ST",verbose=0):
        self.scaler=scaler
        self.verbose = verbose

    def train(self):
        scaler=self.scaler

        # relative path
        try:
            X_train = load(open('../processed_data/X_train.pkl', 'rb'))
            X_test = load(open('../processed_data/X_test.pkl', 'rb'))
            Y_test = load(open('../processed_data/Y_test.pkl', 'rb'))
            Y_train = load(open('../processed_data/Y_train.pkl', 'rb'))
        except:
            X_train = load(open('../machine_learning/processed_data/X_train.pkl', 'rb'))
            X_test = load(open('../machine_learning/processed_data/X_test.pkl', 'rb'))
            Y_test = load(open('../machine_learning/processed_data/Y_test.pkl', 'rb'))
            Y_train = load(open('../machine_learning/processed_data/Y_train.pkl', 'rb'))

        #Grid search 
        parameters = {'n_neighbors':list(range(1, 20)) ,'weights': ['uniform', 'distance']}
        knn = KNeighborsClassifier()
        grid_search_knn = GridSearchCV(knn, parameters, cv=2, scoring='accuracy', return_train_score=True, verbose=self.verbose)

        # chosing scaler
        if scaler=="ST":
            from sklearn.preprocessing import SplineTransformer
            scaler = SplineTransformer()
        else:
            from sklearn.preprocessing import StandardScaler
            scaler = StandardScaler()

        # fit the scaler to data, then transform X_train and X_test
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        #Save fitted scaler model to be used with new images or else model will not predict well
        # relative path
        try:
            dump(scaler, open("../processed_data/KNN_fit_transformed.pkl", "wb"))
            dump(X_test, open("../processed_data/X_test_transformed_knn.pkl", "wb"))
        except:
            dump(scaler, open("../machine_learning/processed_data/KNN_fit_transformed.pkl", "wb"))
            dump(X_test, open("../machine_learning/processed_data/X_test_transformed_knn.pkl", "wb"))


        grid_search_knn.fit(X_train, Y_train)

        # retrieve the best estimator
        knn_best=grid_search_knn.best_estimator_

        # relative path
        try:
            dump(knn_best, "../best_estimators/KNN_BEST.joblib")
        except:
            dump(knn_best, "../machine_learning/best_estimators/KNN_BEST.joblib")

        print("Accuracy Score KNN")
        print(accuracy_score(Y_test, knn_best.predict(X_test)))

        
