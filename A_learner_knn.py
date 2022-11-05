#from A_data_prep import X_train, X_test, Y_train, Y_test
from sklearn.neighbors import KNeighborsClassifier
from pickle import load
from sklearn.preprocessing import StandardScaler, SplineTransformer



X_train = load(open('data/X_train.pkl', 'rb'))
X_test = load(open('data/X_test.pkl', 'rb'))
Y_test = load(open('data/Y_test.pkl', 'rb'))
Y_train = load(open('data/Y_train.pkl', 'rb'))

# from sklearn.preprocessing import StandardScaler
scaler = SplineTransformer()
X_train_scaled = scaler.fit_transform(X_train)
#we must apply the scaling to the test set that we computed for the training set
X_test_scaled = scaler.transform(X_test)

X_train_scaled = X_train
X_test_scaled = X_test

# Import Matplotlib for visualization
import matplotlib.pyplot as plt

acc = []
# Will take some time
from sklearn import metrics

# for i in range(1, 20):
#     neigh = KNeighborsClassifier(n_neighbors=i).fit(X_train_scaled, Y_train)
#     yhat = neigh.predict(X_test_scaled)
#     acc.append(metrics.accuracy_score(Y_test, yhat))
#
# plt.figure(figsize=(10, 6))
# plt.plot(range(1, 20), acc, color='blue', linestyle='dashed',
#          marker='o', markerfacecolor='red', markersize=10)
# plt.title('accuracy vs. K Value')
# plt.xlabel('K')
# plt.ylabel('Accuracy')
# print("Maximum accuracy:-", max(acc), "at K =", acc.index(max(acc)))
# knn = KNeighborsClassifier(n_neighbors = acc.index(max(acc)))
# knn.fit(X_train_scaled, Y_train)

#Checking performance on the training set

# Confusion matrix
    # We can also plot a :ref:`confusion matrix <confusion_matrix>` of the
    # true digit values and the predicted digit values.


knn = KNeighborsClassifier()
disp = metrics.ConfusionMatrixDisplay.from_predictions(Y_test, knn.fit(X_train,Y_train).predict(X_test))
disp.figure_.suptitle("Confusion Matrix")
print(f"Confusion matrix:\n{disp.confusion_matrix}")

plt.show()

#Grid search best k-NN parameters
parameters = {'n_neighbors':list(range(1, 20)) ,'weights': ['uniform', 'distance']}
from sklearn.model_selection import GridSearchCV
# create an instance of the knn classifier
knn_grid_tuned = KNeighborsClassifier()
# create an instance of grid search with the above parameters
grid_search = GridSearchCV(knn_grid_tuned, parameters, cv=2, scoring='accuracy', return_train_score=True, verbose=10)
# fit the grid search with training set
grid_search.fit(X_train_scaled, Y_train)
# retrieve the best estimator
knn_best = grid_search.best_estimator_
print(knn_best)

from sklearn.metrics import accuracy_score
print(accuracy_score(Y_test, knn_best.predict(X_test_scaled)))
print (grid_search.best_params_)