import matplotlib.pyplot as plt
from data_preparation import X_train, X_test, Y_train, Y_test
from sklearn import datasets, svm, metrics
# Create a classifier: a support vector classifier
clf = svm.SVC(gamma=0.001)
# Learn the digits on the train subset
clf.fit(X_train, Y_train)
# Predict the value of the digit on the test subset
predicted = clf.predict(X_test)
print(
    f"Classification report for classifier {clf}:\n"
    f"{metrics.classification_report(Y_test, predicted)}\n"
)