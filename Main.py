#from A_learner_svm import SVM_class
from joblib import dump, load
from A_data_prep import X_train, X_test, Y_train, Y_test
from sklearn.preprocessing import SplineTransformer
import joblib
import matplotlib.pyplot as plt
from sklearn import datasets, svm, metrics

loaded_model = joblib.load("models/svm-1.sav")

predicted = loaded_model.predict(X_test)
#result = loaded_model.score(X_test, Y_test)
#print(result)

disp = metrics.ConfusionMatrixDisplay.from_predictions(Y_test, predicted)
disp.figure_.suptitle("Confusion Matrix")
print(f"Confusion matrix:\n{disp.confusion_matrix}")

plt.show()
#sc = SplineTransformer()

#X_train = sc.fit_transform(X_train)
#X_test = sc.transform(X_test)


######################
# estimator.fit(X_train, Y_train)

# model = estimator
# joblib.dump(model, "models/svm-1.sav")

#########################################


# from sklearn.metrics import accuracy_score
# print(accuracy_score(Y_test, estimator.predict(X_test)))

##################################



#SVM = SVM_class()

