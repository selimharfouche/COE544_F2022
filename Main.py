from data_prep import data_prep

from joblib import dump, load

from sklearn.preprocessing import StandardScaler, SplineTransformer

p1 = data_prep(numerical = True, feature1="pixel_intensity")
p1.prep_data()

p1 = data_prep(numerical = True, feature1="pixel_intensity", uploaded=True)
p1.prep_data()




from A_learner_svm import SVM_class

svm = SVM_class(confusion_matrix=False)

svm.train()

print("ALL DATA")
X_test = load(open('data/X_test.pkl', 'rb'))
Y_test = load(open('data/Y_test.pkl', 'rb'))
estimator = load("best_estimators/SVM_BEST.joblib")
print(estimator.predict(X_test))



print("SIGNLE IMAGLE")
X_test = load(open('data/X_test_uploaded.pkl', 'rb'))
print(estimator.predict(X_test))



# print("ESTIMATOR PREDICTION")
# print(estimator.predict(X_test))
# for i in range(len(X_test)):
# 	print("X=%s, Predicted=%s" % (X_test[i], Y_test[i]))
