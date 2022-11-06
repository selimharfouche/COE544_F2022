from data_prep import data_prep

from joblib import dump, load


p1 = data_prep(numerical = True, feature1="pixel_intensity")
p1.prep_data()



from A_learner_svm import SVM_class

svm = SVM_class(confusion_matrix=False)

svm.train()


X_test = load(open('data/X_test.pkl', 'rb'))
estimator = load("best_estimators/SVM_BEST.joblib")

print("ESTIMATOR PREDICTION")
print(estimator.predict(X_test))
