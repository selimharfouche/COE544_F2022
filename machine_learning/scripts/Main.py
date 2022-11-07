##################### Path configuration start #####################
import sys
from pathlib import Path # if you haven't already done so
file = Path(__file__).resolve()
parent, root = file.parent, file.parents[1]
sys.path.append(str(root))

# Additionally remove the current file's directory from sys.path
try:
    sys.path.remove(str(parent))
except ValueError: # Already removed
    pass
##################### Path configuration end #####################


from scripts.data_processing.data_prep import data_prep

from joblib import dump, load


import numpy as np


from scripts.user_processing import user_processing

user_processing.prepare_data(features=["pixel_intensity","histogram"])

up= user_processing(learner="SVM")
up.give_label()


up= user_processing(learner="KNN")
up.give_label()


up= user_processing(learner="ensemble")
up.give_label()




# p1 = data_prep(numerical = True, features_array=["pixel_intensity","histogram"])
# p1.prep_data()

# p1 = data_prep(numerical = True, features_array=["pixel_intensity","histogram"], uploaded=True)
# p1.prep_data()




# from A_learner_svm import SVM_class

# svm = SVM_class(confusion_matrix=False)

# svm.train()

# # print("ALL DATA")
# X_test = load(open('data/X_test.pkl', 'rb'))
# Y_test = load(open('data/Y_test.pkl', 'rb'))


# print ("main length of X_test")
# print(len(X_test[0]))

# estimator = load("best_estimators/SVM_BEST.joblib")
# print(estimator.predict(X_test))



# print("SINGLE IMAGLE")
# X_test = load(open('data/X_test_uploaded.pkl', 'rb'))
# sc = load(open("data/SVM_fit_transformed.pkl", "rb"))

# #transformation for svm
# X_test = sc.transform(X_test)


# print ("expected label")
# print(estimator.predict(X_test))
# print("probability")
# print(np.max(estimator.predict_proba(X_test)))




# #print(estimator.predict(X_test))





# # print("ESTIMATOR PREDICTION")
# # print(estimator.predict(X_test))
# # for i in range(len(X_test)):
# # 	print("X=%s, Predicted=%s" % (X_test[i], Y_test[i]))
