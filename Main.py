from data_prep import data_prep
from A_learner_svm import SVM_class
p1 = data_prep(numerical=True, feature1='pixel_intensity')

p1.prep_data()

svm = SVM_class(confusion_matrix=False)

svm.train()

