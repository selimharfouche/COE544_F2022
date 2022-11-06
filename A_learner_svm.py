############################################################
# SVM Trainer
# Trains an SVM model (SVC, NuSVC or LinearSVC) 
# Does preprocessing SplineTransformer or StandardScaler
# Transform Data (A FAIRE PLUS DE RECHERCHE)
# Learn Digits on train subsets (A FAIRE PLUS DE RECHERCHE)
# Predict the value of the digit on the test subset (A FAIRE PLUS DE RECHERCHE)
# Perform GridSearchCV with predefined parameters
# save the best model to best_estimators

############################################################

from sklearn import svm, metrics
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler, SplineTransformer
from sklearn.model_selection import GridSearchCV
from joblib import load
import joblib as jl
import pickle
import matplotlib.pyplot as plt
from joblib import load, dump

X_train = load(open('data/X_train.pkl', 'rb'))
X_test = load(open('data/X_test.pkl', 'rb'))
Y_test = load(open('data/Y_test.pkl', 'rb'))
Y_train = load(open('data/Y_train.pkl', 'rb'))



class SVM_class:

    def __init__(self,learner="SVC",scaler="ST",confusion_matrix=True,classification_report=True,minimal_grid_search=True, k_fold=2):
        if learner=="SVC":
            # https://scikit-learn.org/stable/modules/svm.html#scores-probabilities
            self.clf = svm.SVC(probability=True)
            
        elif learner=="NuSVC":
            self.clf = svm.NuSVC(probability=True)
        else:
            self.clf = svm.LinearSVC(probability=True)


        # preprocessings
        # https://scikit-learn.org/stable/modules/preprocessing.html#preprocessing
        if scaler=="ST":
            self.scaler = SplineTransformer()
        else:
            self.scaler = StandardScaler()

        self.confusion_matrix=confusion_matrix
        self.classification_report=classification_report
        self.minimal_grid_search=minimal_grid_search
        self.cv=k_fold

    def train(self):
        clf=self.clf
        scaler=self.scaler
        confusion_matrix=self.confusion_matrix
        classification_report=self.classification_report
        minimal_grid_search=self.minimal_grid_search

        X_train = load(open('data/X_train.pkl', 'rb'))
        X_test = load(open('data/X_test.pkl', 'rb'))
        Y_test = load(open('data/Y_test.pkl', 'rb'))
        Y_train = load(open('data/Y_train.pkl', 'rb'))


        # Transforming Data
        # Compute knot positions of splines.
        # https://datascience.stackexchange.com/questions/12321/whats-the-difference-between-fit-and-fit-transform-in-scikit-learn-models
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
        dump(scaler, open("data/SVM_fit_transformed.pkl", "wb"))
        dump(X_test, open("data/X_test.pkl", "wb"))

        # training
        # Learn the digits on the train subset
        # https://datascience.stackexchange.com/questions/87361/when-should-i-use-fitx-train-and-when-should-i-fit-x-train-y-train
        clf.fit(X_train, Y_train)

        # Predict the value of the digit on the test subset
        # https://stackoverflow.com/questions/62646058/how-does-the-predict-method-work-on-scikit-learn
        predicted = clf.predict(X_test)

   
    ############################# PLOTTING ###############################
   
        if classification_report:
            print(
                f"Classification report for classifier {clf}:\n"
                f"{metrics.classification_report(Y_test, predicted,zero_division=1)}\n"
            )

        if confusion_matrix:
            disp = metrics.ConfusionMatrixDisplay.from_predictions(Y_test, predicted)
            disp.figure_.suptitle("Confusion Matrix")
            print(f"Confusion matrix:\n{disp.confusion_matrix}")
            plt.show()

        # GridSearch CV
        # https://www.vebuso.com/2020/03/svm-hyperparameter-tuning-using-gridsearchcv/   
        if minimal_grid_search:
            Cs = [0.1]
            Gammas = [1]
            Kernels = ['poly']
        else:
            # defining parameter range
            Cs = [0.1, 1, 10, 100, 1000]
            Gammas = [1, 0.1, 0.01, 0.001, 0.0001]
            Kernels = ['rbf', 'poly', 'sigmoid', 'linear']

        # parameter grid using the range defined just above
        param_grid = {'C': Cs, 
                    'gamma': Gammas,
                    'kernel': Kernels} 


        # https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html#sklearn.model_selection.GridSearchCV
        grid = GridSearchCV(clf, param_grid, refit = True, cv = self.cv,return_train_score=True, verbose = 10)
        
        # fitting the model for grid search
        grid.fit(X_train, Y_train)
    ######################################################################

        # save the best parameters of the gridsearch
        dump(grid.best_estimator_, "best_estimators/SVM_BEST.joblib")
        print ("parameters saved in best_estimators/SVM_BEST.joblib ")
        print("Grid best estimator:")
        print(grid.best_estimator_)
        print("Grid best parameters:")
        print (grid.best_params_)
        print("Grid best score:")
        print (grid.best_score_)

        # prediction = grid.best_estimator_.predict(X_test)
        # print (classification_report(Y_test, prediction))
    ###############################################################################

