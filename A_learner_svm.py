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

######################################################################
############################## IMPORTS ###############################
######################################################################

from sklearn import svm, metrics
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler, SplineTransformer
from sklearn.svm import SVC
from A_data_prep import X_train, X_test, Y_train, Y_test
from sklearn.model_selection import GridSearchCV

from joblib import dump

######################################################################


class SVM_class:
######################################################################
######################## CHOOSING SVM LEARNER ########################
######################################################################

    # SVC
    clf = svm.SVC()
    # # NuSVC
    #clf = svm.NuSVC()
    # # Linear SVC()
    # clf = svm.LinearSVC()
   
###############################################################################

######################################################################
########################### PREPROCESSING ############################
######################################################################
    # https://scikit-learn.org/stable/modules/preprocessing.html#preprocessing
    sc = SplineTransformer()
    # sc = StandardScaler()
######################################################################


######################################################################
######################## DATA TRANSFORMATION #########################
######################################################################
    # Transforming Data
    # Compute knot positions of splines.
    # https://datascience.stackexchange.com/questions/12321/whats-the-difference-between-fit-and-fit-transform-in-scikit-learn-models
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)

######################################################################
############################# TRAINING ###############################
######################################################################
    # Learn the digits on the train subset
    # https://datascience.stackexchange.com/questions/87361/when-should-i-use-fitx-train-and-when-should-i-fit-x-train-y-train
    clf.fit(X_train, Y_train)

    # Predict the value of the digit on the test subset
    # https://stackoverflow.com/questions/62646058/how-does-the-predict-method-work-on-scikit-learn
    predicted = clf.predict(X_test)

    # https://scikit-learn.org/stable/modules/svm.html#scores-probabilities
    clf = svm.SVC(probability=True) 
######################################################################


######################################################################
############################# PLOTTING ###############################
######################################################################
    # Classification report
    # print(
    #     f"Classification report for classifier {clf}:\n"
    #     f"{metrics.classification_report(Y_test, predicted,zero_division=1)}\n"
    # )

    # Confusion matrix
    # We can also plot a :ref:`confusion matrix <confusion_matrix>` of the
    # true digit values and the predicted digit values.

    # disp = metrics.ConfusionMatrixDisplay.from_predictions(Y_test, predicted)
    # disp.figure_.suptitle("Confusion Matrix")
    #print(f"Confusion matrix:\n{disp.confusion_matrix}")

    #plt.show()
###############################################################################
   

######################################################################
########################## GRID SEARCH CV ############################
######################################################################


    # GridSearch CV
    # https://www.vebuso.com/2020/03/svm-hyperparameter-tuning-using-gridsearchcv/

    # defining parameter range
    Cs = [0.1, 1, 10, 100, 1000]
    Gammas = [1, 0.1, 0.01, 0.001, 0.0001]
    Kernels = ['rbf', 'poly', 'sigmoid']

    # parameter grid using the range defined just above
    param_grid = {'C': Cs, 
                'gamma': Gammas,
                'kernel': Kernels} 


    # https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html#sklearn.model_selection.GridSearchCV
    grid = GridSearchCV(SVC(), param_grid, refit = True, cv = 10,return_train_score=True, verbose = 10)
    
    # fitting the model for grid search
    grid.fit(X_train, Y_train)
######################################################################


######################################################################
########################## DEBUG-PRINTING ############################
######################################################################
    # # print best parameter after tuning
    # print("Grid best parameters:")
    # print(grid.best_params_)
    
    # # print how our model looks after hyper-parameter tuning
    # svm_best=grid.best_estimator_
    # print("Grid best estimator")
    # print(svm_best)
###############################################################################

######################################################################
######################### SAVING PARAMETERS ##########################
######################################################################
    # save the best parameters of the gridsearch
    dump(grid.best_estimator_, "best_estimators/svm_best.joblib")
    print ("parameters saved in best_estimators/svm_best1.joblib ")
###############################################################################


############################################################################################################################################
############################################################################################################################################
############################################################################################################################################
############################################################################################################################################
############################################################################################################################################
    # scores = [x[1] for x in clf.grid_scores_]
    # scores = np.array(scores).reshape(len(Cs), len(Gammas))

    # for ind, i in enumerate(Cs):
    #     plt.plot(Gammas, scores[ind], label='C: ' + str(i))
    #     plt.legend()
    #     plt.xlabel('Gamma')
    #     plt.ylabel('Mean score')
    #     plt.show()
    
    # # print classification report
    # print(classification_report(Y_test, grid_predictions))