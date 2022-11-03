############################################################
# Imports


from sklearn import svm, metrics
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler, SplineTransformer
from sklearn.svm import SVC
from A_data_prep import X_train, X_test, Y_train, Y_test


class SVM_class:
    
    #choosing the learner: svm SVC
    clf = svm.SVC()

    ###############################################################################
    #NuSVC
    #clf = svm.NuSVC()
    #Linear SVC()
    #clf = svm.LinearSVC()
    ###############################################################################

    #choosing preprocessing
    sc = SplineTransformer()
    ###############################################################################
    #sc = StandardScaler()
    ###############################################################################




    # Compute knot positions of splines.
    # transform data
    # https://datascience.stackexchange.com/questions/12321/whats-the-difference-between-fit-and-fit-transform-in-scikit-learn-models
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)


    # Learn the digits on the train subset
    # https://datascience.stackexchange.com/questions/87361/when-should-i-use-fitx-train-and-when-should-i-fit-x-train-y-train
    clf.fit(X_train, Y_train)

    # Predict the value of the digit on the test subset
    # https://stackoverflow.com/questions/62646058/how-does-the-predict-method-work-on-scikit-learn
    predicted = clf.predict(X_test)

    # https://scikit-learn.org/stable/modules/svm.html#scores-probabilities
    clf = svm.SVC(probability=True) 



    ###############################################################################
    # Classification report
    # print(
    #     f"Classification report for classifier {clf}:\n"
    #     f"{metrics.classification_report(Y_test, predicted,zero_division=1)}\n"
    # )

    ###############################################################################
    # We can also plot a :ref:`confusion matrix <confusion_matrix>` of the
    # true digit values and the predicted digit values.

    # disp = metrics.ConfusionMatrixDisplay.from_predictions(Y_test, predicted)
    # disp.figure_.suptitle("Confusion Matrix")
    #print(f"Confusion matrix:\n{disp.confusion_matrix}")

    #plt.show()
    ###############################################################################


    from sklearn.model_selection import GridSearchCV
    
    # defining parameter range
    param_grid = {'C': [0.1, 1], 
                'gamma': [1, 0.1],
                'kernel': ['rbf']} 

    # param_grid = {'C': [0.1, 1, 10, 100, 1000], 
    #               'gamma': [1, 0.1, 0.01, 0.001, 0.0001],
    #               'kernel': ['rbf']} 
    # https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html#sklearn.model_selection.GridSearchCV
    grid = GridSearchCV(SVC(), param_grid, refit = True, cv = 10,return_train_score=True, verbose = 10)
    
    # fitting the model for grid search
    grid.fit(X_train, Y_train)
    # print best parameter after tuning
    print(grid.best_params_)
    
    # print how our model looks after hyper-parameter tuning
    svm_best=grid.best_estimator_
    print(grid.best_estimator_)
    grid_predictions = grid.predict(X_test)
    
    # print classification report
    print(classification_report(Y_test, grid_predictions))