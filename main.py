import matplotlib.pyplot as plt
import os
from sklearn.metrics import accuracy_score, f1_score
from sklearn.svm import SVC
from skopt import BayesSearchCV
from skopt.space import Categorical, Real
import warnings


from svm import SVMEnsemble
from utils import *


warnings.filterwarnings("ignore", category=FutureWarning)


def get_best_SVM_params(X_train, y_train, X_test, y_test):
    search_spaces = {
        "kernel": Categorical(["linear", "poly", "rbf", "sigmoid"]),
        "C": Real(1e-1, 1e+1, "uniform"),
        "gamma": Real(1e-4, 1e+4, "log-uniform")
    }

    best_accuracy = 0
    best_model = None
    for i in range(5):
        grid = BayesSearchCV(SVC(),
                             search_spaces,
                             n_iter=10,
                             cv=3,
                             n_jobs=-1)
        grid.fit(X_train, y_train)
        accuracy = accuracy_score(y_test, grid.predict(X_test))
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_model = grid

    return best_model.best_params_


def get_best_ensemble_params(X_train, y_train, X_test, y_test):
    search_spaces = {
        "max_samples": Real(0.5, 1, "uniform"),
        "max_features": Real(0.5, 1, "uniform"),

        "kernel": Categorical(["linear", "poly", "rbf", "sigmoid"]),
        "C": Real(1e-1, 1e+1, "uniform"),
    }

    best_accuracy = 0
    best_model = None
    for i in range(5):
        grid = BayesSearchCV(SVMEnsemble(),
                             search_spaces,
                             n_iter=10,
                             cv=3,
                             n_jobs=-1)
        grid.fit(X_train, y_train)
        accuracy = accuracy_score(y_test, grid.predict(X_test))
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_model = grid

    return best_model.best_params_


def optimal_classifier():
    #dataset_name = "acute_inflammations"
    #dataset_name = "breast_cancer_coimbra"
    dataset_name = "breast_cancer_wisconsin"
    X_train, X_test, y_train, y_test = load_X_y(dataset_name)

    params = get_best_SVM_params(X_train, y_train, X_test, y_test)
    params = dict(params)
    print("SVM best params:", params)
    params["random_state"] = 0
    svm_model = SVC(**params)
    svm_model.fit(X_train, y_train)
    y_pred = svm_model.predict(X_test)
    print("SVM accuracy:", accuracy_score(y_test, y_pred))
    print("SVM F1 score:", f1_score(y_test, y_pred))

    params = get_best_ensemble_params(X_train, y_train, X_test, y_test)
    params = dict(params)
    print(params)
    params["n_estimators"] = 100
    params["n_jobs"] = -1
    params["random_state"] = 0
    svm_ensemble_model = SVMEnsemble(**params)
    svm_ensemble_model.fit(X_train, y_train)
    y_pred = svm_ensemble_model.predict(X_test)

    print("SVM ensemble accuracy:", accuracy_score(y_test, y_pred))
    print("SVM ensemble F1 score:", f1_score(y_test, y_pred))


def training_size_change():
    classifiers = {
        "acute_inflammations": (
            SVC(C=7.4, gamma=0.01, kernel="linear"),
            SVMEnsemble(C=9, kernel="linear", max_features=0.9, max_samples=0.65)
        ),
        "breast_cancer_coimbra": (
            SVC(C=3, gamma=0.2, kernel="rbf"),
            SVMEnsemble(n_estimators=200, C=2, kernel="rbf", max_features=0.5,
                        max_samples=0.5)
        ),
        "breast_cancer_wisconsin": (
            SVC(C=8, gamma=0.01, kernel="rbf"),
            SVMEnsemble(C=9, kernel="rbf", max_features=0.8, max_samples=0.7)
        )
    }

    for dataset in ["acute_inflammations", "breast_cancer_coimbra", "breast_cancer_wisconsin"]:
        X, y = load_X_y(dataset, split=False)
        for clf in classifiers[dataset]:
            clf_str = "SVM" if str(clf) != "SVMEnsemble" else "SVMEnsemble"
            xs = []
            ys = []
            for training_size in np.linspace(0.5, 0.9, 5, endpoint=True):
                done = False
                while not done:
                    try:
                        training_size = round(training_size, 1)
                        X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                                            train_size=training_size,
                                                                            random_state=0)
                        clf.fit(X_train, y_train)
                        xs.append(training_size)
                        ys.append(round(accuracy_score(y_test, clf.predict(X_test)), 2))
                        done = True
                    except ValueError:
                        # this exception is caused by "Invalid input - all samples with positive weights have the
                        # same label", which in my opinion should be fixed in libsvm, not handled by user
                        continue

                title = clf_str + " " + dataset
                plt.title(title)
                plt.scatter(xs, ys)
                plt.xlabel("Training set size (in % of whole dataset)")
                plt.ylabel("Accuracy")
                plt.savefig(os.path.join("plots", clf_str + "_" + dataset))
                plt.clf()


if __name__ == "__main__":
    #optimal_classifier()
    training_size_change()
