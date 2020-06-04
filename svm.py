import numpy as np
from sklearn.ensemble import BaggingClassifier
from sklearn.svm import SVC


class SVMEnsemble(BaggingClassifier):
    def __init__(self, kernel="linear", voting_method=None, svm_args=None, *args, **kwargs):
        if kernel not in {"linear", "poly", "rbf", "sigmoid", "precomputed"}:
            raise ValueError(f"kernel {kernel} is not recognized.")

        if voting_method not in {None, "hard", "soft"}:
            raise ValueError(f"voting_method {voting_method} is not recognized.")

        probability = True if voting_method == "soft" else False
        svm_args = dict() if not svm_args else svm_args
        base_estimator = SVC(kernel=kernel, probability=probability, **svm_args)

        super().__init__(base_estimator=base_estimator, *args, **kwargs)
        self.voting_method = voting_method

    def predict(self, X):
        if self.voting_method in {None, "hard"}:
            return super().predict(X)
        elif self.voting_method == "soft":
            probabilities = np.zeros((X.shape[0], self.classes_.shape[0]))
            for estimator in self.estimators_:
                estimator_probabilities = estimator.predict_proba(X)
                probabilities += estimator_probabilities
            return self.classes_[probabilities.argmax(axis=1)]
        else:
            raise ValueError(f"voting_method {self.voting_method} is not recognized.")
