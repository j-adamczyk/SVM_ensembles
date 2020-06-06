import inspect
import numpy as np
from sklearn.ensemble import BaggingClassifier
from sklearn.svm import SVC
from skopt import BayesSearchCV


svm_possible_args = {"C", "kernel", "degree", "gamma", "coef0", "shrinking", "probability", "tol", "cache_size",
                     "class_weight", "max_iter", "decision_function_shape", "break_ties"}

bagging_possible_args = {"n_estimators", "max_samples", "max_features", "bootstrap", "bootstrap_features",
                         "oob_score", "warm_start", "n_jobs"}

common_possible_args = {"random_state", "verbose"}


class SVMEnsemble(BaggingClassifier):
    def __init__(self, voting_method="hard", n_jobs=-1,
                 n_estimators=10, max_samples=0.7, max_features=0.7,
                 C=1.0, kernel="linear", gamma="scale",
                 **kwargs):
        if voting_method not in {"hard", "soft"}:
            raise ValueError(f"voting_method {voting_method} is not recognized.")

        self._voting_method = voting_method
        self._C = C
        self._gamma = gamma
        self._kernel = kernel

        passed_args = {
            "n_jobs": n_jobs,
            "n_estimators": n_estimators,
            "max_samples": max_samples,
            "max_features": max_features,
            "C": C,
            "gamma": gamma,
            "cache_size": 1024,
        }

        kwargs.update(passed_args)

        svm_args = {
            "probability": True if voting_method == "soft" else False,
            "kernel": kernel
        }

        bagging_args = dict()

        for arg_name, arg_val in kwargs.items():
            if arg_name in svm_possible_args:
                svm_args[arg_name] = arg_val
            elif arg_name in bagging_possible_args:
                bagging_args[arg_name] = arg_val
            elif arg_name in common_possible_args:
                svm_args[arg_name] = arg_val
                bagging_args[arg_name] = arg_val
            else:
                raise ValueError(f"argument {voting_method} is not recognized.")

        self.svm_args = svm_args
        self.bagging_args = bagging_args

        base_estimator = SVC(**svm_args)
        super().__init__(base_estimator=base_estimator, **bagging_args)

    @property
    def voting_method(self):
        return self._voting_method

    @voting_method.setter
    def voting_method(self, new_voting_method):
        if new_voting_method == "soft":
            self._voting_method = new_voting_method
            self.svm_args["probability"] = True
            base_estimator = SVC(**self.svm_args)
            super().__init__(base_estimator=base_estimator, **self.bagging_args)
        elif self._voting_method == "soft":
            self._voting_method = new_voting_method
            self.svm_args["probability"] = False
            base_estimator = SVC(**self.svm_args)
            super().__init__(base_estimator=base_estimator, **self.bagging_args)
        else:
            self._voting_method = new_voting_method

    @property
    def C(self):
        return self._C

    @C.setter
    def C(self, new_C):
        self._C = new_C
        self.svm_args["C"] = new_C
        base_estimator = SVC(**self.svm_args)
        super().__init__(base_estimator=base_estimator, **self.bagging_args)

    @property
    def gamma(self):
        return self._gamma

    @gamma.setter
    def gamma(self, new_gamma):
        self._gamma = new_gamma
        self.svm_args["gamma"] = new_gamma
        base_estimator = SVC(**self.svm_args)
        super().__init__(base_estimator=base_estimator, **self.bagging_args)

    @property
    def kernel(self):
        return self._kernel

    @kernel.setter
    def kernel(self, new_kernel):
        self._kernel = new_kernel
        self.svm_args["kernel"] = new_kernel
        base_estimator = SVC(**self.svm_args)
        super().__init__(base_estimator=base_estimator, **self.bagging_args)

    def predict(self, X):
        if self._voting_method == "hard":
            return super().predict(X)
        elif self._voting_method == "soft":
            probabilities = np.zeros((X.shape[0], self.classes_.shape[0]))
            for estimator in self.estimators_:
                estimator_probabilities = estimator.predict_proba(X)
                probabilities += estimator_probabilities
            return self.classes_[probabilities.argmax(axis=1)]
        else:
            raise ValueError(f"voting_method {self._voting_method} is not recognized.")

    def __str__(self):
        return "SVMEnsemble"
