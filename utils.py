import numpy as np
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def load_acute_inflammations(label="inflammation"):
    if label not in {"inflammation", "nephritis"}:
        raise ValueError(f"label has unrecognized value {label}.")

    file_name = "acute_inflammations.csv"
    file_path = os.path.join("data_cleaned", file_name)
    dataset = pd.read_csv(file_path)

    y = dataset.iloc[:, -2] if label == "inflammation" else dataset.iloc[:, -1]
    X = dataset.iloc[:, :-2]
    return X, y


def load_breast_cancer_coimbra():
    file_name = "breast_cancer_coimbra.csv"
    file_path = os.path.join("data_cleaned", file_name)
    dataset = pd.read_csv(file_path)

    y = dataset.iloc[:, -1]
    X = dataset.iloc[:, :-1]
    return X, y


def load_breast_cancer_wisconsin():
    file_name = "breast_cancer_wisconsin.csv"
    file_path = os.path.join("data_cleaned", file_name)
    dataset = pd.read_csv(file_path)

    y = dataset.iloc[:, -1]
    X = dataset.iloc[:, :-1]
    return X, y


def load_X_y(dataset_name, split=True):
    loading_functions = {
        "acute_inflammations": load_acute_inflammations,
        "breast_cancer_coimbra": load_breast_cancer_coimbra,
        "breast_cancer_wisconsin": load_breast_cancer_wisconsin
    }

    X, y = loading_functions[dataset_name]()
    if not split:
        return X, y

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.fit_transform(X_test)

    return X_train, X_test, y_train, y_test


def add_X_noise(X, mu=0, sigma=0.2, bool_change_prob=0.1):
    for col in X.select_dtypes(include=np.float).columns:
        X[col] = X[col] + np.random.normal(mu, sigma)

    for col in X.select_dtypes(include=np.bool).columns:
        X[col] ^= np.random.rand(X[col].size) < bool_change_prob

    return X


def add_y_noise(y, prob=0.1):
    return y ^ np.random.rand(y.size) < prob

