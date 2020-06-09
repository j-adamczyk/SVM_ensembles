import numpy as np
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns


def load_acute_inflammations(label="inflammation", split=True):
    if label not in {"inflammation", "nephritis"}:
        raise ValueError(f"label has unrecognized value {label}.")

    file_name = "acute_inflammations.csv"
    file_path = os.path.join("data_cleaned", file_name)
    dataset = pd.read_csv(file_path)
    if not split:
        return dataset

    y = dataset.iloc[:, -2] if label == "inflammation" else dataset.iloc[:, -1]
    X = dataset.iloc[:, :-2]
    return X, y


def load_breast_cancer_coimbra(split=True):
    file_name = "breast_cancer_coimbra.csv"
    file_path = os.path.join("data_cleaned", file_name)
    dataset = pd.read_csv(file_path)
    if not split:
        return dataset

    y = dataset.iloc[:, -1]
    X = dataset.iloc[:, :-1]
    return X, y


def load_breast_cancer_wisconsin(split=True):
    file_name = "breast_cancer_wisconsin.csv"
    file_path = os.path.join("data_cleaned", file_name)
    dataset = pd.read_csv(file_path)
    if not split:
        return dataset

    y = dataset.iloc[:, -1]
    X = dataset.iloc[:, :-1]
    return X, y


def load_X(dataset_name):
    loading_functions = {"acute_inflammations": load_acute_inflammations,
        "breast_cancer_coimbra": load_breast_cancer_coimbra,
        "breast_cancer_wisconsin": load_breast_cancer_wisconsin}

    return loading_functions[dataset_name](split=False)


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
    return y ^ (np.random.rand(y.size) < prob)


def histogram(feature, score_feature, label_1, label_2, dataset, value_left,
              value_right):
    sns.set_style("white")

    # Import data
    df = load_X(dataset)

    x1 = df.loc[df[score_feature] == True, feature]
    x2 = df.loc[df[score_feature] == False, feature]

    # Plot
    kwargs = dict(hist_kws={'alpha': .6}, kde_kws={'linewidth': 2})

    plt.figure(figsize=(10, 7), dpi=80)
    sns.distplot(x1, color="dodgerblue", label=label_1, **kwargs)
    sns.distplot(x2, color="orange", label=label_2, **kwargs)
    plt.xlim(value_left, value_right)
    plt.legend()
    plt.title(dataset)
    plt.savefig(os.path.join("plots", "hist_" + dataset + "_" + feature))
    plt.clf()

