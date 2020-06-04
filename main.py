import os
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


from svm import SVMEnsemble


if __name__ == "__main__":
    file_name = "cardiovascular.csv"
    file_path = os.path.join("data_cleaned", file_name)
    dataset = pd.read_csv(file_path)

    #dataset = dataset.sample(frac=1, random_state=0).reset_index(drop=True)
    #dataset = dataset.iloc[:70000, :]
    y = dataset.iloc[:, -2]
    X = dataset.iloc[:, :-2]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.fit_transform(X_test)

    model = SVMEnsemble(voting_method="hard", max_samples=0.5, n_jobs=-1, kernel="rbf")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    print("Accuracy (hard voting):", accuracy_score(y_test, y_pred))

    model = SVMEnsemble(voting_method="soft", max_samples=0.5, n_jobs=-1, kernel="rbf")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    print("Accuracy (soft voting):", accuracy_score(y_test, y_pred))
