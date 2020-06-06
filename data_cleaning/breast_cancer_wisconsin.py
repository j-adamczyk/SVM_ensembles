import os
import pandas as pd


# load dataset
data_dir = os.path.join(os.path.dirname(os.getcwd()), "data_raw")
file_name = "breast_cancer_wisconsin.csv"
file_path = os.path.join(data_dir, file_name)
dataset = pd.read_csv(file_path)

# remove the id column, since it's useless for ML
dataset.drop(columns="id", inplace=True)

# move "diagnosis" column to the last position (easier to pop class later for ML)
dataset.insert(len(dataset.columns) - 1, "diagnosis", dataset.pop("diagnosis"))

# classes are "B" (benign) / "M" (malignant) strings, change them to false (B) / true (M) booleans
string_to_boolean = lambda string: True if string == "M" else False
dataset["diagnosis"] = dataset["diagnosis"].map(string_to_boolean)

# save results
cleaned_data_dir = os.path.join(os.path.dirname(os.getcwd()), "data_cleaned")
result_file_path = os.path.join(cleaned_data_dir, file_name)
dataset.to_csv(path_or_buf=result_file_path, index=False)