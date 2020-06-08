import os
import pandas as pd


# load dataset
data_dir = os.path.join(os.path.dirname(os.getcwd()), "data_raw")
file_name = "breast_cancer_coimbra.csv"
file_path = os.path.join(data_dir, file_name)
dataset = pd.read_csv(file_path)

labels_to_boolean = lambda classif: True if classif == 2 else False
dataset.loc[:, "Classification"] = dataset.loc[:, "Classification"].map(labels_to_boolean)

# save results
cleaned_data_dir = os.path.join(os.path.dirname(os.getcwd()), "data_cleaned")
result_file_path = os.path.join(cleaned_data_dir, file_name)
dataset.to_csv(path_or_buf=result_file_path, index=False)