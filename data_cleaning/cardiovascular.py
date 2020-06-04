import os
import pandas as pd


# load dataset
data_dir = os.path.join(os.path.dirname(os.getcwd()), "data_raw")
file_name = "cardiovascular.csv"
file_path = os.path.join(data_dir, file_name)
dataset = pd.read_csv(file_path, sep=";")

# remove the id column, since it's useless for ML
dataset.drop(columns="id", inplace=True)

# age is in days, convert it to years (with true integer division)
dataset.loc[:, "age"] //= 365

# map gender from integer (1/2) to boolean to negate magnitude difference
gender_to_boolean = lambda gender: False if gender == 1 else True
dataset.loc[:, "gender"] = dataset.loc[:, "gender"].map(gender_to_boolean)

# "smoke", "alco", "active" features and "cardio" class should be boolean, not int
dataset[["smoke", "alco", "active", "cardio"]] = dataset[["smoke", "alco", "active", "cardio"]].astype("bool")

# save results
cleaned_data_dir = os.path.join(os.path.dirname(os.getcwd()), "data_cleaned")
result_file_path = os.path.join(cleaned_data_dir, file_name)
dataset.to_csv(path_or_buf=result_file_path, index=False)
