import numpy as np
import pandas as pd

# Read dataset file ckd.csv
dataset = pd.read_csv(r"C:\Users\I346327\Desktop\Assignment_BLR\kidneyChronic.csv",header=0, na_values="?")

# Replace null values "?" by numpy.NaN
dataset.replace("?", np.NaN)

# Convert nominal values to binary values
nom_bin = {"rbc": {"normal": 1, "abnormal": 0},
           "pc": {"normal": 1, "abnormal": 0},
           "pcc": {"present": 1, "notpresent": 0},
           "ba": {"present": 1, "notpresent": 0},
           "htn": {"yes": 1, "no": 0},
           "dm": {"yes": 1, "no": 0},
           "cad": {"yes": 1, "no": 0},
           "appet": {"good": 1, "poor": 0},
           "pe": {"yes": 1, "no": 0},
           "ane": {"yes": 1, "no": 0}}

# Replace binary values into dataset
dataset.replace(nom_bin, inplace=True)
dataset.fillna(round(dataset.mean(),2), inplace=True)
#print(dataset.dtypes)
# Fill null values with mean value of the respective column
obj_cols=['rbcc','pcv','wbcc','dm','cad']
for i in obj_cols:
    dataset[i] = pd.to_numeric(dataset[i], errors='coerce')
    dataset[i].fillna(round(dataset[i].mean(),2), inplace=True)
dataset.head(10) #printing the first 10 records of the dataset

# Save this dataset as preprocessed.csv for further prediction
dataset.to_csv(r"C:\Users\I346327\Desktop\Assignment_BLR\preprocessed.csv", sep=',', index=False)
