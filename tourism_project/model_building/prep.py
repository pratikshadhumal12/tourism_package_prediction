# for data manipulation
import pandas as pd
import numpy as np
import sklearn
# for creating a folder
import os
# for data preprocessing and pipeline creation
from sklearn.model_selection import train_test_split
# for converting text data in to numerical representation
from sklearn.preprocessing import LabelEncoder
# for hugging face space authentication to upload files
from huggingface_hub import login, HfApi

# Define constants for the dataset and output paths
api = HfApi(token=os.getenv("HF_TOKEN"))
DATASET_PATH = "hf://datasets/pratikshadhumal12/tourism-package-prediction-dataset/tourism.csv"
df = pd.read_csv(DATASET_PATH)
print("Dataset loaded successfully.")

# Print shape of the dataset
print(f"Dataset shape: {df.shape}")

# Drop the unique identifier and any 'Unnamed: 0' column
df.drop(columns=['CustomerID'], inplace=True)
if 'Unnamed: 0' in df.columns:
    df.drop(columns=['Unnamed: 0'], inplace=True)
print(df.head())

# Replace the gender column value from 'Fe Male' to 'Female'
df['Gender'] = df['Gender'].replace('Fe Male', 'Female')
print(df['Gender'].value_counts())

# Replace the Unmarried to Signle from MaritalStatus column to combine both the values
df['MaritalStatus'] = df['MaritalStatus'].replace('Unmarried', 'Single')
print(df['MaritalStatus'].value_counts())

# Check missing value
print(df.isnull().sum())

# For numerical columns, fill missing value with median
numerical_cols = df.select_dtypes(include='number').columns
print(numerical_cols)

df[numerical_cols] = df[numerical_cols].fillna(
    df[numerical_cols].median()
)

# For categorical columns, fill missing value with mode
categorical_cols = df.select_dtypes(include='object').columns
print(categorical_cols)

df[categorical_cols] = df[categorical_cols].fillna(
    df[categorical_cols].mode().iloc[0]
)

# Target variable
target_col = 'ProdTaken'

# Combine all feature columns that should be in X
all_cols = list(numerical_cols) + list(categorical_cols)

# Split into X (features) and y (target)
# Ensure X contains only the explicitly defined feature columns
X = df.drop(columns=[target_col])
y = df[target_col]

# Perform train-test split
Xtrain, Xtest, ytrain, ytest = train_test_split(
    X, y, test_size=0.2, random_state=42
)

Xtrain.to_csv("Xtrain.csv",index=False)
Xtest.to_csv("Xtest.csv",index=False)
ytrain.to_csv("ytrain.csv",index=False)
ytest.to_csv("ytest.csv",index=False)


files = ["Xtrain.csv","Xtest.csv","ytrain.csv","ytest.csv"]

for file_path in files:
    api.upload_file(
        path_or_fileobj=file_path,
        path_in_repo=file_path.split("/")[-1],  # just the filename
        repo_id="pratikshadhumal12/tourism-package-prediction-dataset",
        repo_type="dataset",
    )
