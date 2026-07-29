import os
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder

# Load the dataset into a DataFrame
df = pd.read_csv('temp\dataset_89aedea7-4812-42a7-a291-15ab560666d6.csv')

# Dynamically identify numeric and categorical columns using select_dtypes
numeric_cols = df.select_dtypes(include=['int64']).columns
categorical_cols = df.select_dtypes(include=['str']).columns

# Drop columns with more than 80% missing values
df = df.dropna(thresh=df.shape[0]*0.2, axis=1)

# Identify columns with >=90% identical values
duplicate_columns = []
for col in df.columns:
    if df[col].duplicated().sum() / len(df) >= 0.9:
        duplicate_columns.append(col)
df = df.drop(duplicate_columns, axis=1)

# Impute missing numeric columns with median using SimpleImputer
numeric_imputer = SimpleImputer(strategy='median')
df[numeric_cols] = numeric_imputer.fit_transform(df[numeric_cols])

# Impute missing categorical columns with mode using SimpleImputer
categorical_imputer = SimpleImposer(mode='most_frequent')
df[categorical_cols] = categorical_imputer.fit_transform(df[categorical_cols])

# Handle outliers in numeric columns
for col in numeric_cols:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    df.loc[df[col] < lower_bound, col] = np.nan
    df[col] = (df[col] >= lower_bound) & (df[col] <= upper_bound)
    numeric_imputer = SimpleImputer(strategy='median')
    df[numeric_cols] = numeric_imputer.fit_transform(df[numeric_cols])

# Ensure all column operations include checks for existence and avoid hardcoding column names
if 'show_id' in df.columns:
    show_id_le = LabelEncoder()
    for col in categorical_cols:
        if col != 'show_id':
            show_id_le.fit(df[col])
            df[col] = show_id_le.transform(df[col])

# Save the cleaned dataset to the provided output path
df.to_csv('temp\cleaned_89aedea7-4812-42a7-a291-15ab560666d6.csv', index=False)