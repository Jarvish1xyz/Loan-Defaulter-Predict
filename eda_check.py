import pandas as pd
import numpy as np

df = pd.read_csv("Loan_default.csv")
print("Shape:", df.shape)
print("\nColumns and Dtypes:")
print(df.dtypes)
print("\nMissing values:")
print(df.isnull().sum())
print("\nFirst 5 rows:")
print(df.head())
print("\nSummary Stats:")
print(df.describe().T)
print("\nValue counts for potential target and categorical columns:")
for col in df.select_dtypes(include=['object']).columns:
    print(f"\n--- {col} ---")
    print(df[col].value_counts(dropna=False))

# Also check target column if numeric (e.g. Default or LoanStatus or Loan_Status)
for col in df.columns:
    if 'def' in col.lower() or 'status' in col.lower() or 'target' in col.lower():
        print(f"\nTarget candidate {col} value counts:")
        print(df[col].value_counts(normalize=True))
