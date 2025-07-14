import pandas as pd
import numpy as np


def main(df):
    # Columns with missing values (you can specify which ones or auto-detect)
    columns_with_nans = df.columns[df.isna().any()].tolist()

    # Loop over each cancer type (classe_name)
    for class_name, group in df.groupby("classe_name"):
        for col in columns_with_nans:
            # Skip non-numeric columns
            if not pd.api.types.is_numeric_dtype(df[col]):
                continue

            # Compute mean and std for the group (excluding NaNs)
            mean = group[col].mean()
            std = group[col].std()

            # Get indices of missing values in this group for this column
            mask = (df["classe_name"] == class_name) & (df[col].isna())

            # Fill NaNs with: mean + U(0,1) * std
            df.loc[mask, col] = mean + np.random.uniform(-1, 1, size=mask.sum()) * std

        # Save the modified DataFrame to a new CSV file
    df.to_csv("impute.csv", index=False, sep=";")
    return df
