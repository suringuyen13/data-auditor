import pandas as pd


def profile_dataset(df):
    """Create a basic profile of a pandas DataFrame."""
    column_profile = pd.DataFrame({
        "column": df.columns,
        "data_type": df.dtypes.astype(str).values,
        "non_missing_count": df.notna().sum().values,
        "missing_count": df.isna().sum().values,
        "unique_count": df.nunique(dropna=True).values
    })
    dataset_summary = {
        "number_of_rows": df.shape[0],
        "number_of_columns": df.shape[1],
        "duplicate_rows": int(df.duplicated().sum())
    }
    return dataset_summary, column_profile


def detect_missing_values(df):
    """Calculate missing-value counts and percentages for each column."""
    missing_report = pd.DataFrame({
        "column": df.columns,
        "missing_count": df.isna().sum().values,
        "missing_percentage": (df.isna().mean() * 100).round(2).values
    })
    return missing_report.sort_values(
        by="missing_percentage", ascending=False
    ).reset_index(drop=True)
