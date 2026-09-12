import pandas as pd


MISSING_MARKERS = {"nan", "none", "null", "n/a", "empty"}


def _missing_series_mask(series):
    """Return values that are null or use a common text missing marker."""
    normalized = series.astype("string").str.strip().str.casefold()
    return series.isna() | normalized.eq("") | normalized.isin(MISSING_MARKERS)


def _missing_mask(df):
    """Return cells that are null or use a common text missing-value marker."""
    return df.apply(_missing_series_mask)


def profile_dataset(df):
    """Create a basic profile of a pandas DataFrame."""
    missing = _missing_mask(df)
    column_profile = pd.DataFrame({
        "column": df.columns,
        "data_type": df.dtypes.astype(str).values,
        "non_missing_count": (~missing).sum().values,
        "missing_count": missing.sum().values,
        "unique_count": df.mask(missing).nunique(dropna=True).values
    })
    dataset_summary = {
        "number_of_rows": df.shape[0],
        "number_of_columns": df.shape[1],
        "duplicate_rows": int(df.duplicated().sum())
    }
    return dataset_summary, column_profile


def detect_missing_values(df):
    """Calculate missing-value counts and percentages for each column."""
    missing = _missing_mask(df)
    missing_report = pd.DataFrame({
        "column": df.columns,
        "missing_count": missing.sum().values,
        "missing_percentage": (missing.mean() * 100).round(2).values
    })
    return missing_report.sort_values(
        by="missing_percentage", ascending=False
    ).reset_index(drop=True)
