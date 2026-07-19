import argparse
from pathlib import Path

import pandas as pd


def load_csv(file_path):
    """
    Load a CSV file into a pandas DataFrame.

    Parameters
    ----------
    file_path : str
        Path to the CSV file.

    Returns
    -------
    pandas.DataFrame
        The loaded dataset.
    """
    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    if path.suffix.lower() != ".csv":
        raise ValueError("The input file must be a CSV file.")

    try:
        df = pd.read_csv(path)
    except pd.errors.EmptyDataError:
        raise ValueError("The CSV file is empty.")
    except pd.errors.ParserError:
        raise ValueError("The CSV file could not be parsed correctly.")
    except UnicodeDecodeError:
        raise ValueError(
            "The file could not be read using UTF-8 encoding."
        )

    if df.empty:
        raise ValueError("The CSV file contains no data rows.")

    return df


def profile_dataset(df):
    """
    Create a basic profile of a pandas DataFrame.
    """
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


def display_profile(df, dataset_summary, column_profile):
    """
    Print the dataset preview and profile to the terminal.
    """
    print("\nDATASET PREVIEW")
    print("=" * 60)
    print(df.head())

    print("\nDATASET SUMMARY")
    print("=" * 60)
    print(f"Rows: {dataset_summary['number_of_rows']}")
    print(f"Columns: {dataset_summary['number_of_columns']}")
    print(f"Duplicate rows: {dataset_summary['duplicate_rows']}")

    print("\nCOLUMN PROFILE")
    print("=" * 60)
    print(column_profile.to_string(index=False))

    print("\nNUMERIC SUMMARY")
    print("=" * 60)

    numeric_columns = df.select_dtypes(include="number")

    if numeric_columns.empty:
        print("No numeric columns were detected.")
    else:
        numeric_summary = numeric_columns.describe().round(2)
        print(numeric_summary.to_string())


def main():
    parser = argparse.ArgumentParser(
        description="Inspect the basic structure and quality of a CSV file."
    )

    parser.add_argument(
        "file_path",
        help="Path to the CSV file that you want to inspect."
    )

    args = parser.parse_args()

    try:
        df = load_csv(args.file_path)
        dataset_summary, column_profile = profile_dataset(df)
        display_profile(df, dataset_summary, column_profile)

    except (FileNotFoundError, ValueError) as error:
        print(f"\nError: {error}")


if __name__ == "__main__":
    main()