from pathlib import Path

import pandas as pd


def load_csv(file_path):
    """Load a CSV file into a pandas DataFrame."""
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
        raise ValueError("The file could not be read using UTF-8 encoding.")

    if df.empty:
        raise ValueError("The CSV file contains no data rows.")

    return df
