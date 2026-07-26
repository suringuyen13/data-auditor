import argparse
from pathlib import Path
from examples.example_type_rule import type_rules as tRules

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


def detect_missing_values(df):
    """
    Calculate missing-value counts and percentages for each column.
    """
    missing_count = df.isna().sum()
    missing_percentage = (df.isna().mean() * 100).round(2)

    missing_report = pd.DataFrame({
        "column": df.columns,
        "missing_count": missing_count.values,
        "missing_percentage": missing_percentage.values
    })

    # Show columns with the most missing data first
    missing_report = missing_report.sort_values(
        by="missing_percentage",
        ascending=False
    ).reset_index(drop=True)

    return missing_report


def display_missing_report(df, missing_report):
    """
    Display the missing-value report in the terminal.
    """
    print("\nMISSING-VALUE REPORT")
    print("=" * 60)

    total_missing = int(df.isna().sum().sum())
    rows_with_missing = int(df.isna().any(axis=1).sum())

    print(f"Total missing cells: {total_missing}")
    print(f"Rows containing missing values: {rows_with_missing}")

    columns_with_missing = missing_report[
        missing_report["missing_count"] > 0
    ]

    if columns_with_missing.empty:
        print("\nNo missing values were detected.")
    else:
        print("\nColumns with missing values:")
        print(columns_with_missing.to_string(index=False))


def validate_expected_type(series, rule):
    """
    Validate the data type of a DataFrame column against expected types.
    """
    expected_type = rule["type"].lower()
    non_missing = series.notna()

    if expected_type == "integer":
        converted = pd.to_numeric(series, errors="coerce")

        invalid_map = non_missing & (converted.isna() | converted % 1 != 0)

    elif expected_type in ["float", "numeric"]:
        converted = pd.to_numeric(series, errors="coerce")

        invalid_map = non_missing & converted.isna()

    elif expected_type == "date":
        date_format = rule.get("format")
        converted = pd.to_datetime(series, format=date_format, errors="coerce")

        invalid_map = non_missing & converted.isna()

    elif expected_type == "boolean":
        default_boolean_values = {
                "true", "false",
                "yes", "no",
                "0", "1"
            }

        boolean_values = rule.get("allowed_values", default_boolean_values)

        boolean_values = {
            str(value).strip().lower()
            for value in boolean_values
        }

        normalized = (
            series.astype("string")
            .str.strip()
            .str.lower()
        )

        invalid_map = non_missing & ~normalized.isin(boolean_values)

    elif expected_type == "string":
        invalid_map = pd.Series(
            False, index=series.index
        )

    else:
        raise ValueError(
            f"Unsupported expected type: {expected_type}"
        )

    invalid_count = int(invalid_map.sum())
    non_missing_count = int(non_missing.sum())

    if non_missing_count == 0:
        invalid_percentage = 0.0
    else:
        invalid_percentage = round((invalid_count / non_missing_count) * 100, 2)
    
    return {
        "column": series.name,
        "current_pandas_type": str(series.dtype),
        "expected_type": expected_type,
        "invalid_count": invalid_count,
        "invalid_percentage": invalid_percentage,
        "invalid_rows": series.index[invalid_map].tolist(),
        "invalid_values": series[invalid_map].tolist()
    }

def type_inference(series, THRESHOLD=90):
    """
    Suggest a data type this column most likely represents
    """
    non_missing_series = series.dropna()
    values_tested = len(non_missing_series)

    if non_missing_series.empty:
        return {
            "column": series.name,
            "current_pandas_type": str(series.dtype),
            "suggested_type": "unknown",
            "confidence": None, # not print
            "values_tested": 0, # not print
            "compatible_count": 0, # not print
            "incompatible_count": 0, # not print
            "incompatible_rows": [], # not print
            "incompatible_values": [], # not print
            "candidates": [], # not print 
            "multiple_candidates": False, # not print 
            "classification": "unknown",
            "reason": "No nonmissing values are available."
        }

    # infer from column's name
    text_values = (non_missing_series.astype("string").str.strip())

    column_name = series.name.lower()
    name_parts = set(column_name.replace("-", "_").split("_"))

    boolean_name_hints = {"is", "has", "can", "active", "approved", "enabled", "flag"}
    date_name_hints = {"date", "time", "birth", "created", "updated", "start", "end", "deadline"}
    identifier_name_hints = {"id", "code", "zip", "postal", "phone", "account", "sku"}

    has_boolean_hints = bool(name_parts & boolean_name_hints)
    has_date_hints = bool(name_parts & date_name_hints)
    has_identifier_hints = bool(name_parts & identifier_name_hints)
    has_leading_zero = text_values.str.fullmatch(r"0\d+").any()

    # reason reading
    if has_identifier_hints or has_leading_zero:
        reason = ("The column's name looks identifier-like"
            if has_identifier_hints
            else "Some column's values have leading zero")

        return {
            "column": series.name,
            "current_pandas_type": str(series.dtype),
            "suggested_type": "string",
            "confidence": None, # not print
            "values_tested": 0, # not print
            "compatible_count": 0, # not print
            "incompatible_count": 0, # not print
            "incompatible_rows": [], # not print
            "incompatible_values": [], # not print
            "candidates": [], # not print
            "multiple_candidates": False, # not print
            "classification": "safeguarded suggestion",
            "reason": reason
        }

    # considering candidates
    candidates = []

    # 1. boolean
    bool_result=validate_expected_type(series, {"type": "boolean"})
    bool_compatibility=round(100-(bool_result["invalid_percentage"]),2)
    if bool_compatibility >= THRESHOLD:
        candidates.append({
            "type": "boolean",
            "confidence": bool_compatibility,
            "validation": bool_result
        })

    # 2. numeric
    numeric_result = validate_expected_type(series, {"type": "numeric"})
    numeric_compatibility=round(100-(numeric_result["invalid_percentage"]),2)

    integer_result = validate_expected_type(series, {"type": "integer"})
    integer_compatibility=round(100-(integer_result["invalid_percentage"]),2)

    if numeric_compatibility >= THRESHOLD:
        if integer_compatibility == numeric_compatibility:
            candidates.append({
                "type": "integer",
                "confidence": integer_compatibility,
                "validation": integer_result
            })
        else:
            candidates.append({
                "type": "float",
                "confidence": numeric_compatibility,
                "validation": numeric_result
            })

    # 3. date
    date_result = validate_expected_type(series, {"type": "date"})
    date_compatibility=round(100-(date_result["invalid_percentage"]),2) #evidence 1

    date_pattern = "^(?:\d{4}[-/]\d{1,2}[-/]\d{1,2} | \d{1,2}[-/]\d{1,2}[-/]\d{2,4})$"

    has_date_shape = text_values.str.match(date_pattern, na=False)
    date_shape_percentage = round(has_date_shape.mean() * 100,2)
    supporting_evidence_date = has_date_hints or (date_shape_percentage >= THRESHOLD) #evidence 2

    if date_compatibility >= THRESHOLD & supporting_evidence_date:
        candidates.append({
                "type": "date",
                "confidence": date_compatibility,
                "validation": date_result
            })

    # 4. string (no candidate)
    if not candidates:
        return {
            "column": series.name,
            "current_pandas_type": str(series.dtype),
            "suggested_type": "string",
            "confidence": None, # not print
            "values_tested": values_tested,
            "compatible_count": 0, # not print
            "incompatible_count": 0, # not print
            "incompatible_rows": [], # not print
            "incompatible_values": [], # not print
            "candidates": [], # not print
            "multiple_candidates": False, # not print
            "classification": "fallback string",
            "reason": "No Boolean, numeric, or date candidate met the required confidence threshold."
        }
    
    # 5. Choosing candidates
    candidates.sort(
        key=lambda candidate: candidate["confidence"],
        reverse=True
    )

    candidate_types = {
        candidate["type"]
        for candidate in candidates
    }

    selected_candidate = None
    selection_reason = ""
    multiple_candidates = False

    if len(candidates) == 1:
        selected_candidate = candidates[0]
        selection_reason = "This was the only candidate that met the threshold."

    else:
        multiple_candidates = True

        if ("boolean" in candidate_types and has_boolean_hints):
            selected_candidate = next(
                candidate for candidate in candidates
                if candidate["type"] == "boolean"
            )
            selection_reason = "Multiple types were compatible, but the column name provides supporting evidence for Boolean data."

        elif ("date" in candidate_types and has_date_hints):
            selected_candidate = next(
                candidate for candidate in candidates
                if candidate["type"] == "date"
            )
            selection_reason = "Multiple types were compatible, but the column name provides supporting evidence for date data."

        elif (candidates[0][confidence] - candidates[1][confidence] >= 5):
            selected_candidate = candidates[0]
            selection_reason = "The strongest candidate exceeded the next candidate by at least five percentage points."

    # 6. cannot choose a candidate - ambiguity
    if selected_candidate is None:
        return {
            "column": series.name,
            "current_pandas_type": str(series.dtype),
            "suggested_type": "ambiguous",
            "confidence": candidates[0]["confidence"],
            "values_tested": values_tested,
            "compatible_count": 0, # not print
            "incompatible_count": None, # not print
            "incompatible_rows": None, # not print
            "incompatible_values": None, # not print
            "candidates": [
                {
                    "type": candidate["type"],
                    "confidence": candidate["confidence"]
                }
                for candidate in candidates
            ],
            "multiple_candidates": True,
            "classification": "ambiguous",
            "reason": "Multiple types met the threshold, and there was not enough evidence to choose safely."
        }

    # 7. Final report
    selection_validation = selected_candidate["validation"]
    confidence = selected_candidate["confidence"]

    if (confidence == 100):
        classification = "strong suggestion"
    else:
        classification = "suggestion with exception"

    return {
            "column": series.name,
            "current_pandas_type": str(series.dtype),
            "suggested_type": selected_candidate["type"],
            "confidence": confidence,
            "values_tested": values_tested,
            "compatible_count": values_tested - selection_validation["invalid_count"],
            "incompatible_count": selection_validation["invalid_count"],
            "incompatible_rows": selection_validation["invalid_rows"],
            "incompatible_values": selection_validation["invalid_values"],
            "candidates": [
                {
                    "type": candidate["type"],
                    "confidence": candidate["confidence"]
                }
                for candidate in candidates
            ],
            "multiple_candidates": multiple_candidates, # not print
            "classification": classification,
            "reason": selection_reason
        }

def detect_type_issues(df, type_rules):
    results=[]

    if type_rules is None:
        type_rules = {}

    # check invalid column name in type_rules
    for column in type_rules:
        if column not in df.columns:
            results.append({
                "column": column,
                "source": "configuration_error",
                "error": "Column specified in rules was not found."
            })

    for column in df.columns:
        if column in type_rules:
            rule = type_rules[column]
            result = validate_expected_type(df[column], rule)
            result["source"] = "user_rules"
        else:
            result = type_inference(df[column])
            result["source"] = "inference"

        results.append(result)  
        
    return results

def display_type_issues(df, type_report):
    for result in type_report:
        source = result["source"]

        print(f"\nColumn: {result['column']}")
        if source == "configuration_error":
            print(f"Error: {result['error']}")

        elif source == "user_rules":
            print("-- User-rule validation --")
            print(f"Current pandas type: {result['current_pandas_type']}")
            print(f"Expected type: {result['expected_type']}")
            print(f"Invalid count: {result['invalid_count']}")
            print(f"Invalid percentage: {result['invalid_percentage']}%")
            print(f"Invalid rows: {result['invalid_rows']}")
            print(f"Invalid values: {result['invalid_values']}")
        
        else:
            print("-- Automatic type inference --")
            print(f"Current pandas type: {result['current_pandas_type']}")
            print(f"Suggested type: {result['suggested_type']}")
            print(f"Classification: {result['classification']}")
            print(f"Reason: {result['reason']}")
            if result['confidence'] != None:
                print(f"Confidence: {result['confidence']}%")

            if result['values_tested'] != 0:
                print(f"Values tested: {result['values_tested']}")
            
            if result['compatible_count'] != 0:
                print(f"Compatible count: {result['compatible_count']}")
                print(f"Incompatible count: {result['incompatible_count']}")
                print(f"Incompatible rows: {result['incompatible_rows']}")
                print(f"Incompatible values: {result['incompatible_values']}")

            if result['multiple_candidates']:
                print(f"Candidates: {result['candidates'] ["type" for candidate in candidates
                ],}")




            



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
        missing_report = detect_missing_values(df)
        type_report=detect_type_issues(df,tRules)

        display_profile(df, dataset_summary, column_profile)
        display_missing_report(df, missing_report)
        display_type_issues(df, type_report)

    except (FileNotFoundError, ValueError) as error:
        print(f"\nError: {error}")

if __name__ == "__main__":
    main()