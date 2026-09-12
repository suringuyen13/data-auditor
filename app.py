import argparse
from pathlib import Path
from examples.type_infer_rule import type_rules as tRules
from examples.category_rules import category_rules as CRules

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
        date_format = rule.get("format", "mixed")
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

def type_inference(series, STRONG_THRESHOLD=90, MIN_THRESHOLD=60):
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
    if bool_compatibility >= MIN_THRESHOLD:
        candidates.append({
            "type": "boolean",
            "confidence": bool_compatibility,
            "validation": bool_result,
            "reason": "Values accepted Boolean representations such as true/false, yes/no, or 1/0."
        })

    # 2. numeric
    numeric_result = validate_expected_type(series, {"type": "numeric"})
    numeric_compatibility=round(100-(numeric_result["invalid_percentage"]),2)

    integer_result = validate_expected_type(series, {"type": "integer"})
    integer_compatibility=round(100-(integer_result["invalid_percentage"]),2)

    if numeric_compatibility >= MIN_THRESHOLD:
        if integer_compatibility == numeric_compatibility:
            candidates.append({
                "type": "integer",
                "confidence": integer_compatibility,
                "validation": integer_result,
                "reason": "Values can be converted to whole numbers without a decimal remainder"
            })
        else:
            candidates.append({
                "type": "float",
                "confidence": numeric_compatibility,
                "validation": numeric_result,
                "reason": "Values can be converted to numbers. Some compatible values contain decimals, so float is more appropriate than integer."
            })

    # 3. date
    evidence = []

    date_pattern = r"^(?:\d{4}[-/]\d{1,2}[-/]\d{1,2}|\d{1,2}[-/]\d{1,2}[-/]\d{2,4})$"

    has_date_shape = text_values.str.match(date_pattern, na=False)
    date_shape_percentage = round(has_date_shape.mean() * 100,2)
    supporting_evidence_date = has_date_hints or (date_shape_percentage >= MIN_THRESHOLD) #evidence 1
    if has_date_hints:
        evidence.append("the column name appears date-related")
    if date_shape_percentage >= MIN_THRESHOLD:
        evidence.append("values have a date-like structure")

    if supporting_evidence_date:
        date_result = validate_expected_type(series, {
                "type": "date",
                "format": "mixed"
            })
        date_compatibility=round(100-(date_result["invalid_percentage"]),2) #evidence 2

        if evidence:
            evidence_message = "" + " and ".join(evidence) + "."
        else:
            evidence_message = "Values can be parsed as valid dates."

        if date_compatibility >= MIN_THRESHOLD:
            candidates.append({
                    "type": "date",
                    "confidence": date_compatibility,
                    "validation": date_result,
                    "reason": evidence_message
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
        selection_reason = candidates[0].get("reason")

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

    if (confidence >= 90):
        classification = "strong suggestion"
    elif (confidence >= 60 and multiple_candidates):
        classification = "suggestion with exception"
    else:
        classification = "low-confidence suggestion"

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
                "error": "Column specified in data type rules was not found in the dataset."
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
    print("\nDATA-TYPE VALIDATION/SUGGESTION")
    print("=" * 60, end="")
    
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
                candidates = result.get("candidates", [])
                print("Candidates: " + ", ".join(c['type'] for c in candidates))

def _is_categorical(non_missing_series, THRESHOLD = 0.05):
    dtype = non_missing_series.dtype
    right_type = (
        isinstance(dtype, pd.StringDtype)
        or isinstance(dtype, pd.CategoricalDtype)
        or dtype == object
    )
    
    normalized_series = non_missing_series.astype("string").str.strip().str.casefold()

    unique_count=normalized_series.nunique(dropna=True)
    total_count = normalized_series.count()

    if total_count == 0:
        return False

    unique_ratio=unique_count / total_count

    low_cardinality=False
    if unique_ratio < THRESHOLD:
        low_cardinality=True

    classification = "is_categorical"
    reason = ""

    if not right_type:
        classification = "not_selected_as_categorical"
        reason = "The column was not selected for category detection because its data type or value structure does not appear categorical."
    elif not low_cardinality:
        classification = "high_cardinality_skipped"
        reason = "Rare-category detection was skipped because large number of unique values indicating a possible identifier or high-cardinality column."

    return {
        "is_categorical_type": right_type,
        "low_cardinality": low_cardinality,
        "unique_count": unique_count,
        "values_tested": total_count,
        "cardinality_ratio": unique_ratio,
        "classification": classification,
        "reason": reason
    }

def _categories_normalization(series, allowed_rules):
    allowed_values = allowed_rules.get("allowed_values", [])
    case_sensitive = allowed_rules.get("case_sensitive", False)
    strip_whitespace = allowed_rules.get("strip_whitespace", True)

    normalized_series=series.astype("string")
    normalized_allowed_values = [str(v) for v in allowed_values]

    if not case_sensitive:
        normalized_series=normalized_series.str.casefold()
        normalized_allowed_values=[v.casefold() for v in normalized_allowed_values]

    if strip_whitespace:
        normalized_series=normalized_series.str.strip()
        normalized_allowed_values= [v.strip() for v in normalized_allowed_values]

    return {
        "normalized_series": normalized_series,
        "normalized_allowed_values": normalized_allowed_values,
        "allowed_values": allowed_values
    }

def validate_allowed_categories(series, normalization):
    non_missing_series = series.dropna()
    non_missing = series.notna()
    values_tested = len(non_missing_series)

    normalized_series= normalization["normalized_series"]
    normalized_allowed_values = normalization["normalized_allowed_values"]
    # future application: normalization_collision - identical allowed_values after normalization

    if not normalized_allowed_values:
        return {
            "column": series.name,
            "classification": "configuration_error",
            "error": "The allowed-values rule for column is missing or has an invalid format."
        }

    invalid_mask = non_missing & ~normalized_series.isin(normalized_allowed_values)
    invalid_count = int(invalid_mask.sum())

    classification = "outside_allowed_values"
    reason = "Some/All nonmissing values do not match the configured allowed categories."

    if invalid_count == 0:
        classification = "all_values_allowed"
        reason = "All nonmissing values match the configured allowed categories."

    invalid_percentage = round((invalid_count / values_tested) * 100, 2)

    invalid_series = series[invalid_mask]

    return {
        "values_tested": values_tested,
        "invalid_count": invalid_count,
        "invalid_percentage": invalid_percentage,
        "invalid_records": {
            value: {
                "count": int(count),
                "rows": invalid_series[invalid_series == value].index.tolist()
            }
            for value, count in invalid_series.value_counts().items()
        },
        "classification": classification,
        "reason": reason,
        "format_inconsistencies": detect_format_inconsistencies(series, normalization["allowed_values"])
    }

def detect_format_inconsistencies(series, allowed_values):
    non_missing = series.notna()
    normalized_series=series.astype("string")

    allowed_casefold = [v.casefold() for v in allowed_values]

    cap_mask = non_missing & ~normalized_series.str.strip().isin(allowed_values)
    cap_invalid_count = int(cap_mask.sum())

    whitespace_mask = non_missing & ~normalized_series.str.casefold().isin(allowed_casefold)
    whitespace_invalid_count = int(whitespace_mask.sum())

    if cap_invalid_count > 0 and whitespace_invalid_count > 0:
        classification = "case_and_whitespace_inconsistency"
        reason = "Some/All values match an allowed category only after both capitalization and whitespace normalization."
    elif cap_invalid_count > 0:
        classification = "capitalization_inconsistency"
        reason = "Values capitalization is inconsistent with the canonical allowed values."
    elif whitespace_invalid_count > 0:
        classification = "whitespace_inconsistency"
        reason = "Values contain leading or trailing whitespace but otherwise match an allowed category."
    else:
        classification = "no_formatting_inconsistencies"
        reason = "No formatting inconsistencies was found."

    format_mask = non_missing & ~normalized_series.isin(allowed_values)

    return {
        "capitalization_count": cap_invalid_count,
        "whitespace_count": whitespace_invalid_count,
        "affected_rows": series.index[format_mask].tolist(),
        "affected_values": series[format_mask].tolist(),
        "classification": classification,
        "reason": reason
    }

def detect_rare_categories(series, normalization, threshold):
    normalized_series = normalization["normalized_series"]

    category_proportions = normalized_series.value_counts(normalize=True)
    rare_categories = category_proportions[category_proportions < threshold].index # series keeps only names of rare cateories
    # future application: suggest threshold is too small for small dataset size

    rare_mask = series.notna() & series.isin(rare_categories)
    rare_series = series[rare_mask]

    rare_category_count = int(rare_mask.sum())

    if rare_category_count == 0:
        classification = "no_rare_categories"
        reason = "No valid category appears below the rarity threshold."
    else:
        classification = "rare_categories_detected"
        reason = "There are valid categories appear below the rarity threshold. Rare categories are statistical warnings and are not necessarily invalid."

    return {
        "check_performed": True,
        "rare_category_count": int(rare_mask.sum()),
        "rare_values": series[rare_mask].tolist(),
        "rare_records": {
            value: {
                "count": int(count),
                "rows": rare_series[rare_series == value].index.tolist()
            }
            for value, count in rare_series.value_counts().items()
        },
        "classification": classification,
        "reason": reason
    }

def detect_category_issues(df, category_rules):
    results=[]

    if category_rules is None:
        category_rules = {}

    # check invalid column name in type_rules
    for column in category_rules:
        if column not in df.columns:
            results.append({
                "column": column,
                "classification": "no_nonmissing_values",
                "error": "Column specified in category rules was not found in the dataset."
            })

    for column in df.columns:
        non_missing_series = df[column].dropna()
        values_tested = len(non_missing_series)
        if values_tested == 0:
            results.append({
                "column": column,
                "classification": "configuration_error",
                "error": "No nonmissing values are available for category detection."
            })
            continue

        column_report = {"column": column}
        pre_detect_check = _is_categorical(non_missing_series)
        is_categorical_type = pre_detect_check["is_categorical_type"]
        low_cardinality = pre_detect_check["low_cardinality"]

        if not is_categorical_type:
            result = pre_detect_check # not_selected_as_categorical: cl+reason
            column_report["is_categorical"] = result
            results.append(column_report)
            continue

        rule = category_rules.get(column, {})
        normalization = _categories_normalization(df[column], rule)
        if column in category_rules:
            result = validate_allowed_categories(df[column], normalization) # configuration_error, outside_allowed_values, all_values_allowed
                # + case_and_whitespace_inconsistency, capitalization_inconsistency, whitespace_inconsistency, no_formatting_inconsistencies
            column_report["validation"] = result

        if not low_cardinality:
            result = pre_detect_check # high_cardinality_skipped
            column_report["high_cardinality"] = result
            results.append(column_report)
            continue

        rare_threshold = rule.get("rare_threshold", 0.01)
        result = detect_rare_categories(df[column], normalization, rare_threshold) # rare_categories_detected, no_rare_categories
        column_report["rare_categories"] = result
        results.append(column_report)
        
    return results

def display_category_issues(df, issues_report):
    print("\nUNUSUAL CATEGORY DETECTION")
    print("=" * 60, end="")
    
    for result in issues_report:
        print(f"\nColumn: {result['column']}")
        if "classification" in result:
            print(f"Classification: {result['classification']}")
            print(f"Error: {result['error']}") # error
            continue

        if "is_categorical" in result:
            is_categorical = result["is_categorical"]
            print(f"Classification: {is_categorical['classification']}")
            print(f"Reason: {is_categorical['reason']}")
            continue

        if "validation" in result:
            validation = result["validation"]
            print("-- Allowed categories Validation --")
            print(f"Classification: {validation['classification']}")
            print(f"Reason: {validation['reason']}")
            print(f"Values tested: {validation['values_tested']}")
            print(f"Invalid values count: {validation['invalid_count']}")

            if validation['invalid_count'] != 0:
                print(f"Invalid percentage: {validation['invalid_percentage']}")
                print("Invalid values:")
                for value, info in validation["invalid_records"].items():
                    print(f"   {value}: count={info['count']}, rows={info['rows']}")  

            format = validation["format_inconsistencies"]
            print("-- Format inconsistencies Detection --")
            print(f"Classification: {format['classification']}")
            print(f"Reason: {format['reason']}")
            if format['classification'] != 'no_formatting_inconsistencies':
                if format['classification'] == 'capitalization_inconsistency':
                    print(f"Cap inconsistency count: {format['capitalization_count']}")
                elif format['classification'] == 'whitespace_inconsistency':
                    print(f"Whitespace inconsistency count: {format['whitespace_count']}")
                else:
                    print(f"Cap inconsistency count: {format['capitalization_count']}")
                    print(f"Whitespace inconsistency count: {format['whitespace_count']}")
                print(f"Affected values: {format['affected_values']}")
                print(f"Affected rows: {format['affected_rows']}")

        if "high_cardinality" in result:
            high_cardinality = result["high_cardinality"]
            print("-- High category cardinality detected --")
            print(f"Classification: {high_cardinality['classification']}")
            print(f"Reason: {high_cardinality['reason']}")
            print(f"Values tested: {high_cardinality['values_tested']}")
            print(f"Unique count: {high_cardinality['unique_count']}")
            print(f"Unique-to-All ratio: {high_cardinality['cardinality_ratio']}")
            continue

        if "rare_categories" in result:
            rare_categories = result["rare_categories"]
            print("-- Rare categories Detection --")
            print(f"Classification: {rare_categories['classification']}")
            print(f"Reason: {rare_categories['reason']}")
            print(f"Rare category count: {rare_categories['rare_category_count']}")
            print("Rare values:")
            for value, info in rare_categories["rare_records"].items():
                print(f"   {value}: count={info['count']}, rows={info['rows']}")  

def detect_outliers_iqr(df):
    """
    Detect outliers in each DataFrame column using the 1.5 * IQR method.
    """
    results = []

    for column in df.columns:
        series = df[column]
        is_numeric = pd.api.types.is_numeric_dtype(series)
        is_boolean = pd.api.types.is_bool_dtype(series)
        non_missing_series = series.dropna()
        values_tested = len(non_missing_series)

        if not is_numeric or is_boolean:
            results.append({
                "column": column,
                "check_performed": False,
                "values_tested": values_tested,
                "outlier_count": 0,
                "classification": "not_numeric_skipped",
                "reason": "IQR outlier detection applies only to numeric columns."
            })
            continue

        if values_tested == 0:
            results.append({
                "column": column,
                "check_performed": False,
                "values_tested": 0,
                "outlier_count": 0,
                "classification": "no_nonmissing_values",
                "reason": "No nonmissing values are available for IQR outlier detection."
            })
            continue

        q1 = non_missing_series.quantile(0.25)
        q3 = non_missing_series.quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - (1.5 * iqr)
        upper_bound = q3 + (1.5 * iqr)

        outlier_mask = (series < lower_bound) | (series > upper_bound)
        outlier_series = series[outlier_mask]
        outlier_count = int(outlier_mask.sum())
        outlier_percentage = round((outlier_count / values_tested) * 100, 2)

        if outlier_count == 0:
            classification = "no_outliers_detected"
            reason = "No nonmissing values fall outside the IQR bounds."
        else:
            classification = "outliers_detected"
            reason = "Values fall below the lower IQR bound or above the upper IQR bound."

        results.append({
            "column": column,
            "check_performed": True,
            "values_tested": values_tested,
            "q1": q1,
            "q3": q3,
            "iqr": iqr,
            "lower_bound": lower_bound,
            "upper_bound": upper_bound,
            "outlier_count": outlier_count,
            "outlier_percentage": outlier_percentage,
            "outlier_values": outlier_series.tolist(),
            "outlier_records": {
                value: {
                    "count": int(count),
                    "rows": outlier_series[outlier_series == value].index.tolist()
                }
                for value, count in outlier_series.value_counts().items()
            },
            "classification": classification,
            "reason": reason
        })

    return results

def display_outliers_report(df, outlisers_report):
    print("\nIQR OUTLIER DETECTION")
    print("=" * 60, end="")

    for result in outlisers_report:
        print(f"\nColumn: {result['column']}")
        print(f"Classification: {result['classification']}")
        print(f"Reason: {result['reason']}")
        print(f"Values tested: {result['values_tested']}")

        if not result["check_performed"]:
            continue

        print(f"Q1: {result['q1']}")
        print(f"Q3: {result['q3']}")
        print(f"IQR: {result['iqr']}")
        print(f"Lower bound: {result['lower_bound']}")
        print(f"Upper bound: {result['upper_bound']}")
        print(f"Outlier count: {result['outlier_count']}")
        print(f"Outlier percentage: {result['outlier_percentage']}%")

        if result["outlier_count"] != 0:
            print("Outlier values:")
            for value, info in result["outlier_records"].items():
                print(f"   {value}: count={info['count']}, rows={info['rows']}")



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
        type_report = detect_type_issues(df,tRules)
        category_report = detect_category_issues(df, CRules)
        outliers_report = detect_outliers_iqr(df)

        display_profile(df, dataset_summary, column_profile)
        display_missing_report(df, missing_report)
        display_type_issues(df, type_report)
        display_category_issues(df, category_report)
        display_outliers_report(df, outliers_report)

    except (FileNotFoundError, ValueError) as error:
        print(f"\nError: {error}")

if __name__ == "__main__":
    main()
