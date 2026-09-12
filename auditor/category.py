import pandas as pd


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