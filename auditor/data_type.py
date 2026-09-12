import pandas as pd

from auditor.profiler import _missing_series_mask

def validate_expected_type(series, rule):
    """
    Validate the data type of a DataFrame column against expected types.
    """
    expected_type = rule["type"].lower()
    non_missing = ~_missing_series_mask(series)

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
    non_missing_series = series[~_missing_series_mask(series)]
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
