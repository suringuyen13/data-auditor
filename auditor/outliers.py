import pandas as pd


def detect_outliers_iqr(df):
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
            "outlier_percentage": round((outlier_count / values_tested) * 100, 2),
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
