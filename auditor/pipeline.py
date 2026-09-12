from auditor.category import detect_category_issues
from auditor.data_type import detect_type_issues
from auditor.outliers import detect_outliers_iqr
from auditor.profiler import _missing_mask, detect_missing_values, profile_dataset


def run_auditor(df, type_rules=None, category_rules=None):
    """Run the complete auditor and return all detection reports."""
    dataset_summary, column_profile = profile_dataset(df)

    return {
        "dataset_summary": dataset_summary,
        "column_profile": column_profile,
        "missing_report": detect_missing_values(df),
        "missing_mask": _missing_mask(df),
        "type_report": detect_type_issues(df, type_rules),
        "category_report": detect_category_issues(df, category_rules),
        "outliers_report": detect_outliers_iqr(df),
    }
