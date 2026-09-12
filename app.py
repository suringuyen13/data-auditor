import argparse

from auditor.category import detect_category_issues
from auditor.config import load_rules
from auditor.data_type import detect_type_issues
from auditor.loader import load_csv
from auditor.outliers import detect_outliers_iqr
from auditor.profiler import detect_missing_values, profile_dataset
from auditor.reporter import (
    display_category_issues,
    display_missing_report,
    display_outliers_report,
    display_profile,
    display_type_issues,
)


def main():
    parser = argparse.ArgumentParser(
        description="Inspect the basic structure and quality of a CSV file."
    )
    parser.add_argument(
        "file_path",
        help="Path to the CSV file that you want to inspect."
    )
    parser.add_argument(
        "--type-rules",
        help="Path to a Python file defining a type_rules dictionary."
    )
    parser.add_argument(
        "--category-rules",
        help="Path to a Python file defining a category_rules dictionary."
    )
    args = parser.parse_args()

    try:
        df = load_csv(args.file_path)
        type_rules = (
            load_rules(args.type_rules, "type_rules")
            if args.type_rules else None
        )
        category_rules = (
            load_rules(args.category_rules, "category_rules")
            if args.category_rules else None
        )
        dataset_summary, column_profile = profile_dataset(df)
        missing_report = detect_missing_values(df)
        type_report = detect_type_issues(df, type_rules)
        category_report = detect_category_issues(df, category_rules)
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
