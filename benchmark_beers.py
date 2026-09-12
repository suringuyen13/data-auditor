import argparse
from pathlib import Path

from auditor.config import load_rules
from auditor.loader import load_csv
from auditor.pipeline import run_auditor
from benchmark_flights import (
    _prepare_pair,
    actual_dirty_cells,
    actual_missing_error_cells,
    actual_type_error_cells,
    calculate_metrics,
    category_predictions,
    display_metrics,
    missing_predictions,
    outlier_predictions,
    type_predictions,
)


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_DIRTY = PROJECT_ROOT / "examples" / "dirty_index_beers.csv"
DEFAULT_CLEAN = PROJECT_ROOT / "examples" / "clean_index_beers.csv"
DEFAULT_TYPE_RULES = PROJECT_ROOT / "examples" / "beer_type_rules.py"


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark the auditor against clean beer data."
    )
    parser.add_argument("--dirty", default=DEFAULT_DIRTY, help="Dirty CSV path.")
    parser.add_argument("--clean", default=DEFAULT_CLEAN, help="Clean CSV path.")
    parser.add_argument("--key", default="index", help="Unique row-key column.")
    parser.add_argument(
        "--type-rules",
        default=DEFAULT_TYPE_RULES,
        help="Python file defining type_rules. Defaults to the beer rules."
    )
    parser.add_argument(
        "--category-rules",
        help="Optional Python file defining category_rules."
    )
    args = parser.parse_args()

    dirty_df, clean_df = _prepare_pair(
        load_csv(args.dirty), load_csv(args.clean), args.key
    )
    type_rules = load_rules(args.type_rules, "type_rules")
    category_rules = (
        load_rules(args.category_rules, "category_rules")
        if args.category_rules else None
    )

    auditor_report = run_auditor(dirty_df, type_rules, category_rules)
    actual = actual_dirty_cells(dirty_df, clean_df)
    actual_missing_errors = actual_missing_error_cells(
        dirty_df, clean_df, actual
    )
    actual_type_errors = actual_type_error_cells(
        dirty_df, clean_df, type_rules, actual
    )
    predictions = {
        "Missing-value detection": missing_predictions(auditor_report["missing_mask"]),
        "Data-type detection": type_predictions(auditor_report["type_report"]),
        "Category detection": category_predictions(auditor_report["category_report"]),
        "IQR outlier detection": outlier_predictions(auditor_report["outliers_report"]),
    }

    print("\nAUDITOR BENCHMARK: INDEX BEERS")
    print("=" * 60)
    ground_truths = {
        "Missing-value detection": actual_missing_errors,
        "Data-type detection": actual_type_errors,
        "Category detection": actual,
        "IQR outlier detection": actual,
    }
    for name, predicted in predictions.items():
        display_metrics(name, calculate_metrics(predicted, ground_truths[name]))

    combined = set().union(*predictions.values())
    display_metrics("Combined auditor", calculate_metrics(combined, actual))


if __name__ == "__main__":
    main()
