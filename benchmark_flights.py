import argparse
from pathlib import Path

import pandas as pd

from auditor.config import load_rules
from auditor.data_type import validate_expected_type
from auditor.loader import load_csv
from auditor.pipeline import run_auditor
from auditor.profiler import _missing_mask


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_DIRTY = PROJECT_ROOT / "examples" / "dirty_index_flights.csv"
DEFAULT_CLEAN = PROJECT_ROOT / "examples" / "clean_index_flights.csv"
DEFAULT_TYPE_RULES = PROJECT_ROOT / "examples" / "flight_type_rules.py"


def _prepare_pair(dirty_df, clean_df, key_column):
    if list(dirty_df.columns) != list(clean_df.columns):
        raise ValueError("Dirty and clean datasets must have identical columns.")
    if key_column not in dirty_df.columns:
        raise ValueError(f"Key column not found: {key_column}")
    if dirty_df[key_column].duplicated().any() or clean_df[key_column].duplicated().any():
        raise ValueError(f"Key column must contain unique values: {key_column}")

    dirty_keys = set(dirty_df[key_column])
    clean_keys = set(clean_df[key_column])
    if dirty_keys != clean_keys:
        raise ValueError("Dirty and clean datasets must contain the same record keys.")

    return (
        dirty_df.set_index(key_column, drop=False).sort_index(),
        clean_df.set_index(key_column, drop=False).sort_index(),
    )


def _values_equal(dirty_value, clean_value):
    dirty_missing = pd.isna(dirty_value)
    clean_missing = pd.isna(clean_value)
    if dirty_missing or clean_missing:
        return dirty_missing and clean_missing
    return dirty_value == clean_value


def actual_dirty_cells(dirty_df, clean_df):
    """Return (row key, column) cells whose dirty and clean values differ."""
    cells = set()
    for column in dirty_df.columns:
        for row_key in dirty_df.index:
            if not _values_equal(dirty_df.at[row_key, column], clean_df.at[row_key, column]):
                cells.add((row_key, column))
    return cells


def actual_type_error_cells(dirty_df, clean_df, type_rules, changed_cells):
    """Return changed cells where dirty fails a type rule that clean passes."""
    cells = set()
    for column, rule in type_rules.items():
        if column not in dirty_df.columns:
            continue

        dirty_validation = validate_expected_type(dirty_df[column], rule)
        clean_validation = validate_expected_type(clean_df[column], rule)
        dirty_invalid_rows = set(dirty_validation["invalid_rows"])
        clean_invalid_rows = set(clean_validation["invalid_rows"])

        for row_key in dirty_invalid_rows - clean_invalid_rows:
            cell = (row_key, column)
            if cell in changed_cells:
                cells.add(cell)

    return cells


def actual_missing_error_cells(dirty_df, clean_df, changed_cells):
    """Return changed cells that are missing only in the dirty dataset."""
    dirty_missing = _missing_mask(dirty_df)
    clean_missing = _missing_mask(clean_df)

    return {
        (row_key, column)
        for column in dirty_df.columns
        for row_key in dirty_df.index
        if dirty_missing.at[row_key, column]
        and not clean_missing.at[row_key, column]
        and (row_key, column) in changed_cells
    }


def missing_predictions(mask):
    return {
        (row_key, column)
        for column in mask.columns
        for row_key in mask.index[mask[column]]
    }


def type_predictions(type_report):
    cells = set()
    for result in type_report:
        column = result["column"]
        if result["source"] == "user_rules":
            rows = result.get("invalid_rows", [])
        elif result["source"] == "inference":
            rows = result.get("incompatible_rows") or []
        else:
            rows = []
        cells.update((row_key, column) for row_key in rows)
    return cells


def category_predictions(category_report):
    cells = set()
    for column_report in category_report:
        column = column_report["column"]

        validation = column_report.get("validation", {})
        for record in validation.get("invalid_records", {}).values():
            cells.update((row_key, column) for row_key in record["rows"])

        formatting = validation.get("format_inconsistencies", {})
        cells.update(
            (row_key, column)
            for row_key in formatting.get("affected_rows", [])
        )

        rare_report = column_report.get("rare_categories", {})
        for record in rare_report.get("rare_records", {}).values():
            cells.update((row_key, column) for row_key in record["rows"])

    return cells


def outlier_predictions(outliers_report):
    cells = set()
    for result in outliers_report:
        column = result["column"]
        for record in result.get("outlier_records", {}).values():
            cells.update((row_key, column) for row_key in record["rows"])
    return cells


def calculate_metrics(predicted_cells, actual_cells):
    true_positives = len(predicted_cells & actual_cells)
    false_positives = len(predicted_cells - actual_cells)
    false_negatives = len(actual_cells - predicted_cells)

    precision = (
        true_positives / (true_positives + false_positives)
        if true_positives + false_positives else 0.0
    )
    recall = (
        true_positives / (true_positives + false_negatives)
        if true_positives + false_negatives else 0.0
    )
    f1_score = (
        2 * precision * recall / (precision + recall)
        if precision + recall else 0.0
    )

    return {
        "predicted": len(predicted_cells),
        "actual": len(actual_cells),
        "true_positives": true_positives,
        "false_positives": false_positives,
        "false_negatives": false_negatives,
        "precision": precision,
        "recall": recall,
        "f1_score": f1_score,
    }


def display_metrics(name, metrics):
    print(f"\n{name}")
    print("-" * 60)
    print(f"Predicted cells: {metrics['predicted']}")
    print(f"Actual positive cells: {metrics['actual']}")
    print(f"True positives: {metrics['true_positives']}")
    print(f"False positives: {metrics['false_positives']}")
    print(f"False negatives: {metrics['false_negatives']}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall: {metrics['recall']:.4f}")
    print(f"F1 score: {metrics['f1_score']:.4f}")


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark the auditor against clean flight data."
    )
    parser.add_argument("--dirty", default=DEFAULT_DIRTY, help="Dirty CSV path.")
    parser.add_argument("--clean", default=DEFAULT_CLEAN, help="Clean CSV path.")
    parser.add_argument("--key", default="index", help="Unique row-key column.")
    parser.add_argument(
        "--type-rules",
        default=DEFAULT_TYPE_RULES,
        help="Python file defining type_rules. Defaults to the flight rules."
    )
    parser.add_argument("--category-rules", help="Python file defining category_rules.")
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

    print("\nAUDITOR BENCHMARK: INDEX FLIGHTS")
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
