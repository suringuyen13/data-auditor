# Data Quality Auditor

## Overview

Data Quality Auditor is a command-line Python project for profiling CSV files and detecting common data-quality problems. It produces readable terminal reports for missing values, incompatible data types, unusual categories, and numeric outliers.

The auditor can run automatically or accept user-defined type and category rules. Detection results include classifications, reasons, affected values, and original row indexes so they can later support a cleaning workflow.

## Features

- CSV validation and loading.
- Dataset, column, and numeric summaries.
- Missing-value detection for pandas nulls, blank cells, and the text markers `nan`, `none`, `null`, `n/a`, and `empty`.
- Automatic data-type inference or validation against user rules.
- Category validation, capitalization and whitespace checks, high-cardinality safeguards, and rare-category warnings.
- Numeric outlier detection using the 1.5 × IQR method.
- Per-column classifications and human-readable reasons.
- Cell-level benchmark scripts for the indexed flights and beers datasets.

## Demo / Screenshots

Run an audit from the terminal:

```bash
python app.py examples/dirty_index_flights.csv \
  --type-rules examples/flight_type_rules.py
```

Example report shape:

```text
IQR OUTLIER DETECTION
============================================================
Column: example_numeric_column
Classification: outliers_detected
Reason: Values fall below the lower IQR bound or above the upper IQR bound.
Values tested: 50
Outlier count: 2
```

The project currently provides terminal output rather than a graphical interface, so no application screenshots are included.

## Benchmark Results

The missing-value benchmark currently produces these verified cell-level results:

| Dataset | Precision | Recall | F1 | Notes |
|---|---:|---:|---:|---|
| Indexed flights | 1.0000 | 1.0000 | 1.0000 | 2,312 dirty-only missing cells detected |
| Indexed beers | 0.1058 | 1.0000 | 0.1914 | 127 dirty-only missing cells detected; 1,073 cells are missing in both dirty and clean data |

The 1,073 beer cells are false positives under the corruption-specific benchmark because the clean reference also contains missing values. They are still valid missing-value detections if the goal is to report every missing cell.

Run the complete benchmarks to calculate results for the current code:

```bash
python benchmark_flights.py
python benchmark_beers.py
```

## How It Works

1. `app.py` loads a CSV file and optional rule files.
2. `run_auditor()` sends the DataFrame through the shared audit pipeline.
3. The profiler calculates dataset statistics and missing-value results.
4. The type detector either validates configured types or infers likely types.
5. The category detector validates allowed values and identifies formatting, cardinality, and rarity concerns.
6. The outlier detector evaluates numeric columns using Q1, Q3, and 1.5 × IQR boundaries.
7. The reporter prints the resulting classifications, reasons, counts, values, and affected rows.

The auditor analyzes data without modifying the input DataFrame.

## Tech Stack

- Python
- pandas
- argparse
- Standard-library `pathlib` and `importlib`

## Installation

Clone the repository, open its directory, and create a virtual environment:

```bash
python -m venv .venv
```

Activate it on Windows:

```powershell
.venv\Scripts\Activate.ps1
```

Install the dependency:

```bash
pip install -r requirements.txt
```

## Usage

Run automatic detection without rules:

```bash
python app.py path/to/data.csv
```

Run with type rules:

```bash
python app.py path/to/data.csv --type-rules path/to/type_rules.py
```

Run with both type and category rules:

```bash
python app.py path/to/data.csv \
  --type-rules path/to/type_rules.py \
  --category-rules path/to/category_rules.py
```

A type-rule file must define a `type_rules` dictionary:

```python
type_rules = {
    "age": {"type": "integer"},
    "departure_time": {"type": "date", "format": "%I:%M %p"},
}
```

A category-rule file must define a `category_rules` dictionary:

```python
category_rules = {
    "status": {
        "allowed_values": ["active", "inactive"],
        "case_sensitive": False,
        "strip_whitespace": True,
        "rare_threshold": 0.01,
    }
}
```

Rule files are executed as Python modules. Only load rule files you trust.

## Project Structure

```text
data-auditor/
├── app.py                       # CLI entry point
├── benchmark_flights.py         # Flights benchmark
├── benchmark_beers.py           # Beers benchmark
├── requirements.txt
├── auditor/
│   ├── category.py              # Category detection
│   ├── config.py                # Rule-file loading
│   ├── data_type.py             # Type inference and validation
│   ├── loader.py                # CSV loading
│   ├── outliers.py              # IQR outlier detection
│   ├── pipeline.py              # Shared audit pipeline
│   ├── profiler.py              # Profiling and missing detection
│   └── reporter.py              # Terminal report formatting
├── examples/
│   ├── dirty_index_flights.csv
│   ├── clean_index_flights.csv
│   ├── flight_type_rules.py
│   ├── dirty_index_beers.csv
│   ├── clean_index_beers.csv
│   └── beer_type_rules.py
└── tests/                        # Test modules
```

## Benchmark Methodology

Each benchmark loads a dirty dataset, runs it through the same `run_auditor()` pipeline used by `app.py`, and uses the corresponding clean dataset as ground truth. Records are aligned using the unique `index` column, and predictions are evaluated at cell level.

The metrics are:

```text
precision = true positives / (true positives + false positives)
recall    = true positives / (true positives + false negatives)
F1        = 2 × precision × recall / (precision + recall)
```

Missing-value ground truth contains only changed cells that are missing in the dirty data but not missing in the clean reference. Type ground truth contains only changed cells where the dirty value fails a configured type rule that the clean value passes. Combined auditor scoring continues to use all dirty-versus-clean cell differences.

This detector-specific ground truth prevents valid-but-different values, such as two correctly formatted times, from being counted as missed type errors.

## Limitations & Future Improvements

- Data-type validation detects representational problems, not semantically incorrect but valid values.
- The category and IQR benchmark sections still use broad dirty-versus-clean differences; they need detector-specific labeled ground truth.
- A clean reference may legitimately contain missing values, which affects how missing-detector precision should be interpreted.
- Numeric-looking columns loaded as text are skipped by IQR detection until their types are cleaned or converted.
- Duplicate column names are not yet handled explicitly.
- Rule files currently execute Python code; JSON or YAML configuration would be safer for untrusted inputs.
- Automated tests and benchmark result export should be expanded.
- A future cleaning layer can use reported row indexes and values to remove, replace, cap, or normalize detected issues.
