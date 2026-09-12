
from auditor.profiler import _missing_mask


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


def display_missing_report(df, missing_report):
    """
    Display the missing-value report in the terminal.
    """
    print("\nMISSING-VALUE REPORT")
    print("=" * 60)

    missing = _missing_mask(df)
    total_missing = int(missing.sum().sum())
    rows_with_missing = int(missing.any(axis=1).sum())

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


def display_outliers_report(df, outliers_report):
    print("\nIQR OUTLIER DETECTION")
    print("=" * 60, end="")
    for result in outliers_report:
        print(f"\nColumn: {result['column']}")
        print(f"Classification: {result['classification']}")
        print(f"Reason: {result['reason']}")
        print(f"Values tested: {result['values_tested']}")
        if not result["check_performed"]:
            continue
        for label, key in [("Q1", "q1"), ("Q3", "q3"), ("IQR", "iqr"), ("Lower bound", "lower_bound"), ("Upper bound", "upper_bound"), ("Outlier count", "outlier_count")]:
            print(f"{label}: {result[key]}")
        print(f"Outlier percentage: {result['outlier_percentage']}%")
        for value, info in result["outlier_records"].items():
            print(f"   {value}: count={info['count']}, rows={info['rows']}")
