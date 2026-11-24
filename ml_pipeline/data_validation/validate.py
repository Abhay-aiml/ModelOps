
import pandas as pd
import numpy as np
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

logger = logging.getLogger(__name__)


def validate_dataset(data: pd.DataFrame, target_column: str):
    """
    Validate a dataset before ML training.

    Returns:
        dict: Validation report
    """

    logger.info("Starting dataset validation")

    errors = []
    warnings = []

    # 1. Check empty dataset
    if data.empty:
        errors.append("Dataset is empty")
        return {
            "valid": False,
            "errors": errors,
            "warnings": warnings
        }

    # 2. Check target column
    if target_column not in data.columns:
        errors.append(f"Target column '{target_column}' not found")

    # 3. Check duplicate records
    duplicates = int(data.duplicated().sum())

    if duplicates > 0:
        warnings.append(f"{duplicates} duplicate records detected")

    # 4. Check missing values
    missing_values = data.isnull().sum().to_dict()

    for column, count in missing_values.items():
        if count > 0:
            warnings.append(
                f"{column} contains {count} missing values"
            )

    # 5. Check infinite values in numeric columns
    numeric_data = data.select_dtypes(include=np.number)

    infinite_values = int(
        np.isinf(numeric_data).sum().sum()
    )

    if infinite_values > 0:
        errors.append(
            f"{infinite_values} infinite values detected"
        )

    # 6. Dataset information
    report = {
        "valid": len(errors) == 0,
        "rows": int(data.shape[0]),
        "columns": int(data.shape[1]),
        "duplicates": duplicates,
        "missing_values": missing_values,
        "infinite_values": infinite_values,
        "target_column": target_column,
        "errors": errors,
        "warnings": warnings
    }

    if report["valid"]:
        logger.info("Dataset validation successful")
    else:
        logger.error("Dataset validation failed")

    return report
