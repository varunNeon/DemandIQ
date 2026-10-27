"""
Data validation module for DemandIQ.

Validates raw DataFrames before ingestion into the database.
Records all issues as DataQualityEvents.

Distinguishes between:
- Genuine zero sales (valid observation)
- Missing records (absent rows — treated as unknown, not zero)
- Negative sales (data error)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import date
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)

# Required columns and their expected dtypes
SALES_REQUIRED_COLUMNS = {
    "date": "date",
    "store_id": "str",
    "product_id": "str",
    "units_sold": "numeric",
}

STORE_REQUIRED_COLUMNS = {"store_id": "str"}
PRODUCT_REQUIRED_COLUMNS = {"product_id": "str", "family": "str"}


@dataclass
class ValidationResult:
    """Summary of validation findings for one dataset."""

    dataset: str
    total_rows: int = 0
    valid_rows: int = 0
    issues: list[dict[str, Any]] = field(default_factory=list)

    def add_issue(
        self,
        issue_type: str,
        severity: str,
        row_count: int = 0,
        details: str = "",
    ) -> None:
        self.issues.append(
            {
                "dataset": self.dataset,
                "issue_type": issue_type,
                "severity": severity,
                "row_count": row_count,
                "details": details,
            }
        )

    @property
    def has_errors(self) -> bool:
        return any(i["severity"] == "error" for i in self.issues)

    def summary(self) -> str:
        lines = [f"Dataset: {self.dataset} | Rows: {self.total_rows} | Valid: {self.valid_rows}"]
        for issue in self.issues:
            lines.append(f"  [{issue['severity'].upper()}] {issue['issue_type']}: {issue['details']}")
        return "\n".join(lines)


def validate_sales(df: pd.DataFrame) -> ValidationResult:
    """
    Validate a raw sales DataFrame.

    Checks:
    1. Required columns present
    2. Date column parseable
    3. units_sold is numeric
    4. No negative sales (data error)
    5. Duplicate (date, store_id, product_id) combinations
    6. Implausibly large outliers (>10 000 units/day — flagged, not removed)
    7. Missing store_id or product_id
    """
    result = ValidationResult(dataset="sales_history", total_rows=len(df))

    # 1. Required columns
    missing_cols = [c for c in SALES_REQUIRED_COLUMNS if c not in df.columns]
    if missing_cols:
        result.add_issue(
            "missing_columns",
            "error",
            row_count=0,
            details=f"Missing columns: {missing_cols}",
        )
        return result  # Cannot continue without required columns

    # 2. Date parsing
    try:
        df = df.copy()
        df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.date
        bad_dates = df["date"].isna().sum()
        if bad_dates > 0:
            result.add_issue(
                "unparseable_dates",
                "error",
                row_count=int(bad_dates),
                details=f"{bad_dates} rows have unparseable dates and will be dropped.",
            )
    except Exception as exc:
        result.add_issue("date_parse_error", "error", details=str(exc))
        return result

    # 3. Numeric units_sold
    df["units_sold"] = pd.to_numeric(df["units_sold"], errors="coerce")
    non_numeric = df["units_sold"].isna().sum()
    if non_numeric > 0:
        result.add_issue(
            "non_numeric_units_sold",
            "error",
            row_count=int(non_numeric),
            details=f"{non_numeric} rows have non-numeric units_sold.",
        )

    # 4. Negative sales
    negative = (df["units_sold"] < 0).sum()
    if negative > 0:
        result.add_issue(
            "negative_units_sold",
            "error",
            row_count=int(negative),
            details=f"{negative} rows have negative units_sold (likely returns or data errors).",
        )

    # 5. Duplicates
    dup_mask = df.duplicated(subset=["date", "store_id", "product_id"], keep="first")
    n_dups = dup_mask.sum()
    if n_dups > 0:
        result.add_issue(
            "duplicate_records",
            "warning",
            row_count=int(n_dups),
            details=f"{n_dups} duplicate (date, store_id, product_id) rows. First occurrence kept.",
        )

    # 6. Outliers
    p999 = df["units_sold"].quantile(0.999) if len(df) >= 1000 else None
    if p999 and p999 > 10_000:
        n_outliers = (df["units_sold"] > 10_000).sum()
        result.add_issue(
            "extreme_outliers",
            "warning",
            row_count=int(n_outliers),
            details=f"{n_outliers} rows with units_sold > 10,000. Flagged for review.",
        )

    # 7. Missing identifiers
    null_store = df["store_id"].isna().sum()
    null_product = df["product_id"].isna().sum()
    if null_store:
        result.add_issue("null_store_id", "error", row_count=int(null_store))
    if null_product:
        result.add_issue("null_product_id", "error", row_count=int(null_product))

    # Zero sales — these are VALID observations (genuine zero demand), log info only
    zero_sales = (df["units_sold"] == 0).sum()
    if zero_sales > 0:
        pct = 100 * zero_sales / len(df)
        result.add_issue(
            "zero_sales_days",
            "info",
            row_count=int(zero_sales),
            details=(
                f"{zero_sales} rows ({pct:.1f}%) have units_sold=0. "
                "These are treated as genuine zero-demand observations, not missing data."
            ),
        )

    valid = len(df) - int(negative) - int(non_numeric) - int(n_dups)
    result.valid_rows = max(0, valid)
    return result


def validate_stores(df: pd.DataFrame) -> ValidationResult:
    result = ValidationResult(dataset="stores", total_rows=len(df))
    if "store_id" not in df.columns:
        result.add_issue("missing_columns", "error", details="Missing: store_id")
        return result
    dupes = df["store_id"].duplicated().sum()
    if dupes:
        result.add_issue("duplicate_store_ids", "warning", row_count=int(dupes))
    result.valid_rows = len(df) - int(dupes)
    return result


def validate_products(df: pd.DataFrame) -> ValidationResult:
    result = ValidationResult(dataset="products", total_rows=len(df))
    missing = [c for c in PRODUCT_REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        result.add_issue("missing_columns", "error", details=f"Missing: {missing}")
        return result
    dupes = df["product_id"].duplicated().sum()
    if dupes:
        result.add_issue("duplicate_product_ids", "warning", row_count=int(dupes))
    result.valid_rows = len(df) - int(dupes)
    return result
