"""Data audit for Rawaj Pay transfer extracts.

This module checks data quality and potential label leakage before a model is
trained. It does not read files or access the network.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


def _missing_mask(series: pd.Series) -> pd.Series:
    """Mark NaN/None and strings that are empty or contain only whitespace."""
    blank_strings = series.map(lambda value: isinstance(value, str) and not value.strip())
    return (series.isna() | blank_strings).astype(bool)


def _sorted_values(values: list[Any]) -> list[Any]:
    """Sort values naturally, with a deterministic fallback for mixed types."""
    try:
        return sorted(values)
    except TypeError:
        return sorted(values, key=lambda value: (type(value).__name__, str(value)))


def _has_at_most_one_value_including_missing(
    series: pd.Series, missing: pd.Series
) -> bool:
    """Whether the full column has at most one value, counting missing as a value."""
    non_missing_unique = int(series.loc[~missing].nunique(dropna=True))
    distinct_total = non_missing_unique + int(bool(missing.any()))
    return distinct_total <= 1


def _auc_is_extreme(labels: np.ndarray, values: np.ndarray) -> bool:
    """Return whether ROC AUC is at least 0.98 or at most 0.02.

    ROC AUC is undefined when the rows supplied contain only one label class;
    such a candidate is therefore not considered a leak by this check.
    """
    if labels.size == 0 or np.unique(labels).size < 2:
        return False
    try:
        auc = float(roc_auc_score(labels, values))
    except (ValueError, TypeError, FloatingPointError):
        return False
    return auc >= 0.98 or auc <= 0.02


def audit(df: pd.DataFrame, *, target: str, id_col: str, time_col: str) -> dict:
    """Audit a transfer extract before anyone trains a model on it.

    Raises KeyError if `target`, `id_col` or `time_col` is not a column of `df`.

    Returns a dict with exactly these keys:

    - "n_rows" (int): number of rows.
    - "missing" ({column: float}): fraction missing, for every column. Missing means
      NaN/None, or a string that is empty or only whitespace.
    - "duplicate_ids" (list): sorted non-missing `id_col` values that appear more than once.
    - "duplicate_rows" (int): rows that exactly repeat an earlier row.
    - "constant_columns" (list): sorted columns other than `target` with at most one
      distinct non-missing value.
    - "unlabelled_rows" (int): rows where `target` is missing.
    - "positive_rate" (float): share of labelled rows with target == 1; 0.0 if none is labelled.
    - "bad_timestamps" (int): non-missing values in `time_col` that do not parse as dates.
    - "time_range" (list): [min, max] of the parsed timestamps as ISO strings, or [None, None].
    - "leakage_suspects" (list): sorted columns (not target, id_col or time_col) that look
      like leaks. See README.md for the exact rule.
    """
    
    for required_column in (target, id_col, time_col):
        if required_column not in df.columns:
            raise KeyError(required_column)

    n_rows = int(len(df))

    # Whitespace-only strings count as missing in every column.
    missing_masks = {column: _missing_mask(df[column]) for column in df.columns}
    missing = {
        column: (float(mask.sum() / n_rows) if n_rows else 0.0)
        for column, mask in missing_masks.items()
    }

    # Duplicate IDs: missing IDs are ignored, and each duplicated ID appears once.
    id_missing = missing_masks[id_col]
    non_missing_ids = df.loc[~id_missing, id_col]
    duplicated_id_values = non_missing_ids.loc[
        non_missing_ids.duplicated(keep=False)
    ].drop_duplicates().tolist()
    duplicate_ids = _sorted_values(duplicated_id_values)

    duplicate_rows = int(df.duplicated(keep="first").sum())

    constant_columns = []
    for column in df.columns:
        if column == target:
            continue
        non_missing_values = df.loc[~missing_masks[column], column]
        if non_missing_values.nunique(dropna=True) <= 1:
            constant_columns.append(column)
    constant_columns = _sorted_values(constant_columns)

    target_missing = missing_masks[target]
    labelled = df.loc[~target_missing, target]
    labelled_count = int(len(labelled))
    unlabelled_rows = int(target_missing.sum())
    positive_count = int((labelled == 1).sum())
    positive_rate = float(positive_count / labelled_count) if labelled_count else 0.0

    # Parse timestamps, but count only present values that fail to parse as bad.
    time_missing = missing_masks[time_col]
    parsed_times = pd.to_datetime(df[time_col], errors="coerce")
    bad_timestamps = int((~time_missing & parsed_times.isna()).sum())
    valid_times = parsed_times.loc[parsed_times.notna()]
    if valid_times.empty:
        time_range = [None, None]
    else:
        minimum = pd.Timestamp(valid_times.min()).isoformat()
        maximum = pd.Timestamp(valid_times.max()).isoformat()
        time_range = [minimum, maximum]

    # Leakage checks are performed on labelled rows only. For the value-AUC
    # check, values must also be present and both label classes must remain.
    leakage_suspects = []
    eligible_for_leak_check = (
        labelled_count >= 20
        and bool((labelled == 0).any())
        and bool((labelled == 1).any())
    )
    excluded = {target, id_col, time_col}

    if eligible_for_leak_check:
        labelled_positions = np.flatnonzero((~target_missing).to_numpy())
        labels = (labelled.to_numpy() == 1).astype(np.int8)

        for column in df.columns:
            if column in excluded:
                continue

            column_missing_full = missing_masks[column]
            # Missing is itself a value for this exception. A column with one
            # value across the extract (including the all-blank case) is skipped.
            if _has_at_most_one_value_including_missing(df[column], column_missing_full):
                continue

            series = df[column].iloc[labelled_positions]
            missing_on_labelled = missing_masks[column].iloc[labelled_positions].to_numpy(dtype=bool)
            is_suspect = False

            # (a) Numeric/boolean value leakage, requiring 20 present labelled values.
            is_numeric_or_boolean = (
                pd.api.types.is_numeric_dtype(df[column].dtype)
                or pd.api.types.is_bool_dtype(df[column].dtype)
            )
            present_mask = ~missing_on_labelled
            if is_numeric_or_boolean and int(present_mask.sum()) >= 20:
                feature_labels = labels[present_mask]
                feature_values = pd.to_numeric(series, errors="coerce").to_numpy(dtype=float)[present_mask]
                if np.unique(feature_labels).size >= 2:
                    is_suspect = _auc_is_extreme(feature_labels, feature_values)

            # (b) Missingness leakage, allowed for columns of any dtype.
            if not is_suspect:
                missing_indicator = missing_on_labelled.astype(np.int8)
                is_suspect = _auc_is_extreme(labels, missing_indicator)

            if is_suspect:
                leakage_suspects.append(column)

    leakage_suspects = _sorted_values(leakage_suspects)

    return {
        "n_rows": n_rows,
        "missing": missing,
        "duplicate_ids": duplicate_ids,
        "duplicate_rows": duplicate_rows,
        "constant_columns": constant_columns,
        "unlabelled_rows": unlabelled_rows,
        "positive_rate": positive_rate,
        "bad_timestamps": bad_timestamps,
        "time_range": time_range,
        "leakage_suspects": leakage_suspects,
    }

