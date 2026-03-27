"""Feature drift detection utilities for numeric and categorical data."""

from __future__ import annotations

from typing import Literal

import numpy as np
import pandas as pd
from pydantic import BaseModel
from scipy.stats import chi2_contingency


DriftStatus = Literal["no_drift", "warning", "critical"]


class DriftResult(BaseModel):
    """Single-feature drift result."""

    feature: str
    psi: float | None = None
    chi2_stat: float | None = None
    status: DriftStatus


def _safe_proportions(counts: np.ndarray) -> np.ndarray:
    counts = np.asarray(counts, dtype=float)
    total = counts.sum()
    if total <= 0:
        return np.full_like(counts, 1.0 / max(len(counts), 1), dtype=float)
    proportions = counts / total
    return np.clip(proportions, 1e-6, None)


def _compute_psi(reference: pd.Series, current: pd.Series, bins: int = 10) -> float:
    reference_clean = pd.to_numeric(reference, errors="coerce").dropna()
    current_clean = pd.to_numeric(current, errors="coerce").dropna()

    if reference_clean.empty or current_clean.empty:
        return 0.0

    quantiles = np.linspace(0.0, 1.0, bins + 1)
    breakpoints = np.unique(np.quantile(reference_clean, quantiles))
    if breakpoints.size < 3:
        return 0.0

    reference_counts, _ = np.histogram(reference_clean, bins=breakpoints)
    current_counts, _ = np.histogram(current_clean, bins=breakpoints)

    reference_pct = _safe_proportions(reference_counts)
    current_pct = _safe_proportions(current_counts)
    psi = np.sum((reference_pct - current_pct) * np.log(reference_pct / current_pct))
    return float(psi)


def _status_from_psi(psi: float) -> DriftStatus:
    if psi > 0.2:
        return "critical"
    if psi >= 0.1:
        return "warning"
    return "no_drift"


def _status_from_p_value(p_value: float) -> DriftStatus:
    if p_value < 0.01:
        return "critical"
    if p_value < 0.05:
        return "warning"
    return "no_drift"


def _compute_categorical_chi2(reference: pd.Series, current: pd.Series) -> tuple[float, DriftStatus]:
    reference_counts = reference.fillna("missing").astype(str).value_counts()
    current_counts = current.fillna("missing").astype(str).value_counts()
    categories = sorted(set(reference_counts.index).union(current_counts.index))

    contingency = np.array(
        [
            [float(reference_counts.get(category, 0.0)) for category in categories],
            [float(current_counts.get(category, 0.0)) for category in categories],
        ]
    )
    if contingency.sum() == 0 or contingency.shape[1] <= 1:
        return 0.0, "no_drift"

    chi2_stat, p_value, _, _ = chi2_contingency(contingency)
    return float(chi2_stat), _status_from_p_value(float(p_value))


def detect_drift(reference_df: pd.DataFrame, current_df: pd.DataFrame) -> list[DriftResult]:
    """Compare reference and current windows and return per-feature drift results."""

    if reference_df.empty or current_df.empty:
        return []

    common_columns = [column for column in reference_df.columns if column in current_df.columns]
    results: list[DriftResult] = []

    for column in common_columns:
        if pd.api.types.is_numeric_dtype(reference_df[column]):
            psi = _compute_psi(reference_df[column], current_df[column])
            results.append(
                DriftResult(
                    feature=column,
                    psi=psi,
                    chi2_stat=None,
                    status=_status_from_psi(psi),
                )
            )
        else:
            chi2_stat, status = _compute_categorical_chi2(reference_df[column], current_df[column])
            results.append(
                DriftResult(
                    feature=column,
                    psi=None,
                    chi2_stat=chi2_stat,
                    status=status,
                )
            )

    return results
