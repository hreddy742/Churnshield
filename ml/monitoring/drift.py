import numpy as np, pandas as pd
from scipy import stats
from dataclasses import dataclass
from typing import Optional

EXCLUDE_FROM_DRIFT = ["customer_id", "customer_notes", "churn_label"]

@dataclass
class FeatureDrift:
    feature: str
    type: str  # numeric | categorical
    score: float  # PSI for numeric, chi2 stat for categorical
    pvalue: Optional[float]
    status: str  # stable | warning | critical

def _psi(reference: np.ndarray, current: np.ndarray, bins: int = 10) -> float:
    eps = 1e-6
    ref_min, ref_max = reference.min(), reference.max()
    if ref_max == ref_min:
        return 0.0
    edges = np.linspace(ref_min, ref_max, bins + 1)
    ref_counts = np.histogram(reference, bins=edges)[0]
    cur_counts = np.histogram(current, bins=edges)[0]
    ref_pct = (ref_counts / ref_counts.sum()).clip(eps)
    cur_pct = (cur_counts / cur_counts.sum()).clip(eps)
    return float(np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct)))

def detect_drift(
    reference_df: pd.DataFrame,
    current_df: pd.DataFrame
) -> list:
    results = []
    columns = [c for c in reference_df.columns if c not in EXCLUDE_FROM_DRIFT]

    for col in columns:
        if col not in current_df.columns:
            continue

        ref_vals = reference_df[col].dropna()
        cur_vals = current_df[col].dropna()

        if len(cur_vals) == 0:
            continue

        if pd.api.types.is_numeric_dtype(ref_vals):
            psi = _psi(ref_vals.values, cur_vals.values)
            if psi < 0.1:
                status = "stable"
            elif psi < 0.2:
                status = "warning"
            else:
                status = "critical"
            results.append(FeatureDrift(
                feature=col, type="numeric",
                score=round(psi, 4), pvalue=None, status=status
            ))
        else:
            # Chi-squared test
            all_cats = set(ref_vals.unique()) | set(cur_vals.unique())
            ref_counts = ref_vals.value_counts().reindex(all_cats, fill_value=0)
            cur_counts = cur_vals.value_counts().reindex(all_cats, fill_value=0)
            if ref_counts.sum() > 0 and cur_counts.sum() > 0:
                chi2, pvalue, _, _ = stats.chi2_contingency(
                    np.array([ref_counts.values, cur_counts.values])
                )
                if pvalue > 0.05:
                    status = "stable"
                elif pvalue > 0.01:
                    status = "warning"
                else:
                    status = "critical"
                results.append(FeatureDrift(
                    feature=col, type="categorical",
                    score=round(float(chi2), 4),
                    pvalue=round(float(pvalue), 4),
                    status=status
                ))

    return results
