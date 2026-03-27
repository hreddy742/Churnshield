"""Structured alert logging for drift events."""

from __future__ import annotations

import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from monitoring.drift_detector import DriftResult


LOGGER = logging.getLogger("churnguard.alerts")


def log_drift_alert(drift_results: list[DriftResult]) -> None:
    """Emit a structured JSON alert if any feature shows critical drift."""

    critical_results = [result for result in drift_results if result.status == "critical"]
    if not critical_results:
        return

    payload = {
        "event": "critical_drift_detected",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "critical_features": [result.model_dump() for result in critical_results],
        "critical_feature_count": len(critical_results),
    }
    LOGGER.warning(json.dumps(payload))
