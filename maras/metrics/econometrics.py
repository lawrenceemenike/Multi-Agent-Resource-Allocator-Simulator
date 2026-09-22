"""Econometric metrics and market anomaly detectors: Gini coefficient and HHI concentration."""
from __future__ import annotations

import numpy as np
from typing import Dict, List, Union


def calculate_gini(wealth_distribution: Union[List[float], np.ndarray]) -> float:
    """Computes the standard Gini coefficient of inequality for wealth or income.

    Returns a float in [0.0, 1.0].
    """
    arr = np.array(wealth_distribution, dtype=np.float64)
    arr = arr[arr >= 0]
    if len(arr) <= 1 or np.sum(arr) <= 1e-12:
        return 0.0

    # Sort ascending
    arr = np.sort(arr)
    n = len(arr)
    index = np.arange(1, n + 1)
    return float(((2.0 * np.sum(index * arr)) / (n * np.sum(arr))) - ((n + 1) / n))


def calculate_hhi(holdings: Union[Dict[str, float], List[float], np.ndarray]) -> float:
    """Computes the Herfindahl-Hirschman Index (HHI) for resource or market concentration.

    HHI = sum((market_share_i * 100)^2) = 10,000 * sum(s_i^2).
    Returns a float in [0.0, 10000.0].
    - HHI < 1,500: Unconcentrated / Competitive
    - 1,500 <= HHI <= 2,500: Moderately Concentrated
    - HHI > 2,500: Highly Concentrated
    - HHI > 4,000: Severe Oligopoly / Monopoly Cornering
    """
    if isinstance(holdings, dict):
        values = list(holdings.values())
    else:
        values = list(holdings)

    arr = np.array(values, dtype=np.float64)
    arr = arr[arr > 0]
    total = np.sum(arr)
    if total <= 1e-12:
        return 0.0

    shares = arr / total
    hhi = 10000.0 * float(np.sum(shares ** 2))
    return float(min(10000.0, hhi))


def detect_anomalies(
    hhi_by_resource: Dict[str, float],
    res_holdings_by_agent: Optional[Dict[str, Dict[str, float]]] = None,
    hhi_cornering_threshold: float = 6500.0,
) -> List[str]:
    """Scans market metrics for true single-agent adversarial cornering and market manipulation alerts.

    In small multi-agent setups (e.g. 2 producers), duopoly baseline HHI is naturally ~5,000.
    A true cornering / monopoly attack occurs when HHI > 6,500 (or a single agent controls >75% of supply).
    """
    alerts = []
    for res_id, hhi in hhi_by_resource.items():
        if hhi > hhi_cornering_threshold:
            alerts.append(
                f"[ADV-02 ALERT] Severe Monopolistic Cornering on {res_id}: HHI={hhi:.0f} > {hhi_cornering_threshold:.0f}"
            )
    return alerts
