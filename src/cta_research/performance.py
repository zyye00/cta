from __future__ import annotations

import numpy as np
import pandas as pd

from .regimes import FORMAL_REGIMES


def _periods_per_year(index: pd.DatetimeIndex) -> float:
    median_days = index.to_series().diff().dt.days.median()
    if not median_days or median_days <= 0:
        return np.nan
    if median_days <= 2:
        return 252.0
    if median_days <= 10:
        return 52.0
    if median_days <= 45:
        return 12.0
    return 365.25 / median_days


def compute_series_regime_performance(
    values: pd.Series,
    daily_regimes: pd.DataFrame,
    periods_per_year: float | None = None,
) -> pd.DataFrame:
    """Compute conditional performance for one positive NAV or index series."""
    if values.empty:
        raise ValueError("Series data is empty.")
    series = values.copy()
    series.index = pd.to_datetime(series.index)
    series = series.sort_index().rename(str(values.name or "series"))
    returns = series.pct_change(fill_method=None).iloc[1:]
    regimes = daily_regimes.reindex(returns.index)["regime"]
    annualization = periods_per_year if periods_per_year is not None else _periods_per_year(returns.index)
    rows: list[dict[str, object]] = []
    for regime in FORMAL_REGIMES:
        observations = returns.loc[regimes.eq(regime)].dropna()
        standard_deviation = observations.std(ddof=1)
        rows.append(
            {
                "regime": regime,
                "series": str(series.name),
                "n_observations": len(observations),
                "annualized_return": (
                    (1 + observations).prod() ** (annualization / len(observations)) - 1
                    if len(observations) and observations.gt(-1).all() and pd.notna(annualization)
                    else np.nan
                ),
                "annualized_volatility": (
                    standard_deviation * np.sqrt(annualization)
                    if len(observations) >= 2 and pd.notna(annualization)
                    else np.nan
                ),
                "annualized_sharpe": (
                    observations.mean() / standard_deviation * np.sqrt(annualization)
                    if len(observations) >= 2 and standard_deviation > 0 and pd.notna(annualization)
                    else np.nan
                ),
            }
        )
    return pd.DataFrame(rows).set_index("regime")


def compute_regime_annualized_returns(nav: pd.DataFrame, daily_regimes: pd.DataFrame) -> pd.DataFrame:
    """Annualize aligned NAV returns separately for each formal market regime."""
    if nav.empty:
        raise ValueError("NAV data is empty.")
    values = nav.copy()
    values.index = pd.to_datetime(values.index)
    values = values.sort_index()
    periods_per_year = _periods_per_year(pd.DatetimeIndex(values.index))
    results = [
        compute_series_regime_performance(values[name], daily_regimes, periods_per_year)
        for name in values.columns
    ]
    return pd.concat(results).set_index("series", append=True).reorder_levels(["regime", "series"])


def compute_regime_cumulative_returns(values: pd.Series, daily_regimes: pd.DataFrame) -> pd.DataFrame:
    """Accumulate returns observed only while each formal regime is active."""
    if values.empty:
        raise ValueError("Series data is empty.")
    series = values.copy()
    series.index = pd.to_datetime(series.index)
    series = series.sort_index()
    returns = series.pct_change(fill_method=None).iloc[1:]
    regimes = daily_regimes.reindex(returns.index)["regime"]
    cumulative = {
        regime: (1 + returns.loc[regimes.eq(regime)].dropna()).cumprod()
        for regime in FORMAL_REGIMES
    }
    return pd.concat(cumulative, axis=1).sort_index()
