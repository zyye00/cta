from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import pandas as pd

INSUFFICIENT_HISTORY = "INSUFFICIENT_HISTORY"
FORMAL_REGIMES = ("LV_LC", "HV_LC", "LV_HC", "HV_HC")
REGIME_LABELS = {
    "LV_LC": "低波动、低相关",
    "HV_LC": "高波动、低相关",
    "LV_HC": "低波动、高相关",
    "HV_HC": "高波动、高相关",
    INSUFFICIENT_HISTORY: "历史不足",
}


def add_rolling_thresholds(
    monthly_environment: pd.DataFrame,
    lookback_months: int = 60,
    min_periods: int = 60,
) -> pd.DataFrame:
    """Add strictly lagged rolling medians on a complete calendar-month index."""
    required = {"volatility", "correlation"}
    if missing := required.difference(monthly_environment.columns):
        raise ValueError(f"Monthly environment is missing columns: {sorted(missing)}")
    environment = monthly_environment.copy()
    environment.index = pd.to_datetime(environment.index).to_period("M").to_timestamp("M")
    complete_index = pd.date_range(environment.index.min(), environment.index.max(), freq="ME")
    environment = environment[~environment.index.duplicated(keep="last")].reindex(complete_index)
    environment.index.name = "date"
    environment["volatility_threshold"] = (
        environment["volatility"].shift(1).rolling(lookback_months, min_periods=min_periods).median()
    )
    environment["correlation_threshold"] = (
        environment["correlation"].shift(1).rolling(lookback_months, min_periods=min_periods).median()
    )
    return environment


def classify_monthly_regimes(monthly_environment: pd.DataFrame) -> pd.DataFrame:
    """Classify each month into the four volatility-correlation states."""
    required = {"volatility", "correlation", "volatility_threshold", "correlation_threshold"}
    if missing := required.difference(monthly_environment.columns):
        raise ValueError(f"Monthly environment is missing columns: {sorted(missing)}")
    result = monthly_environment.copy()
    sufficient = result[list(required)].notna().all(axis=1)
    high_volatility = result["volatility"].gt(result["volatility_threshold"])
    high_correlation = result["correlation"].gt(result["correlation_threshold"])
    result["regime"] = INSUFFICIENT_HISTORY
    result.loc[sufficient & ~high_volatility & ~high_correlation, "regime"] = "LV_LC"
    result.loc[sufficient & high_volatility & ~high_correlation, "regime"] = "HV_LC"
    result.loc[sufficient & ~high_volatility & high_correlation, "regime"] = "LV_HC"
    result.loc[sufficient & high_volatility & high_correlation, "regime"] = "HV_HC"
    return result


def compute_regime_agreement_matrix(
    daily_regimes: Mapping[str, pd.DataFrame | pd.Series],
) -> pd.DataFrame:
    """Compare formal-regime agreement across scenario daily series."""
    if not daily_regimes:
        raise ValueError("Daily regime scenarios are empty.")
    series = {
        label: (value["regime"] if isinstance(value, pd.DataFrame) else value).astype("string")
        for label, value in daily_regimes.items()
    }
    labels = list(series)
    matrix = pd.DataFrame(np.nan, index=labels, columns=labels, dtype="float64")
    for left in labels:
        for right in labels:
            aligned = pd.concat([series[left], series[right]], axis=1).dropna()
            aligned = aligned.loc[~aligned.iloc[:, 0].eq(INSUFFICIENT_HISTORY)]
            aligned = aligned.loc[~aligned.iloc[:, 1].eq(INSUFFICIENT_HISTORY)]
            if not aligned.empty:
                matrix.loc[left, right] = aligned.iloc[:, 0].eq(aligned.iloc[:, 1]).mean()
    return matrix


def map_monthly_regimes_to_daily(
    monthly_regimes: pd.DataFrame | pd.Series,
    trading_dates: pd.Index,
) -> pd.DataFrame:
    """Make each month-end regime effective on the next available trading day."""
    monthly = monthly_regimes["regime"] if isinstance(monthly_regimes, pd.DataFrame) else monthly_regimes
    monthly = monthly.copy()
    monthly.index = pd.to_datetime(monthly.index).to_period("M").to_timestamp("M")
    dates = pd.DatetimeIndex(pd.to_datetime(trading_dates)).drop_duplicates().sort_values()
    effective_records: list[tuple[pd.Timestamp, str, pd.Timestamp]] = []
    for month_end, regime in monthly.items():
        position = dates.searchsorted(month_end, side="right")
        if position < len(dates):
            effective_records.append((dates[position], str(regime), month_end))
    effective = pd.DataFrame(
        effective_records,
        columns=["date", "regime", "effective_monthly_regime_date"],
    ).drop_duplicates("date", keep="last")
    effective = effective.set_index("date").reindex(dates).ffill()
    effective["regime"] = effective["regime"].fillna(INSUFFICIENT_HISTORY)
    effective.index.name = "date"
    return effective


def validate_target_regime(
    target_date: str | pd.Timestamp,
    daily_regimes: pd.DataFrame,
    monthly_regimes: pd.DataFrame,
) -> pd.Series:
    """Return the target-date audit record and fail loudly when classification is unavailable."""
    target = pd.Timestamp(target_date)
    if target not in daily_regimes.index:
        raise AssertionError(f"Target date {target.date()} is not a market-index trading day.")
    daily = daily_regimes.loc[target]
    source_date = pd.Timestamp(daily["effective_monthly_regime_date"])
    if pd.isna(source_date):
        raise AssertionError("No completed monthly regime exists before the target date.")
    monthly = monthly_regimes.loc[source_date]
    record = pd.Series(
        {
            "target_date": target,
            "effective_monthly_regime_date": source_date,
            "volatility": monthly["volatility"],
            "volatility_threshold": monthly["volatility_threshold"],
            "correlation": monthly["correlation"],
            "correlation_threshold": monthly["correlation_threshold"],
            "regime": daily["regime"],
        }
    )
    if record["regime"] == INSUFFICIENT_HISTORY:
        missing = monthly[
            ["volatility", "volatility_threshold", "correlation", "correlation_threshold"]
        ].index[monthly[["volatility", "volatility_threshold", "correlation", "correlation_threshold"]].isna()]
        raise AssertionError(f"Target date has insufficient history; missing monthly fields: {list(missing)}")
    if source_date >= target:
        raise AssertionError("Target date is mapped to a non-lagged monthly regime.")
    return record


def compress_regime_stages(
    daily_regimes: pd.DataFrame,
    nhci: pd.Series,
    monthly_environment: pd.DataFrame,
) -> pd.DataFrame:
    """Compress daily states into contiguous stages with market performance summaries."""
    daily = daily_regimes.reindex(pd.DatetimeIndex(nhci.index)).copy()
    daily["nhci"] = nhci
    changes = daily["regime"].ne(daily["regime"].shift())
    daily["stage_id"] = changes.cumsum().astype("int64")
    rows: list[dict[str, object]] = []
    for stage_id, group in daily.groupby("stage_id", sort=True):
        source_dates = pd.DatetimeIndex(group["effective_monthly_regime_date"].dropna().unique())
        environment = monthly_environment.reindex(source_dates)
        start_value, end_value = group["nhci"].iloc[[0, -1]]
        rows.append(
            {
                "stage_id": stage_id,
                "regime": group["regime"].iloc[0],
                "start_date": group.index.min(),
                "end_date": group.index.max(),
                "n_trading_days": len(group),
                "n_months": len(source_dates),
                "nhci_start": start_value,
                "nhci_end": end_value,
                "nhci_return": end_value / start_value - 1 if start_value > 0 else np.nan,
                "mean_volatility": environment["volatility"].mean(),
                "mean_correlation": environment["correlation"].mean(),
            }
        )
    return pd.DataFrame(rows)
