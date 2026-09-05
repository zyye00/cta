from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class MonthlyEnvironmentResult:
    """Monthly metrics plus the diagnostics needed for independent review."""

    metrics: pd.DataFrame
    volatility_by_month: dict[pd.Timestamp, pd.Series]
    correlation_by_month: dict[pd.Timestamp, pd.DataFrame]
    exclusions_by_month: dict[pd.Timestamp, dict[str, str]]
    sector_counts_by_month: dict[pd.Timestamp, pd.Series]


@dataclass(frozen=True)
class RampInEnvironmentResult:
    """Raw and linear-ramp environment results plus the applied weights."""

    raw_result: MonthlyEnvironmentResult
    ramp_result: MonthlyEnvironmentResult
    variety_weights_by_month: dict[pd.Timestamp, pd.Series]
    member_weights_by_month: dict[pd.Timestamp, pd.Series]
    pair_weights_by_month: dict[pd.Timestamp, pd.Series]


def compute_sector_volatility_contribution_shares(
    volatility_by_month: Mapping[pd.Timestamp, pd.Series],
    member_weights_by_month: Mapping[pd.Timestamp, pd.Series] | None = None,
) -> pd.DataFrame:
    """Normalize monthly sector volatilities into optional weighted contribution shares."""
    if not volatility_by_month:
        raise ValueError("Monthly sector volatility data is empty.")
    volatility = pd.DataFrame(volatility_by_month).T
    volatility.index = pd.to_datetime(volatility.index)
    volatility = volatility.sort_index()
    volatility.columns = volatility.columns.astype(str)
    volatility = volatility.apply(pd.to_numeric, errors="coerce")
    if volatility.lt(0).any().any():
        raise ValueError("Sector volatilities must be non-negative.")
    if member_weights_by_month is not None:
        weights = pd.DataFrame(member_weights_by_month).T
        weights.index = pd.to_datetime(weights.index)
        weights = weights.sort_index()
        weights.columns = weights.columns.astype(str)
        weights = weights.apply(pd.to_numeric, errors="coerce")
        if weights.lt(0).any().any():
            raise ValueError("Sector contribution weights must be non-negative.")
        volatility = volatility.mul(weights.reindex(index=volatility.index, columns=volatility.columns))
    volatility = volatility.dropna(how="all")
    totals = volatility.sum(axis=1, min_count=1)
    valid_dates = totals.index[totals.gt(0)]
    if valid_dates.empty:
        return volatility.iloc[0:0]
    shares = volatility.loc[valid_dates].div(totals.loc[valid_dates], axis=0)
    ordered_columns = shares.mean().sort_values(ascending=False, kind="stable").index
    return shares.loc[:, ordered_columns]


def compute_linear_ramp_weights(
    valid_members_by_month: Mapping[pd.Timestamp, Iterable[str]],
    ramp_in_months: int = 6,
) -> dict[pd.Timestamp, pd.Series]:
    """Assign calendar-month linear ramp weights to members entering a sample."""
    if not isinstance(ramp_in_months, int) or ramp_in_months <= 0:
        raise ValueError("ramp_in_months must be a positive integer.")
    if not valid_members_by_month:
        return {}
    normalized = {
        pd.Timestamp(date).to_period("M").to_timestamp("M"): {str(member) for member in members}
        for date, members in valid_members_by_month.items()
    }
    first_valid: dict[str, pd.Timestamp] = {}
    for date in sorted(normalized):
        for member in normalized[date]:
            first_valid.setdefault(member, date)
    weights: dict[pd.Timestamp, pd.Series] = {}
    for date in sorted(normalized):
        values = {}
        for member in sorted(normalized[date]):
            first_date = first_valid[member]
            months_since_first = (date.year - first_date.year) * 12 + date.month - first_date.month + 1
            values[member] = min(1.0, months_since_first / ramp_in_months)
        weights[date] = pd.Series(values, dtype="float64", name=date)
    return weights


def compute_log_returns(price_history: pd.DataFrame) -> pd.DataFrame:
    """Convert RQData long-form positive close levels to an unfilled return matrix."""
    data = price_history.reset_index() if isinstance(price_history.index, pd.MultiIndex) else price_history.copy()
    if {"underlying_symbol", "close"}.issubset(data.columns):
        data = data.rename(columns={"underlying_symbol": "index_code", "close": "value"})
    required = {"date", "index_code", "value"}
    if missing := required.difference(data.columns):
        raise ValueError(f"Price history is missing columns: {sorted(missing)}")
    data = data[list(required)].copy()
    data["date"] = pd.to_datetime(data["date"], errors="coerce")
    data["value"] = pd.to_numeric(data["value"], errors="coerce")
    data["index_code"] = data["index_code"].astype("string").str.strip().str.upper()
    data = data.dropna().sort_values(["index_code", "date"])
    if data.duplicated(["date", "index_code"]).any():
        raise ValueError("Variety data contains duplicate date-index_code observations.")
    if data["value"].le(0).any():
        raise ValueError("Variety index levels must be positive before taking logarithms.")
    data["return"] = data.groupby("index_code", observed=True)["value"].transform(lambda values: np.log(values).diff())
    return data.pivot(index="date", columns="index_code", values="return").sort_index()


def compute_sector_returns(returns: pd.DataFrame, sector_mapping: pd.DataFrame) -> pd.DataFrame:
    """Aggregate mapped commodity returns into equally weighted sector portfolios."""
    required = {"index_code", "sector"}
    if missing := required.difference(sector_mapping.columns):
        raise ValueError(f"Sector mapping is missing columns: {sorted(missing)}")
    data = returns.copy()
    data.index = pd.to_datetime(data.index)
    data.columns = data.columns.astype(str).str.strip().str.upper()
    mapping = sector_mapping[["index_code", "sector"]].copy()
    mapping["index_code"] = mapping["index_code"].astype(str).str.strip().str.upper()
    mapping = mapping.drop_duplicates("index_code").set_index("index_code")["sector"]
    mapped_codes = data.columns.intersection(mapping.index)
    if mapped_codes.empty:
        raise ValueError("No return columns match the sector mapping.")
    return data.loc[:, mapped_codes].T.groupby(mapping.loc[mapped_codes], sort=True).mean().T.sort_index()


def _compute_ramp_in_sector_returns(
    returns: pd.DataFrame,
    sector_mapping: pd.DataFrame,
    variety_weights_by_month: Mapping[pd.Timestamp, pd.Series],
) -> pd.DataFrame:
    data = returns.copy()
    data.index = pd.to_datetime(data.index)
    data.columns = data.columns.astype(str).str.strip().str.upper()
    mapping = sector_mapping[["index_code", "sector"]].copy()
    mapping["index_code"] = mapping["index_code"].astype(str).str.strip().str.upper()
    mapping = mapping.drop_duplicates("index_code").set_index("index_code")["sector"]
    mapped_codes = data.columns.intersection(mapping.index)
    if mapped_codes.empty:
        raise ValueError("No return columns match the sector mapping.")
    data = data.loc[:, mapped_codes]
    records = []
    for month, month_data in data.groupby(data.index.to_period("M"), sort=True):
        month_end = month.to_timestamp("M")
        weights = variety_weights_by_month.get(month_end, pd.Series(dtype="float64"))
        weights = weights.reindex(mapped_codes).dropna()
        weighted_data = month_data.loc[:, month_data.columns.intersection(weights.index)]
        if weighted_data.empty:
            sector_names = mapping.reindex(mapped_codes).dropna().unique()
            records.append(pd.DataFrame(np.nan, index=month_data.index, columns=sector_names))
            continue
        weighted_returns = pd.DataFrame(index=month_data.index)
        for sector in mapping.reindex(weighted_data.columns).dropna().unique():
            members = mapping.index[mapping.eq(sector)].intersection(weighted_data.columns)
            values = weighted_data.loc[:, members]
            member_weights = weights.reindex(members)
            numerator = values.mul(member_weights, axis=1).sum(axis=1, min_count=1)
            denominator = values.notna().mul(member_weights, axis=1).sum(axis=1)
            weighted_returns[sector] = numerator.div(denominator.where(denominator.gt(0)))
        records.append(weighted_returns)
    if not records:
        return data.iloc[:, 0:0]
    return pd.concat(records).sort_index()


def _pair_weights(
    pairs: pd.DataFrame,
    member_weights: pd.Series,
) -> pd.Series:
    pair_columns = (
        ("index_code_1", "index_code_2")
        if {"index_code_1", "index_code_2"}.issubset(pairs.columns)
        else ("sector_1", "sector_2")
    )
    if pairs.empty:
        return pd.Series(
            dtype="float64",
            index=pd.MultiIndex.from_arrays([[], []], names=list(pair_columns)),
        )
    left = pairs[pair_columns[0]].astype(str)
    right = pairs[pair_columns[1]].astype(str)
    index = pd.MultiIndex.from_arrays([left, right], names=list(pair_columns))
    values = member_weights.reindex(left).to_numpy() * member_weights.reindex(right).to_numpy()
    return pd.Series(values, index=index, dtype="float64")


def _apply_member_weights(
    raw_result: MonthlyEnvironmentResult,
    member_weights_by_month: Mapping[pd.Timestamp, pd.Series],
    ramp_in_months: int,
) -> tuple[MonthlyEnvironmentResult, dict[pd.Timestamp, pd.Series]]:
    metrics = raw_result.metrics.copy()
    pair_weights_by_month: dict[pd.Timestamp, pd.Series] = {}
    for date in metrics.index:
        volatility = raw_result.volatility_by_month.get(date, pd.Series(dtype="float64"))
        weights = member_weights_by_month.get(date, pd.Series(dtype="float64"))
        valid_weights = weights.reindex(volatility.index).dropna()
        total_weight = valid_weights.sum()
        if total_weight > 0:
            metrics.at[date, "volatility"] = float(
                volatility.reindex(valid_weights.index).mul(valid_weights).sum() / total_weight
            )
        else:
            metrics.at[date, "volatility"] = np.nan
        pairs = raw_result.correlation_by_month.get(date, pd.DataFrame())
        pair_weights = _pair_weights(pairs, weights)
        pair_weights_by_month[date] = pair_weights
        total_pair_weight = pair_weights.sum()
        if total_pair_weight > 0 and not pairs.empty:
            pair_values = pd.to_numeric(pairs["abs_correlation"], errors="coerce").to_numpy(dtype=float)
            pair_weight_values = pair_weights.to_numpy(dtype=float)
            metrics.at[date, "correlation"] = float(
                np.nansum(pair_values * pair_weight_values) / total_pair_weight
            )
            for scope, column in (("within", "within_sector_correlation"), ("across", "across_sector_correlation")):
                scoped = pairs["pair_scope"].eq(scope).to_numpy() & ~np.isnan(pair_values)
                scope_weight = pair_weight_values[scoped].sum()
                metrics.at[date, column] = (
                    float(np.dot(pair_values[scoped], pair_weight_values[scoped]) / scope_weight)
                    if scope_weight > 0
                    else np.nan
                )
        else:
            metrics.at[date, "correlation"] = np.nan
            metrics.at[date, "within_sector_correlation"] = np.nan
            metrics.at[date, "across_sector_correlation"] = np.nan
        metrics.at[date, "valid_member_count"] = int(len(valid_weights))
        metrics.at[date, "valid_pair_count"] = int(len(pairs))
        metrics.at[date, "total_member_weight"] = total_weight
        metrics.at[date, "ramping_member_count"] = int(valid_weights.lt(1).sum())
        metrics.at[date, "new_member_count"] = int(valid_weights.eq(1 / ramp_in_months).sum())
        metrics.at[date, "total_pair_weight"] = total_pair_weight
    return (
        MonthlyEnvironmentResult(
            metrics,
            raw_result.volatility_by_month,
            raw_result.correlation_by_month,
            raw_result.exclusions_by_month,
            raw_result.sector_counts_by_month,
        ),
        pair_weights_by_month,
    )


def _pairwise_details(
    window: pd.DataFrame,
    valid_codes: list[str],
    min_pair_observations: int,
    sectors: dict[str, str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    correlations = window[valid_codes].corr(min_periods=min_pair_observations)
    counts = window[valid_codes].notna().astype("int64").T.dot(window[valid_codes].notna().astype("int64"))
    rows, columns = np.triu_indices_from(correlations, k=1)
    pairs = pd.DataFrame(
        {
            "index_code_1": correlations.index.take(rows),
            "index_code_2": correlations.columns.take(columns),
            "correlation": correlations.to_numpy()[rows, columns],
        }
    ).dropna(subset=["correlation"])
    if pairs.empty:
        pairs = pairs.assign(
            n_observations=pd.Series(dtype="int64"),
            abs_correlation=pd.Series(dtype="float64"),
            sector_1=pd.Series(dtype="string"),
            sector_2=pd.Series(dtype="string"),
            pair_scope=pd.Series(dtype="string"),
        )
        return correlations, pairs
    pairs["n_observations"] = [
        int(counts.loc[left, right]) for left, right in pairs[["index_code_1", "index_code_2"]].itertuples(index=False)
    ]
    pairs["abs_correlation"] = pairs["correlation"].abs()
    pairs["sector_1"] = pairs["index_code_1"].map(sectors)
    pairs["sector_2"] = pairs["index_code_2"].map(sectors)
    pairs["pair_scope"] = np.where(
        pairs["sector_1"].notna() & pairs["sector_1"].eq(pairs["sector_2"]),
        "within",
        np.where(pairs["sector_1"].notna() & pairs["sector_2"].notna(), "across", "unmapped"),
    )
    return correlations, pairs


def _complete_month_index(index: pd.DatetimeIndex) -> pd.DatetimeIndex:
    first_month_end = index.min().to_period("M").end_time.normalize()
    last_observation = index.max().normalize()
    last_month_end = last_observation.to_period("M").end_time.normalize()
    if last_observation < last_observation + pd.offsets.BMonthEnd(0):
        last_month_end -= pd.offsets.MonthEnd()
    return pd.date_range(first_month_end, last_month_end, freq="ME")


def compute_monthly_environment(
    returns: pd.DataFrame,
    sector_mapping: pd.DataFrame,
    lookback_months: int = 6,
    min_observations: int = 100,
    min_pair_observations: int = 100,
    annualization_days: int = 252,
) -> MonthlyEnvironmentResult:
    """Compute product-equal monthly volatility and pair-equal absolute correlation."""
    if returns.empty:
        raise ValueError("Daily return matrix is empty.")
    returns = returns.copy()
    returns.index = pd.to_datetime(returns.index)
    returns = returns.sort_index()
    returns.columns = returns.columns.astype(str)
    mapping = sector_mapping.copy()
    if missing := {"index_code", "sector"}.difference(mapping.columns):
        raise ValueError(f"Sector mapping is missing columns: {sorted(missing)}")
    mapping["index_code"] = mapping["index_code"].astype(str).str.upper()
    sectors = mapping.drop_duplicates("index_code").set_index("index_code")["sector"].to_dict()
    month_index = _complete_month_index(returns.index)
    records: list[dict[str, object]] = []
    volatility_by_month: dict[pd.Timestamp, pd.Series] = {}
    correlation_by_month: dict[pd.Timestamp, pd.DataFrame] = {}
    exclusions_by_month: dict[pd.Timestamp, dict[str, str]] = {}
    sector_counts_by_month: dict[pd.Timestamp, pd.Series] = {}
    for month_end in month_index:
        available_dates = returns.index[returns.index <= month_end]
        observation_end = available_dates.max() if len(available_dates) else pd.NaT
        window_start = month_end - pd.DateOffset(months=lookback_months)
        window = returns.loc[(returns.index > window_start) & (returns.index <= month_end)]
        observations = window.notna().sum()
        valid_codes = observations.index[observations.ge(min_observations)].tolist()
        excluded = {
            code: f"{int(count)} observations < {min_observations}"
            for code, count in observations.loc[observations.lt(min_observations)].items()
        }
        volatility = window[valid_codes].std(ddof=1) * np.sqrt(annualization_days)
        volatility = volatility.dropna()
        valid_codes = volatility.index.tolist()
        correlations, pairs = _pairwise_details(window, valid_codes, min_pair_observations, sectors)
        within = pairs.loc[pairs["pair_scope"].eq("within"), "abs_correlation"]
        across = pairs.loc[pairs["pair_scope"].eq("across"), "abs_correlation"]
        records.append(
            {
                "date": month_end,
                "observation_date": observation_end,
                "volatility": volatility.mean() if not volatility.empty else np.nan,
                "correlation": pairs["abs_correlation"].mean() if not pairs.empty else np.nan,
                "within_sector_correlation": within.mean() if not within.empty else np.nan,
                "across_sector_correlation": across.mean() if not across.empty else np.nan,
                "n_varieties": len(valid_codes),
                "n_pairs": len(pairs),
                "valid_codes": tuple(valid_codes),
            }
        )
        volatility_by_month[month_end] = volatility
        correlation_by_month[month_end] = pairs
        exclusions_by_month[month_end] = excluded
        sector_counts_by_month[month_end] = pd.Series(valid_codes).map(sectors).value_counts()
    metrics = pd.DataFrame(records).set_index("date")
    return MonthlyEnvironmentResult(
        metrics,
        volatility_by_month,
        correlation_by_month,
        exclusions_by_month,
        sector_counts_by_month,
    )


def compute_monthly_sector_environment(
    returns: pd.DataFrame,
    sector_mapping: pd.DataFrame,
    lookback_months: int = 6,
    min_observations: int = 100,
    min_pair_observations: int = 100,
    annualization_days: int = 252,
) -> MonthlyEnvironmentResult:
    """Compute sector-equal volatility and sector-pair-equal absolute correlation."""
    sector_returns = compute_sector_returns(returns, sector_mapping)
    mapping = sector_mapping[["index_code", "sector"]].copy()
    mapping["index_code"] = mapping["index_code"].astype(str).str.strip().str.upper()
    mapped_codes = set(mapping["index_code"])
    sector_by_code = mapping.drop_duplicates("index_code").set_index("index_code")["sector"]
    unmapped = sorted(set(returns.columns.astype(str).str.upper()).difference(mapped_codes))
    source_counts = sector_by_code.reindex(returns.columns.astype(str).str.upper()).dropna().value_counts()
    month_index = _complete_month_index(sector_returns.index)
    records: list[dict[str, object]] = []
    volatility_by_month: dict[pd.Timestamp, pd.Series] = {}
    correlation_by_month: dict[pd.Timestamp, pd.DataFrame] = {}
    exclusions_by_month: dict[pd.Timestamp, dict[str, str]] = {}
    sector_counts_by_month: dict[pd.Timestamp, pd.Series] = {}
    for month_end in month_index:
        window_start = month_end - pd.DateOffset(months=lookback_months)
        window = sector_returns.loc[(sector_returns.index > window_start) & (sector_returns.index <= month_end)]
        observations = window.notna().sum()
        valid_sectors = observations.index[observations.ge(min_observations)].tolist()
        excluded = {
            sector: f"{int(count)} observations < {min_observations}"
            for sector, count in observations.loc[observations.lt(min_observations)].items()
        }
        if unmapped:
            excluded.update({f"unmapped:{code}": "not in sector mapping" for code in unmapped})
        volatility = window[valid_sectors].std(ddof=1) * np.sqrt(annualization_days)
        volatility = volatility.dropna()
        valid_sectors = volatility.index.tolist()
        correlations = window[valid_sectors].corr(min_periods=min_pair_observations)
        counts = window[valid_sectors].notna().astype("int64").T.dot(window[valid_sectors].notna().astype("int64"))
        rows, columns = np.triu_indices_from(correlations, k=1)
        pairs = pd.DataFrame(
            {
                "sector_1": correlations.index.take(rows),
                "sector_2": correlations.columns.take(columns),
                "correlation": correlations.to_numpy()[rows, columns],
            }
        ).dropna(subset=["correlation"])
        if not pairs.empty:
            pairs["n_observations"] = [
                int(counts.loc[left, right]) for left, right in pairs[["sector_1", "sector_2"]].itertuples(index=False)
            ]
            pairs["abs_correlation"] = pairs["correlation"].abs()
        else:
            pairs = pairs.assign(
                n_observations=pd.Series(dtype="int64"),
                abs_correlation=pd.Series(dtype="float64"),
            )
        records.append(
            {
                "date": month_end,
                "observation_date": sector_returns.index[sector_returns.index <= month_end].max(),
                "volatility": volatility.mean() if not volatility.empty else np.nan,
                "correlation": pairs["abs_correlation"].mean() if not pairs.empty else np.nan,
                "n_sectors": len(valid_sectors),
                "n_pairs": len(pairs),
                "valid_sectors": tuple(valid_sectors),
            }
        )
        volatility_by_month[month_end] = volatility
        correlation_by_month[month_end] = pairs
        exclusions_by_month[month_end] = excluded
        sector_counts_by_month[month_end] = source_counts.reindex(valid_sectors).fillna(0).astype("int64")
    return MonthlyEnvironmentResult(
        pd.DataFrame(records).set_index("date"),
        volatility_by_month,
        correlation_by_month,
        exclusions_by_month,
        sector_counts_by_month,
    )


def _compute_monthly_sector_environment_from_returns(
    sector_returns: pd.DataFrame,
    source_counts: pd.Series,
    unmapped: Iterable[str],
    lookback_months: int,
    min_observations: int,
    min_pair_observations: int,
    annualization_days: int,
) -> MonthlyEnvironmentResult:
    """Compute sector metrics from an already weighted daily sector-return matrix."""
    if sector_returns.empty:
        raise ValueError("Sector return matrix is empty.")
    sector_returns = sector_returns.copy().sort_index()
    sector_returns.columns = sector_returns.columns.astype(str)
    month_index = _complete_month_index(sector_returns.index)
    records: list[dict[str, object]] = []
    volatility_by_month: dict[pd.Timestamp, pd.Series] = {}
    correlation_by_month: dict[pd.Timestamp, pd.DataFrame] = {}
    exclusions_by_month: dict[pd.Timestamp, dict[str, str]] = {}
    sector_counts_by_month: dict[pd.Timestamp, pd.Series] = {}
    for month_end in month_index:
        window_start = month_end - pd.DateOffset(months=lookback_months)
        window = sector_returns.loc[(sector_returns.index > window_start) & (sector_returns.index <= month_end)]
        observations = window.notna().sum()
        valid_sectors = observations.index[observations.ge(min_observations)].tolist()
        excluded = {
            sector: f"{int(count)} observations < {min_observations}"
            for sector, count in observations.loc[observations.lt(min_observations)].items()
        }
        excluded.update({f"unmapped:{code}": "not in sector mapping" for code in unmapped})
        volatility = (window[valid_sectors].std(ddof=1) * np.sqrt(annualization_days)).dropna()
        valid_sectors = volatility.index.tolist()
        correlations = window[valid_sectors].corr(min_periods=min_pair_observations)
        counts = window[valid_sectors].notna().astype("int64").T.dot(window[valid_sectors].notna().astype("int64"))
        rows, columns = np.triu_indices_from(correlations, k=1)
        pairs = pd.DataFrame(
            {
                "sector_1": correlations.index.take(rows),
                "sector_2": correlations.columns.take(columns),
                "correlation": correlations.to_numpy()[rows, columns],
            }
        ).dropna(subset=["correlation"])
        if not pairs.empty:
            pairs["n_observations"] = [
                int(counts.loc[left, right])
                for left, right in pairs[["sector_1", "sector_2"]].itertuples(index=False)
            ]
            pairs["abs_correlation"] = pairs["correlation"].abs()
            pairs["pair_scope"] = "across"
        else:
            pairs = pairs.assign(
                n_observations=pd.Series(dtype="int64"),
                abs_correlation=pd.Series(dtype="float64"),
                pair_scope=pd.Series(dtype="string"),
            )
        records.append(
            {
                "date": month_end,
                "observation_date": sector_returns.index[sector_returns.index <= month_end].max(),
                "volatility": volatility.mean() if not volatility.empty else np.nan,
                "correlation": pairs["abs_correlation"].mean() if not pairs.empty else np.nan,
                "within_sector_correlation": np.nan,
                "across_sector_correlation": pairs["abs_correlation"].mean() if not pairs.empty else np.nan,
                "n_sectors": len(valid_sectors),
                "n_pairs": len(pairs),
                "valid_sectors": tuple(valid_sectors),
            }
        )
        volatility_by_month[month_end] = volatility
        correlation_by_month[month_end] = pairs
        exclusions_by_month[month_end] = excluded
        sector_counts_by_month[month_end] = source_counts.reindex(valid_sectors).fillna(0).astype("int64")
    return MonthlyEnvironmentResult(
        pd.DataFrame(records).set_index("date"),
        volatility_by_month,
        correlation_by_month,
        exclusions_by_month,
        sector_counts_by_month,
    )


def compute_monthly_environment_by_aggregation(
    returns: pd.DataFrame,
    sector_mapping: pd.DataFrame,
    aggregation_method: str,
    lookback_months: int = 6,
    min_observations: int = 100,
    min_pair_observations: int = 100,
    annualization_days: int = 252,
) -> MonthlyEnvironmentResult:
    """Dispatch monthly environment calculation by aggregation method."""
    if aggregation_method == "sector_equal":
        return compute_monthly_sector_environment(
            returns,
            sector_mapping,
            lookback_months,
            min_observations,
            min_pair_observations,
            annualization_days,
        )
    if aggregation_method == "variety_equal":
        return compute_monthly_environment(
            returns,
            sector_mapping,
            lookback_months,
            min_observations,
            min_pair_observations,
            annualization_days,
        )
    raise ValueError(f"Unsupported aggregation method: {aggregation_method!r}")


def compute_monthly_ramp_in_environment_by_aggregation(
    returns: pd.DataFrame,
    sector_mapping: pd.DataFrame,
    aggregation_method: str,
    lookback_months: int = 6,
    min_observations: int = 100,
    min_pair_observations: int = 100,
    annualization_days: int = 252,
    ramp_in_months: int = 6,
) -> RampInEnvironmentResult:
    """Compute raw and six-month linear-ramp environment results together."""
    if not isinstance(ramp_in_months, int) or ramp_in_months <= 0:
        raise ValueError("ramp_in_months must be a positive integer.")
    raw_result = compute_monthly_environment_by_aggregation(
        returns,
        sector_mapping,
        aggregation_method,
        lookback_months,
        min_observations,
        min_pair_observations,
        annualization_days,
    )
    if aggregation_method == "variety_equal":
        variety_weights = compute_linear_ramp_weights(
            {date: result.index for date, result in raw_result.volatility_by_month.items()},
            ramp_in_months,
        )
        ramp_result, pair_weights = _apply_member_weights(raw_result, variety_weights, ramp_in_months)
        return RampInEnvironmentResult(raw_result, ramp_result, variety_weights, variety_weights, pair_weights)
    if aggregation_method != "sector_equal":
        raise ValueError(f"Unsupported aggregation method: {aggregation_method!r}")

    mapping = sector_mapping[["index_code", "sector"]].copy()
    mapping["index_code"] = mapping["index_code"].astype(str).str.strip().str.upper()
    mapping = mapping.drop_duplicates("index_code")
    returns_data = returns.copy()
    returns_data.columns = returns_data.columns.astype(str).str.strip().str.upper()
    mapped_codes = returns_data.columns.intersection(mapping["index_code"])
    mapped_returns = returns_data.loc[:, mapped_codes]
    source_result = compute_monthly_environment(
        mapped_returns,
        mapping,
        lookback_months,
        min_observations,
        min_pair_observations,
        annualization_days,
    )
    variety_weights = compute_linear_ramp_weights(
        {date: result.index for date, result in source_result.volatility_by_month.items()},
        ramp_in_months,
    )
    ramp_sector_returns = _compute_ramp_in_sector_returns(returns_data, mapping, variety_weights)
    source_counts = mapping["sector"].value_counts()
    unmapped = sorted(set(returns_data.columns).difference(set(mapping["index_code"])))
    ramp_sector_raw = _compute_monthly_sector_environment_from_returns(
        ramp_sector_returns,
        source_counts,
        unmapped,
        lookback_months,
        min_observations,
        min_pair_observations,
        annualization_days,
    )
    member_weights = compute_linear_ramp_weights(
        {date: result.index for date, result in ramp_sector_raw.volatility_by_month.items()},
        ramp_in_months,
    )
    ramp_result, pair_weights = _apply_member_weights(ramp_sector_raw, member_weights, ramp_in_months)
    return RampInEnvironmentResult(raw_result, ramp_result, variety_weights, member_weights, pair_weights)
