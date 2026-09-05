from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .config import load_config
from .environment import (
    MonthlyEnvironmentResult,
    RampInEnvironmentResult,
    compute_log_returns,
    compute_monthly_ramp_in_environment_by_aggregation,
    compute_sector_returns,
)
from .histories import normalize_nhci_history, read_history_file
from .regimes import (
    add_rolling_thresholds,
    classify_monthly_regimes,
    compress_regime_stages,
    map_monthly_regimes_to_daily,
    validate_target_regime,
)


@dataclass(frozen=True)
class SectorResearchContext:
    """Shared data and state objects used by the formal analysis notebooks."""

    config: dict[str, Any]
    project_root: Path
    raw_dir: Path
    future_instruments: pd.DataFrame
    dominant_prices: pd.DataFrame
    sector_mapping: pd.DataFrame
    daily_returns: pd.DataFrame
    unramped_sector_returns: pd.DataFrame
    ramp_in_result: RampInEnvironmentResult
    environment_result: MonthlyEnvironmentResult
    monthly_environment: pd.DataFrame
    monthly_regimes: pd.DataFrame
    nhci: pd.Series
    daily_regimes: pd.DataFrame
    stages: pd.DataFrame
    coverage: pd.DataFrame
    target_audit: pd.Series
    unmapped_symbols: tuple[str, ...]


@dataclass(frozen=True)
class BreakpointScenarioResult:
    """Raw/ramp diagnostics and regimes for one comparison scenario."""

    aggregation_method: str
    environment_lookback_months: int
    regime_threshold_lookback_months: int
    ramp_in_result: RampInEnvironmentResult
    diagnostics: pd.DataFrame
    monthly_regimes: pd.DataFrame
    daily_regimes: pd.DataFrame
    stages: pd.DataFrame

    @property
    def lookback_months(self) -> int:
        """Backward-compatible alias for the fixed environment lookback."""
        return self.environment_lookback_months

    @property
    def environment_result(self) -> MonthlyEnvironmentResult:
        """Return the ramp-in result for compatibility with existing notebook code."""
        return self.ramp_in_result.ramp_result

    @property
    def raw_environment_result(self) -> MonthlyEnvironmentResult:
        """Return the unweighted result used as the breakpoint diagnostic baseline."""
        return self.ramp_in_result.raw_result


def _resolve_project_root(config_path: Path, project_root: str | Path | None) -> Path:
    if project_root:
        return Path(project_root).resolve()
    config_dir = config_path.resolve().parent
    return config_dir.parent if config_dir.name == "config" else config_dir


def build_sector_research_context(
    config_path: str | Path = "config/config.yaml",
    project_root: str | Path | None = None,
) -> SectorResearchContext:
    """Build the complete sector-equal market context used by both notebooks."""
    config_path = Path(config_path).resolve()
    project_root = _resolve_project_root(config_path, project_root)
    config = load_config(config_path)
    raw_dir = project_root / "data"
    paths = config["paths"]
    future_path = project_root / paths["future_instruments"]
    dominant_path = project_root / paths["dominant_prices"]
    for path in (future_path, dominant_path):
        if not path.exists():
            raise FileNotFoundError(f"Missing raw input: {path}")
    future_instruments = pd.read_parquet(future_path)
    dominant_prices = pd.read_parquet(dominant_path)
    dominant_prices["date"] = pd.to_datetime(dominant_prices["date"], errors="coerce")
    dominant_prices = dominant_prices.dropna(subset=["date"]).sort_values("date")
    dominant_prices = dominant_prices.loc[dominant_prices["date"].ge(pd.Timestamp(config["start_date"]))]
    if config.get("end_date"):
        dominant_prices = dominant_prices.loc[dominant_prices["date"].le(pd.Timestamp(config["end_date"]))]
    coverage = (
        dominant_prices.groupby("underlying_symbol", as_index=False)
        .agg(start_date=("date", "min"), end_date=("date", "max"), rows=("date", "size"))
        .sort_values("start_date")
    )
    nhci_config = paths.get("external_nhci_file")
    nhci_path = project_root / nhci_config if nhci_config else project_root / paths["nhci"]
    if not nhci_path.exists():
        raise FileNotFoundError(f"Missing NHCI input: {nhci_path}")
    nhci_data = normalize_nhci_history(
        read_history_file(nhci_path), config["start_date"], config.get("end_date")
    )
    mapping_path = project_root / "config" / "commodity_sectors.csv"
    sector_mapping = pd.read_csv(mapping_path)
    daily_returns = compute_log_returns(dominant_prices)
    ramp_in_result = compute_monthly_ramp_in_environment_by_aggregation(
        daily_returns,
        sector_mapping,
        config["aggregation_method"],
        config["environment_lookback_months"],
        config["min_observations"],
        config["min_pair_observations"],
        config["annualization_days"],
        config["ramp_in_months"],
    )
    environment_result = ramp_in_result.ramp_result
    monthly_environment = environment_result.metrics
    monthly_regimes = classify_monthly_regimes(
        add_rolling_thresholds(
            monthly_environment,
            config["regime_threshold_lookback_months"],
            config["regime_threshold_min_periods"],
        )
    )
    nhci = nhci_data.set_index("date")["close"].sort_index()
    daily_regimes = map_monthly_regimes_to_daily(monthly_regimes, nhci.index)
    target_audit = validate_target_regime(config["validation_target_date"], daily_regimes, monthly_regimes)
    stages = compress_regime_stages(daily_regimes, nhci, monthly_regimes)
    mapped_symbols = set(sector_mapping["index_code"].astype(str).str.strip().str.upper())
    unmapped_symbols = tuple(sorted(set(daily_returns.columns).difference(mapped_symbols)))
    return SectorResearchContext(
        config=config,
        project_root=project_root,
        raw_dir=raw_dir,
        future_instruments=future_instruments,
        dominant_prices=dominant_prices,
        sector_mapping=sector_mapping,
        daily_returns=daily_returns,
        unramped_sector_returns=compute_sector_returns(daily_returns, sector_mapping),
        ramp_in_result=ramp_in_result,
        environment_result=environment_result,
        monthly_environment=monthly_environment,
        monthly_regimes=monthly_regimes,
        nhci=nhci,
        daily_regimes=daily_regimes,
        stages=stages,
        coverage=coverage,
        target_audit=target_audit,
        unmapped_symbols=unmapped_symbols,
    )


def _build_breakpoint_scenario(
    context: SectorResearchContext,
    aggregation_method: str,
    environment_months: int,
    threshold_months: int,
    ramp_in_result: RampInEnvironmentResult,
) -> BreakpointScenarioResult:
    raw_result = ramp_in_result.raw_result
    ramp_result = ramp_in_result.ramp_result
    raw_metrics = raw_result.metrics[["volatility", "correlation"]].rename(
        columns={"volatility": "raw_volatility", "correlation": "raw_correlation"}
    )
    ramp_metrics = ramp_result.metrics.drop(columns=["volatility", "correlation"], errors="ignore").copy()
    ramp_metrics["ramp_volatility"] = ramp_result.metrics["volatility"]
    ramp_metrics["ramp_correlation"] = ramp_result.metrics["correlation"]
    diagnostics = raw_metrics.join(ramp_metrics)
    diagnostics["raw_volatility_change"] = diagnostics["raw_volatility"].diff()
    diagnostics["ramp_volatility_change"] = diagnostics["ramp_volatility"].diff()
    diagnostics["raw_correlation_change"] = diagnostics["raw_correlation"].diff()
    diagnostics["ramp_correlation_change"] = diagnostics["ramp_correlation"].diff()
    diagnostics["volatility_level_gap"] = diagnostics["raw_volatility"] - diagnostics["ramp_volatility"]
    diagnostics["correlation_level_gap"] = diagnostics["raw_correlation"] - diagnostics["ramp_correlation"]
    member_sets = {
        date: tuple(raw_result.volatility_by_month.get(date, pd.Series(dtype="float64")).dropna().index.astype(str))
        for date in raw_result.metrics.index
    }
    pair_sets = {
        date: tuple(
            zip(
                pairs.get("sector_1", pairs.get("index_code_1", pd.Series(dtype="string"))).astype(str),
                pairs.get("sector_2", pairs.get("index_code_2", pd.Series(dtype="string"))).astype(str),
                strict=True,
            )
        )
        for date, pairs in raw_result.correlation_by_month.items()
    }
    dates = list(raw_result.metrics.index)
    diagnostics["member_set_changed"] = pd.Series(
        [
            position == 0 or member_sets[date] != member_sets[dates[position - 1]]
            for position, date in enumerate(dates)
        ],
        index=dates,
    )
    diagnostics["pair_set_changed"] = pd.Series(
        [
            position == 0 or pair_sets.get(date, ()) != pair_sets.get(dates[position - 1], ())
            for position, date in enumerate(dates)
        ],
        index=dates,
    )
    ramp_environment = ramp_metrics[["ramp_volatility", "ramp_correlation"]].rename(
        columns={"ramp_volatility": "volatility", "ramp_correlation": "correlation"}
    )
    monthly_regimes = classify_monthly_regimes(
        add_rolling_thresholds(ramp_environment, threshold_months, threshold_months)
    )
    daily_regimes = map_monthly_regimes_to_daily(monthly_regimes, context.nhci.index)
    stages = compress_regime_stages(daily_regimes, context.nhci, monthly_regimes)
    scenario_diagnostics = diagnostics.join(
        monthly_regimes[["regime", "volatility_threshold", "correlation_threshold"]].rename(
            columns={
                "regime": "ramp_regime",
                "volatility_threshold": "ramp_volatility_threshold",
                "correlation_threshold": "ramp_correlation_threshold",
            }
        )
    )
    return BreakpointScenarioResult(
        aggregation_method=aggregation_method,
        environment_lookback_months=environment_months,
        regime_threshold_lookback_months=threshold_months,
        ramp_in_result=ramp_in_result,
        diagnostics=scenario_diagnostics,
        monthly_regimes=monthly_regimes,
        daily_regimes=daily_regimes,
        stages=stages,
    )


def build_breakpoint_scenarios(context: SectorResearchContext) -> dict[tuple[str, int], BreakpointScenarioResult]:
    """Build fixed-environment scenarios across configured regime-threshold lookbacks."""
    settings = context.config.get("breakpoint_analysis")
    if not isinstance(settings, Mapping):
        raise ValueError("Configuration is missing the breakpoint_analysis mapping.")
    methods = settings.get("aggregation_methods")
    environment_months = settings.get("environment_lookback_months")
    min_observations = settings.get("min_observations")
    min_pair_observations = settings.get("min_pair_observations")
    ramp_in_months = settings.get("ramp_in_months")
    threshold_months = settings.get("regime_threshold_lookback_months")
    allowed_methods = {"sector_equal", "variety_equal"}
    if not isinstance(methods, list) or not methods or any(
        not isinstance(method, str) or method not in allowed_methods for method in methods
    ):
        raise ValueError(f"breakpoint_analysis.aggregation_methods must use {sorted(allowed_methods)}.")
    if len(set(methods)) != len(methods):
        raise ValueError("breakpoint_analysis.aggregation_methods must not contain duplicates.")
    if not isinstance(environment_months, int) or environment_months <= 0:
        raise ValueError("breakpoint_analysis.environment_lookback_months must be a positive integer.")
    if not all(isinstance(value, int) and value >= 0 for value in (min_observations, min_pair_observations)):
        raise ValueError("Breakpoint minimum observations must be non-negative integers.")
    if not isinstance(ramp_in_months, int) or ramp_in_months <= 0:
        raise ValueError("breakpoint_analysis.ramp_in_months must be a positive integer.")
    if not isinstance(threshold_months, list) or not threshold_months or any(
        not isinstance(value, int) or value <= 0 for value in threshold_months
    ):
        raise ValueError(
            "breakpoint_analysis.regime_threshold_lookback_months must be a non-empty list of positive integers."
        )
    if len(set(threshold_months)) != len(threshold_months):
        raise ValueError("breakpoint_analysis.regime_threshold_lookback_months must not contain duplicates.")

    ramp_results = {
        method: compute_monthly_ramp_in_environment_by_aggregation(
            context.daily_returns,
            context.sector_mapping,
            method,
            environment_months,
            min_observations,
            min_pair_observations,
            context.config["annualization_days"],
            ramp_in_months,
        )
        for method in methods
    }
    return {
        (method, threshold_window): _build_breakpoint_scenario(
            context,
            method,
            environment_months,
            threshold_window,
            ramp_results[method],
        )
        for method in methods
        for threshold_window in threshold_months
    }


def build_metric_lookback_scenarios(context: SectorResearchContext) -> dict[int, BreakpointScenarioResult]:
    """Build sector-equal scenarios across synchronized metric lookback windows."""
    settings = context.config.get("metric_lookback_analysis")
    if not isinstance(settings, Mapping):
        raise ValueError("Configuration is missing the metric_lookback_analysis mapping.")
    method = settings.get("aggregation_method")
    environment_months = settings.get("environment_lookback_months")
    threshold_months = settings.get("regime_threshold_lookback_months")
    min_observations = settings.get("min_observations")
    min_pair_observations = settings.get("min_pair_observations")
    ramp_in_months = settings.get("ramp_in_months")
    if method != "sector_equal":
        raise ValueError("metric_lookback_analysis.aggregation_method must be 'sector_equal'.")
    if not isinstance(environment_months, list) or not environment_months or any(
        not isinstance(value, int) or value <= 0 for value in environment_months
    ):
        raise ValueError(
            "metric_lookback_analysis.environment_lookback_months must be a non-empty list of positive integers."
        )
    if len(set(environment_months)) != len(environment_months):
        raise ValueError("metric_lookback_analysis.environment_lookback_months must not contain duplicates.")
    if not isinstance(threshold_months, int) or threshold_months <= 0:
        raise ValueError("metric_lookback_analysis.regime_threshold_lookback_months must be a positive integer.")
    if not all(isinstance(value, int) and value >= 0 for value in (min_observations, min_pair_observations)):
        raise ValueError("Metric lookback minimum observations must be non-negative integers.")
    if not isinstance(ramp_in_months, int) or ramp_in_months <= 0:
        raise ValueError("metric_lookback_analysis.ramp_in_months must be a positive integer.")
    results: dict[int, BreakpointScenarioResult] = {}
    for environment_month in environment_months:
        ramp_in_result = compute_monthly_ramp_in_environment_by_aggregation(
            context.daily_returns,
            context.sector_mapping,
            method,
            environment_month,
            min_observations,
            min_pair_observations,
            context.config["annualization_days"],
            ramp_in_months,
        )
        results[environment_month] = _build_breakpoint_scenario(
            context,
            method,
            environment_month,
            threshold_months,
            ramp_in_result,
        )
    return results


def compute_breakpoint_threshold_duration_summary(
    diagnostics_by_scenario: Mapping[tuple[str, int], pd.DataFrame],
) -> pd.DataFrame:
    """Summarize metric durations above and below each scenario threshold."""
    metric_specs = (
        ("volatility", "ramp_volatility", "ramp_volatility_threshold", "年化波动率"),
        ("correlation", "ramp_correlation", "ramp_correlation_threshold", "绝对相关性"),
    )
    columns = [
        "valid_months",
        "low_months",
        "high_months",
        "low_share",
        "high_share",
        "max_low_run_months",
        "max_high_run_months",
        "median_relative_gap",
    ]
    if not diagnostics_by_scenario:
        raise ValueError("Breakpoint scenario diagnostics are empty.")

    def longest_run(mask: pd.Series) -> int:
        if not mask.any():
            return 0
        groups = mask.ne(mask.shift()).cumsum()
        return int(mask.groupby(groups).sum().max())

    rows: list[dict[str, Any]] = []
    for (method, threshold_months), diagnostics in diagnostics_by_scenario.items():
        if diagnostics.empty:
            continue
        data = diagnostics.copy()
        data.index = pd.to_datetime(data.index).to_period("M").to_timestamp("M")
        data = data[~data.index.duplicated(keep="last")].sort_index()
        complete_index = pd.date_range(data.index.min(), data.index.max(), freq="ME")
        data = data.reindex(complete_index)
        for metric, value_column, threshold_column, metric_label in metric_specs:
            values = pd.to_numeric(data[value_column], errors="coerce")
            thresholds = pd.to_numeric(data[threshold_column], errors="coerce")
            valid = values.notna() & thresholds.notna()
            low = valid & values.le(thresholds)
            high = valid & values.gt(thresholds)
            valid_count = int(valid.sum())
            low_count = int(low.sum())
            high_count = int(high.sum())
            relative_gap = (values / thresholds - 1.0).where(valid & thresholds.ne(0))
            rows.append(
                {
                    "aggregation_method": method,
                    "regime_threshold_lookback_months": threshold_months,
                    "metric": metric,
                    "metric_label": metric_label,
                    "valid_months": valid_count,
                    "low_months": low_count,
                    "high_months": high_count,
                    "low_share": low_count / valid_count if valid_count else np.nan,
                    "high_share": high_count / valid_count if valid_count else np.nan,
                    "max_low_run_months": longest_run(low),
                    "max_high_run_months": longest_run(high),
                    "median_relative_gap": relative_gap.median(),
                }
            )
    if not rows:
        return pd.DataFrame(columns=columns + ["metric_label"]).set_index(
            ["aggregation_method", "regime_threshold_lookback_months", "metric"]
        )
    summary = pd.DataFrame(rows).set_index(
        ["aggregation_method", "regime_threshold_lookback_months", "metric"]
    )
    return summary[["metric_label", *columns]]
