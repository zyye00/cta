from __future__ import annotations

from collections.abc import Iterable, Mapping

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import PercentFormatter

from .regimes import FORMAL_REGIMES, INSUFFICIENT_HISTORY, REGIME_LABELS

REGIME_COLORS = {
    "LV_LC": "#4C78A8",
    "HV_LC": "#F58518",
    "LV_HC": "#54A24B",
    "HV_HC": "#E45756",
    INSUFFICIENT_HISTORY: "#D3D3D3",
}
METRIC_LABELS = {"volatility": "年化波动率", "correlation": "绝对相关性"}
AGGREGATION_LABELS = {"sector_equal": "板块等权", "variety_equal": "品种等权"}
AGGREGATION_COLORS = {"sector_equal": "#4C78A8", "variety_equal": "#E45756"}


def _new_axes(title: str, ylabel: str, figsize: tuple[float, float] = (12, 5)) -> tuple[Figure, Axes]:
    figure, axes = plt.subplots(figsize=figsize, constrained_layout=True)
    axes.set(title=title, xlabel="日期", ylabel=ylabel)
    axes.grid(alpha=0.25)
    return figure, axes


def _set_x_limits(axes: Axes, index: pd.Index) -> None:
    dates = pd.DatetimeIndex(index).dropna()
    if len(dates) > 1:
        axes.set_xlim(dates.min(), dates.max())


def plot_market_environment(
    environment: pd.DataFrame,
    metric: str,
    threshold: str,
    title: str,
    ylabel: str,
) -> Figure:
    """Plot one monthly environment metric and its lagged rolling median."""
    figure, axes = _new_axes(title, ylabel)
    axes.plot(environment.index, environment[metric], label=METRIC_LABELS.get(metric, metric), linewidth=1.4)
    axes.plot(
        environment.index,
        environment[threshold],
        label="滚动中位数",
        linestyle="--",
        linewidth=1.3,
    )
    valid_dates = environment.index[environment[metric].notna()]
    _set_x_limits(axes, valid_dates)
    axes.legend()
    return figure


def plot_breakpoint_metric_timeseries(
    diagnostics_by_scenario: Mapping[tuple[str, int], pd.DataFrame],
    metric: str,
) -> Figure:
    """Plot linear-ramp breakpoint metrics against thresholds for all scenarios."""
    if metric not in METRIC_LABELS:
        raise ValueError(f"Unsupported breakpoint metric: {metric}")
    if not diagnostics_by_scenario:
        raise ValueError("Breakpoint scenario diagnostics are empty.")
    column_specs = {
        "volatility": ("ramp_volatility", "ramp_volatility_threshold", "年化波动率"),
        "correlation": ("ramp_correlation", "ramp_correlation_threshold", "绝对相关性"),
    }
    metric_column, threshold_column, ylabel = column_specs[metric]
    methods = list(dict.fromkeys(method for method, _ in diagnostics_by_scenario))
    windows = list(dict.fromkeys(window for _, window in diagnostics_by_scenario))
    expected_keys = {(method, window) for method in methods for window in windows}
    if set(diagnostics_by_scenario) != expected_keys:
        raise ValueError("Breakpoint scenario diagnostics must contain every method-window combination.")
    figure, axes_grid = plt.subplots(
        len(methods),
        len(windows),
        figsize=(18, 8),
        sharex=True,
        sharey=True,
        squeeze=False,
        constrained_layout=True,
    )
    all_valid_dates: list[pd.Timestamp] = []
    for data in diagnostics_by_scenario.values():
        values = data.loc[data[metric_column].notna()].index
        all_valid_dates.extend(pd.to_datetime(values).tolist())
    if not all_valid_dates:
        raise ValueError(f"Breakpoint diagnostics contain no valid {metric} values.")
    for row, method in enumerate(methods):
        for column, window in enumerate(windows):
            axis = axes_grid[row, column]
            data = diagnostics_by_scenario[(method, window)].sort_index().copy()
            data.index = pd.to_datetime(data.index)
            values = pd.to_numeric(data[metric_column], errors="coerce")
            thresholds = pd.to_numeric(data[threshold_column], errors="coerce")
            valid = values.notna() & thresholds.notna()
            low = valid & values.le(thresholds)
            high = valid & values.gt(thresholds)
            x = mdates.date2num(data.index.to_pydatetime())
            value_array = values.to_numpy(dtype=float)
            threshold_array = thresholds.to_numpy(dtype=float)
            axis.fill_between(x, value_array, threshold_array, where=low.to_numpy(), color="#4C78A8", alpha=0.12)
            axis.fill_between(x, value_array, threshold_array, where=high.to_numpy(), color="#E45756", alpha=0.12)
            color = AGGREGATION_COLORS.get(method, f"C{row}")
            axis.plot(
                data.index,
                values,
                color=color,
                linewidth=1.25,
                label=f"{AGGREGATION_LABELS.get(method, method)}·线性渐进指标",
            )
            axis.plot(
                data.index,
                thresholds,
                color=color,
                linestyle="--",
                linewidth=1.1,
                label=f"{AGGREGATION_LABELS.get(method, method)}·滚动中位数阈值",
            )
            axis.set_title(f"中位数回看{_threshold_label(window)}")
            axis.set_ylabel(f"{AGGREGATION_LABELS.get(method, method)}\n{ylabel}" if column == 0 else "")
            axis.grid(alpha=0.25)
            if metric == "volatility":
                axis.yaxis.set_major_formatter(PercentFormatter(1.0))
    start, end = min(all_valid_dates), max(all_valid_dates)
    if start < end:
        for axis in axes_grid.flat:
            axis.set_xlim(start, end)
    for axis in axes_grid[-1]:
        axis.set_xlabel("日期")
    line_handles = [
        Line2D(
            [0],
            [0],
            color=AGGREGATION_COLORS.get(method, f"C{position}"),
            linewidth=1.3,
            linestyle=linestyle,
            label=f"{AGGREGATION_LABELS.get(method, method)}·{label}",
        )
        for position, method in enumerate(methods)
        for linestyle, label in (("-", "线性渐进指标"), ("--", "滚动中位数阈值"))
    ]
    fill_handles = [
        Patch(facecolor="#4C78A8", alpha=0.18, label="低于或等于阈值"),
        Patch(facecolor="#E45756", alpha=0.18, label="高于阈值"),
    ]
    figure.legend(
        handles=line_handles + fill_handles,
        loc="upper center",
        ncols=3,
        bbox_to_anchor=(0.5, 1.08),
        frameon=False,
    )
    figure.suptitle(f"断点影响：六个月线性渐进{ylabel}与滚动中位数阈值", y=1.16)
    return figure


def plot_metric_lookback_timeseries(
    diagnostics_by_window: Mapping[int, pd.DataFrame],
    metric: str,
) -> Figure:
    """Plot sector-equal ramp-in metrics and a fixed threshold across lookback windows."""
    if metric not in METRIC_LABELS:
        raise ValueError(f"Unsupported metric: {metric}")
    if not diagnostics_by_window:
        raise ValueError("Metric lookback diagnostics are empty.")
    column_specs = {
        "volatility": ("ramp_volatility", "ramp_volatility_threshold", "年化波动率"),
        "correlation": ("ramp_correlation", "ramp_correlation_threshold", "绝对相关性"),
    }
    metric_column, threshold_column, ylabel = column_specs[metric]
    windows = list(diagnostics_by_window)
    figure, axes_grid = plt.subplots(
        1,
        len(windows),
        figsize=(6.2 * len(windows), 5.5),
        sharex=True,
        sharey=True,
        squeeze=False,
        constrained_layout=True,
    )
    all_valid_dates: list[pd.Timestamp] = []
    for data in diagnostics_by_window.values():
        values = pd.to_numeric(data[metric_column], errors="coerce")
        all_valid_dates.extend(pd.to_datetime(data.index[values.notna()]).tolist())
    if not all_valid_dates:
        raise ValueError(f"Metric lookback diagnostics contain no valid {metric} values.")
    for column, window in enumerate(windows):
        axis = axes_grid[0, column]
        data = diagnostics_by_window[window].sort_index().copy()
        data.index = pd.to_datetime(data.index)
        values = pd.to_numeric(data[metric_column], errors="coerce")
        thresholds = pd.to_numeric(data[threshold_column], errors="coerce")
        valid = values.notna() & thresholds.notna()
        low = valid & values.le(thresholds)
        high = valid & values.gt(thresholds)
        x = mdates.date2num(data.index.to_pydatetime())
        value_array = values.to_numpy(dtype=float)
        threshold_array = thresholds.to_numpy(dtype=float)
        axis.fill_between(x, value_array, threshold_array, where=low.to_numpy(), color="#4C78A8", alpha=0.12)
        axis.fill_between(x, value_array, threshold_array, where=high.to_numpy(), color="#E45756", alpha=0.12)
        axis.plot(data.index, values, color="#4C78A8", linewidth=1.25, label="线性渐进指标")
        axis.plot(
            data.index,
            thresholds,
            color="#222222",
            linestyle="--",
            linewidth=1.1,
            label="36个月滚动中位数阈值",
        )
        axis.set_title(f"指标回看{window}个月")
        axis.set_ylabel(ylabel if column == 0 else "")
        axis.grid(alpha=0.25)
        if metric == "volatility":
            axis.yaxis.set_major_formatter(PercentFormatter(1.0))
    start, end = min(all_valid_dates), max(all_valid_dates)
    if start < end:
        for axis in axes_grid.flat:
            axis.set_xlim(start, end)
    for axis in axes_grid[0]:
        axis.set_xlabel("日期")
    line_handles = [
        Line2D([0], [0], color="#4C78A8", linewidth=1.3, label="线性渐进指标"),
        Line2D([0], [0], color="#222222", linewidth=1.3, linestyle="--", label="36个月滚动中位数阈值"),
    ]
    fill_handles = [
        Patch(facecolor="#4C78A8", alpha=0.18, label="低于或等于阈值"),
        Patch(facecolor="#E45756", alpha=0.18, label="高于阈值"),
    ]
    figure.legend(handles=line_handles + fill_handles, loc="upper center", ncols=4, bbox_to_anchor=(0.5, 1.04))
    figure.suptitle(f"板块等权{ylabel}：指标窗口敏感性", y=1.12)
    return figure


def plot_metric_lookback_regime_comparison(
    nhci: pd.Series,
    stages_by_window: Mapping[int, pd.DataFrame],
) -> Figure:
    """Plot NHCI and regime shading for each synchronized metric lookback window."""
    if nhci.empty or not stages_by_window:
        raise ValueError("NHCI or metric lookback stages are empty.")
    windows = list(stages_by_window)
    figure, axes_grid = plt.subplots(
        len(windows),
        1,
        figsize=(16, max(8, 2.5 * len(windows))),
        sharex=True,
        squeeze=False,
        constrained_layout=True,
    )
    axes = axes_grid[:, 0]
    for axis, window in zip(axes, windows, strict=True):
        stages = stages_by_window[window]
        valid_stages = stages.loc[stages["regime"].ne(INSUFFICIENT_HISTORY)]
        _shade_stages(axis, valid_stages, regimes=REGIME_COLORS.keys() - {INSUFFICIENT_HISTORY})
        axis.plot(nhci.index, nhci, color="#222222", linewidth=0.8)
        axis.set_ylabel(f"{window}个月", rotation=0, labelpad=28, va="center")
        axis.grid(alpha=0.2)
        first_valid = valid_stages["start_date"].min() if not valid_stages.empty else nhci.index.min()
        axis.set_xlim(max(nhci.index.min(), first_valid), nhci.index.max())
    handles = [
        Line2D([0], [0], color=REGIME_COLORS[regime], linewidth=8, alpha=0.35, label=REGIME_LABELS[regime])
        for regime in FORMAL_REGIMES
    ]
    figure.legend(handles=handles, loc="upper center", ncols=4, bbox_to_anchor=(0.5, 1.01))
    axes[-1].set_xlabel("日期")
    figure.suptitle("南华商品指数与指标窗口场景的线性渐进状态区间", y=1.04)
    return figure


def plot_metric_lookback_performance(performance: pd.DataFrame) -> Figure:
    """Plot NHCI annualized return and Sharpe heatmaps by metric lookback window."""
    required = {"annualized_return", "annualized_sharpe"}
    if performance.empty:
        raise ValueError("Metric lookback performance is empty.")
    if not required.issubset(performance.columns):
        missing = sorted(required.difference(performance.columns))
        raise ValueError(f"Metric lookback performance is missing columns: {missing}")
    if not {"environment_lookback_months", "regime"}.issubset(performance.index.names):
        raise ValueError("Metric lookback performance must use environment_lookback_months and regime index levels.")
    data = performance.reset_index()
    windows = list(dict.fromkeys(data["environment_lookback_months"].tolist()))
    figure, axes = plt.subplots(1, 2, figsize=(16, max(6, 0.65 * len(windows) + 3)), constrained_layout=True)
    specs = (("annualized_return", "年化收益", "RdYlGn", ".1%"), ("annualized_sharpe", "夏普比率", "PuBu", ".2f"))
    for axis, (column, title, colormap, number_format) in zip(axes, specs, strict=True):
        table = data.pivot(index="environment_lookback_months", columns="regime", values=column)
        table = table.reindex(index=windows, columns=FORMAL_REGIMES)
        image = axis.imshow(table.to_numpy(dtype=float), aspect="auto", cmap=colormap)
        axis.set(
            title=title,
            xticks=range(len(FORMAL_REGIMES)),
            xticklabels=[REGIME_LABELS[regime] for regime in FORMAL_REGIMES],
            yticks=range(len(windows)),
            yticklabels=[f"{window}个月" for window in windows],
        )
        axis.tick_params(axis="x", rotation=25)
        for row in range(table.shape[0]):
            for col in range(table.shape[1]):
                value = table.iloc[row, col]
                if pd.notna(value):
                    axis.text(col, row, format(value, number_format), ha="center", va="center", fontsize=9)
        figure.colorbar(image, ax=axis, shrink=0.8)
    figure.suptitle("南华商品指数分状态表现：指标窗口敏感性", y=1.03)
    return figure


def plot_sector_correlation(environment: pd.DataFrame) -> Figure:
    """Plot within-sector and across-sector average absolute correlations."""
    figure, axes = _new_axes("图 3 — 板块内外相关性分解", "绝对相关性")
    axes.plot(environment.index, environment["within_sector_correlation"], label="板块内相关性")
    axes.plot(environment.index, environment["across_sector_correlation"], label="板块间相关性")
    _set_x_limits(axes, environment.index)
    axes.legend()
    return figure


def plot_sector_pair_correlation(environment: pd.DataFrame) -> Figure:
    """Plot the equally weighted absolute correlation of sector pairs."""
    figure, axes = _new_axes("图 3 — 板块组合两两绝对相关性", "绝对相关性")
    axes.plot(environment.index, environment["correlation"], label="板块对相关性")
    _set_x_limits(axes, environment.index)
    axes.legend()
    return figure


def plot_sector_volatility_river(contributions: pd.DataFrame) -> Figure:
    """Plot monthly sector contribution shares as one continuous stacked river."""
    if contributions.empty or contributions.shape[1] == 0:
        raise ValueError("Sector volatility contribution data is empty.")
    data = contributions.copy()
    data.index = pd.to_datetime(data.index)
    data = data.sort_index().apply(pd.to_numeric, errors="coerce")
    filled_data = data.fillna(0.0)
    if not (filled_data.ge(0).all().all() and filled_data.sum(axis=1).sub(1.0).abs().le(1e-8).all()):
        raise ValueError("Sector volatility contribution shares must be non-negative and sum to one.")
    figure, axes = plt.subplots(figsize=(16, 10), constrained_layout=True)
    palette = plt.get_cmap("tab20").colors
    colors = {
        column: palette[position % len(palette)] for position, column in enumerate(data.columns)
    }
    if len(data) == 1:
        bottom = 0.0
        for column in data.columns:
            value = float(data[column].iloc[0])
            axes.bar(
                data.index,
                [value],
                bottom=bottom,
                width=20,
                color=colors[column],
                label=column,
                edgecolor="white",
                linewidth=0.25,
            )
            bottom += value
    else:
        axes.stackplot(
            data.index,
            data.fillna(0.0).T.to_numpy(),
            labels=data.columns,
            colors=[colors[column] for column in data.columns],
            baseline="zero",
            linewidth=0.25,
            edgecolor="white",
        )
    axes.set(xlabel="日期", ylabel="贡献占比", ylim=(0, 1))
    axes.yaxis.set_major_formatter(PercentFormatter(1.0))
    axes.grid(axis="y", alpha=0.25)
    axes.legend(loc="center left", bbox_to_anchor=(1.01, 0.5), frameon=False)
    _set_x_limits(axes, data.index)
    figure.suptitle("各板块对波动率的贡献占比")
    return figure


def _threshold_label(months: int) -> str:
    return f"{months // 12}年" if months % 12 == 0 else f"{months}个月"


def _scenario_label(key: tuple[str, int]) -> str:
    method, threshold_months = key
    return f"{AGGREGATION_LABELS.get(method, method)} · 中位数{_threshold_label(threshold_months)}"


def plot_breakpoint_scenario_comparison(
    diagnostics_by_scenario: Mapping[tuple[str, int], pd.DataFrame],
) -> Figure:
    """Compare fixed-window metrics and threshold medians across breakpoint scenarios."""
    if not diagnostics_by_scenario:
        raise ValueError("Breakpoint scenario diagnostics are empty.")
    windows = sorted({months for _, months in diagnostics_by_scenario})
    methods = list(dict.fromkeys(method for method, _ in diagnostics_by_scenario))
    figure, axes = plt.subplots(
        2,
        len(windows),
        figsize=(6.2 * len(windows), 9),
        sharex="col",
        squeeze=False,
        constrained_layout=True,
    )
    metric_specs = (
        ("volatility", "年化波动率", "raw_volatility", "ramp_volatility", "ramp_volatility_threshold"),
        ("correlation", "绝对相关性", "raw_correlation", "ramp_correlation", "ramp_correlation_threshold"),
    )
    for column, months in enumerate(windows):
        for row, (_, ylabel, raw_column, ramp_column, threshold_column) in enumerate(metric_specs):
            axis = axes[row, column]
            for method in methods:
                data = diagnostics_by_scenario[(method, months)].sort_index()
                color = AGGREGATION_COLORS.get(method, f"C{methods.index(method)}")
                axis.plot(
                    data.index,
                    data[raw_column],
                    color=color,
                    linewidth=1.0,
                    label=f"{AGGREGATION_LABELS.get(method, method)}·原始",
                )
                axis.plot(
                    data.index,
                    data[ramp_column],
                    color=color,
                    linestyle="--",
                    linewidth=1.0,
                    label=f"{AGGREGATION_LABELS.get(method, method)}·线性渐进",
                )
                if threshold_column in data:
                    axis.plot(
                        data.index,
                        data[threshold_column],
                        color=color,
                        linestyle=":",
                        linewidth=1.2,
                        label=f"{AGGREGATION_LABELS.get(method, method)}·阈值",
                    )
                change_column = "member_set_changed" if "member_set_changed" in data else "sector_set_changed"
                for breakpoint in data.index[data[change_column].fillna(False)]:
                    axis.axvline(breakpoint, color="#777777", alpha=0.12, linewidth=0.6)
            axis.set_ylabel(ylabel if column == 0 else "")
            axis.set_title(f"中位数回看{_threshold_label(months)}")
            axis.grid(alpha=0.25)
            _set_x_limits(axis, data.index[data[raw_column].notna()])
    handles = [
        plt.Line2D([0], [0], color=AGGREGATION_COLORS.get(method, f"C{position}"), linewidth=1.2, label=label)
        for position, method in enumerate(methods)
        for label in (
            f"{AGGREGATION_LABELS.get(method, method)}·原始",
            f"{AGGREGATION_LABELS.get(method, method)}·线性渐进",
            f"{AGGREGATION_LABELS.get(method, method)}·阈值",
        )
    ]
    for handle, linestyle in zip(handles, ["-", "--", ":"] * len(methods), strict=True):
        handle.set_linestyle(linestyle)
    figure.legend(handles=handles, loc="upper center", ncols=3, bbox_to_anchor=(0.5, 1.09))
    for axis in axes[-1]:
        axis.set_xlabel("日期")
    figure.suptitle("断点影响：原始指标与六个月线性渐进纳入", y=1.18)
    return figure


def plot_nhci_regime_scenario_comparison(
    nhci: pd.Series,
    stages_by_scenario: Mapping[tuple[str, int], pd.DataFrame],
) -> Figure:
    """Plot NHCI with regime shading for every configured comparison scenario."""
    if nhci.empty or not stages_by_scenario:
        raise ValueError("NHCI or breakpoint scenario stages are empty.")
    scenarios = list(stages_by_scenario)
    figure, axes_grid = plt.subplots(
        len(scenarios),
        1,
        figsize=(16, max(8, 2.5 * len(scenarios))),
        sharex=True,
        squeeze=False,
        constrained_layout=True,
    )
    axes = axes_grid[:, 0]
    for axis, key in zip(axes, scenarios, strict=True):
        stages = stages_by_scenario[key]
        valid_stages = stages.loc[stages["regime"].ne(INSUFFICIENT_HISTORY)]
        _shade_stages(axis, valid_stages, regimes=REGIME_COLORS.keys() - {INSUFFICIENT_HISTORY})
        axis.plot(nhci.index, nhci, color="#222222", linewidth=0.8)
        axis.set_ylabel(_scenario_label(key), rotation=0, labelpad=58, va="center")
        axis.grid(alpha=0.2)
        first_valid = valid_stages["start_date"].min() if not valid_stages.empty else nhci.index.min()
        axis.set_xlim(max(nhci.index.min(), first_valid), nhci.index.max())
    handles = [
        plt.Line2D([0], [0], color=REGIME_COLORS[regime], linewidth=8, alpha=0.35, label=REGIME_LABELS[regime])
        for regime in FORMAL_REGIMES
    ]
    figure.legend(handles=handles, loc="upper center", ncols=4, bbox_to_anchor=(0.5, 1.01))
    axes[-1].set_xlabel("日期")
    figure.suptitle("南华商品指数与中位数窗口场景的线性渐进状态区间", y=1.04)
    return figure


def plot_nhci_scenario_performance(performance: pd.DataFrame) -> Figure:
    """Plot NHCI annualized return and Sharpe heatmaps by scenario and regime."""
    required = {"annualized_return", "annualized_sharpe"}
    if performance.empty:
        raise ValueError("NHCI scenario performance is empty.")
    if not required.issubset(performance.columns):
        missing = sorted(required.difference(performance.columns))
        raise ValueError(f"NHCI scenario performance is missing columns: {missing}")
    threshold_level = (
        "regime_threshold_lookback_months"
        if "regime_threshold_lookback_months" in performance.index.names
        else "lookback_months"
    )
    if not {"aggregation_method", threshold_level, "regime"}.issubset(performance.index.names):
        raise ValueError(
            "NHCI scenario performance must use aggregation_method, threshold lookback, and regime index levels."
        )
    data = performance.reset_index()
    scenarios = data[["aggregation_method", threshold_level]].drop_duplicates().itertuples(index=False, name=None)
    scenarios = list(scenarios)
    labels = [_scenario_label(key) for key in scenarios]
    figure, axes = plt.subplots(1, 2, figsize=(16, max(6, 0.65 * len(scenarios) + 3)), constrained_layout=True)
    specs = (("annualized_return", "年化收益", "RdYlGn", ".1%"), ("annualized_sharpe", "夏普比率", "PuBu", ".2f"))
    for axis, (column, title, colormap, number_format) in zip(axes, specs, strict=True):
        table = data.pivot(index=["aggregation_method", threshold_level], columns="regime", values=column)
        table = table.reindex(pd.MultiIndex.from_tuples(scenarios, names=table.index.names))
        table = table.reindex(columns=FORMAL_REGIMES)
        image = axis.imshow(table.to_numpy(dtype=float), aspect="auto", cmap=colormap)
        axis.set(
            title=title,
            xticks=range(len(FORMAL_REGIMES)),
            xticklabels=[REGIME_LABELS[regime] for regime in FORMAL_REGIMES],
            yticks=range(len(labels)),
            yticklabels=labels,
        )
        axis.tick_params(axis="x", rotation=25)
        for row in range(table.shape[0]):
            for col in range(table.shape[1]):
                value = table.iloc[row, col]
                if pd.notna(value):
                    axis.text(col, row, format(value, number_format), ha="center", va="center", fontsize=9)
        figure.colorbar(image, ax=axis, shrink=0.8)
    figure.suptitle("南华商品指数分状态表现：两种口径与不同回看窗口", y=1.03)
    return figure


def _shade_stages(axes: Axes, stages: pd.DataFrame, regimes: Iterable[str] | None = None) -> None:
    allowed = set(regimes) if regimes else set(REGIME_COLORS)
    for row in stages.itertuples(index=False):
        if row.regime in allowed:
            axes.axvspan(row.start_date, row.end_date, color=REGIME_COLORS[row.regime], alpha=0.17, linewidth=0)


def plot_nhci_regimes(nhci: pd.Series, stages: pd.DataFrame) -> Figure:
    """Plot the unnormalized Nanhua Commodity Index with valid market-regime stages."""
    figure, axes = _new_axes("图 3 — 南华商品指数", "指数点位", (14, 6))
    valid_stages = stages.loc[stages["regime"].ne(INSUFFICIENT_HISTORY)]
    _shade_stages(axes, valid_stages, regimes=REGIME_COLORS.keys() - {INSUFFICIENT_HISTORY})
    axes.plot(nhci.index, nhci, color="#222222", linewidth=1.1, label="南华商品指数")
    first_valid_stage = valid_stages["start_date"].min() if not valid_stages.empty else nhci.index.min()
    axes.set_xlim(max(nhci.index.min(), first_valid_stage), nhci.index.max())
    handles = [
        plt.Line2D([0], [0], color=REGIME_COLORS[regime], linewidth=8, alpha=0.35, label=REGIME_LABELS[regime])
        for regime in REGIME_COLORS
        if regime != INSUFFICIENT_HISTORY
    ]
    axes.legend(handles=handles, ncols=2)
    return figure


def plot_fund_vs_nhci(fund_nav: pd.Series, nhci: pd.Series, stages: pd.DataFrame, fund_name: str) -> Figure:
    """Plot one fund and the Nanhua Commodity Index normalized at their first shared observation."""
    aligned = pd.concat([fund_nav.rename(fund_name), nhci.rename("NHCI")], axis=1, sort=True).dropna()
    if aligned.empty or aligned.iloc[0].le(0).any():
        raise ValueError(f"{fund_name} has no positive observations shared with the Nanhua Commodity Index.")
    aligned = aligned / aligned.iloc[0]
    figure, axes = _new_axes(f"{fund_name} — 净值对比", "归一化净值", (14, 6))
    valid_stages = stages.loc[stages["regime"].ne(INSUFFICIENT_HISTORY)]
    _shade_stages(axes, valid_stages, regimes=REGIME_COLORS.keys() - {INSUFFICIENT_HISTORY})
    axes.plot(aligned.index, aligned[fund_name], linewidth=1.2, label=fund_name)
    axes.plot(aligned.index, aligned["NHCI"], color="#222222", linewidth=1.1, label="南华商品指数")
    first_valid_stage = valid_stages["start_date"].min() if not valid_stages.empty else aligned.index.min()
    axes.set_xlim(max(aligned.index.min(), first_valid_stage), aligned.index.max())
    handles = [
        plt.Line2D([0], [0], color=REGIME_COLORS[regime], linewidth=8, alpha=0.35, label=REGIME_LABELS[regime])
        for regime in REGIME_COLORS
        if regime != INSUFFICIENT_HISTORY
    ]
    axes.legend(handles=[*handles, *axes.lines], ncols=3)
    return figure


def plot_regime_cumulative_returns(
    cumulative_returns: pd.DataFrame,
    title: str = "图 4 — 南华商品指数状态累计收益",
    sharex: bool = False,
    sharey: bool = False,
    logy: bool = False,
) -> Figure:
    """Plot four state-conditional cumulative return curves in a 2x2 layout."""
    if cumulative_returns.empty:
        raise ValueError("Cumulative return data is empty.")
    figure, axes = plt.subplots(
        2,
        2,
        figsize=(14, 8),
        constrained_layout=True,
        sharex=sharex,
        sharey=sharey,
    )
    series_by_regime = {
        regime: cumulative_returns[regime].dropna()
        if regime in cumulative_returns
        else pd.Series(dtype=float)
        for regime in FORMAL_REGIMES
    }
    shared_x_end = max((len(series) for series in series_by_regime.values()), default=0)
    for position, (axis, regime) in enumerate(zip(axes.flat, FORMAL_REGIMES, strict=True)):
        series = series_by_regime[regime]
        axis.axhline(1.0, color="#888888", linestyle="--", linewidth=0.8)
        if not series.empty:
            positions = range(1, len(series) + 1)
            axis.plot(positions, series.to_numpy(), color="#222222", linewidth=1.1)
            if shared_x_end > 1:
                axis.set_xlim(1, shared_x_end if sharex else len(series))
        axis.set(
            title=REGIME_LABELS[regime],
            xlabel="",
            ylabel="累计净值" if position % 2 == 0 else "",
        )
        axis.set_xticks([])
        if logy:
            axis.set_yscale("log")
        axis.grid(alpha=0.25)
    figure.suptitle(title)
    return figure


def plot_cta_regime_performance(cumulative_nav: pd.DataFrame, product_id: str) -> Figure:
    """Plot one CTA product's four state-attributed cumulative NAV paths."""
    if product_id not in cumulative_nav.columns.get_level_values("product_id"):
        raise KeyError(f"CTA product {product_id!r} is not available.")
    figure, axes = _new_axes(f"图 5 — CTA 状态累计表现：{product_id}", "累计净值")
    for regime in FORMAL_REGIMES:
        axes.plot(
            cumulative_nav.index,
            cumulative_nav[(product_id, regime)],
            color=REGIME_COLORS[regime],
            label=REGIME_LABELS[regime],
        )
    axes.legend(ncols=2)
    return figure
