from pathlib import Path
from types import SimpleNamespace

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from matplotlib.ticker import PercentFormatter

from cta_research.cta import (
    compute_cta_monthly_returns,
    compute_cta_regime_statistics,
    compute_regime_cumulative_nav,
)
from cta_research.environment import (
    MonthlyEnvironmentResult,
    RampInEnvironmentResult,
    _compute_ramp_in_sector_returns,
    compute_linear_ramp_weights,
    compute_log_returns,
    compute_monthly_environment,
    compute_monthly_environment_by_aggregation,
    compute_monthly_ramp_in_environment_by_aggregation,
    compute_monthly_sector_environment,
    compute_sector_returns,
    compute_sector_volatility_contribution_shares,
)
from cta_research.histories import normalize_nhci_history
from cta_research.plotting import (
    plot_breakpoint_metric_timeseries,
    plot_breakpoint_scenario_comparison,
    plot_cta_regime_performance,
    plot_fund_vs_nhci,
    plot_market_environment,
    plot_metric_lookback_performance,
    plot_metric_lookback_regime_comparison,
    plot_metric_lookback_timeseries,
    plot_nhci_regime_scenario_comparison,
    plot_nhci_regimes,
    plot_nhci_scenario_performance,
    plot_regime_cumulative_returns,
    plot_sector_correlation,
    plot_sector_volatility_river,
)
from cta_research.regimes import (
    FORMAL_REGIMES,
    INSUFFICIENT_HISTORY,
    add_rolling_thresholds,
    classify_monthly_regimes,
    compress_regime_stages,
    compute_regime_agreement_matrix,
    map_monthly_regimes_to_daily,
    validate_target_regime,
)
from cta_research.research import (
    build_breakpoint_scenarios,
    build_metric_lookback_scenarios,
    build_sector_research_context,
    compute_breakpoint_threshold_duration_summary,
)
from cta_research.rqdata import (
    group_incremental_requests,
    merge_rqdata_price_updates,
    normalize_rqdata_prices,
    select_commodity_instruments,
)

matplotlib.use("Agg")


@pytest.fixture(autouse=True)
def close_matplotlib_figures() -> None:
    yield
    plt.close("all")


def test_rqdata_normalization_and_commodity_selection() -> None:
    raw_metadata = pd.DataFrame(
        {
            "underlying_symbol": ["CU", "IF", "RB"],
            "exchange": ["SHFE", "CFFEX", "SHFE"],
        }
    )
    selected, excluded = select_commodity_instruments(raw_metadata)
    assert selected["underlying_symbol"].tolist() == ["CU", "RB"]
    assert excluded["underlying_symbol"].tolist() == ["IF"]

    raw_prices = pd.DataFrame(
        {"close": [100.0, 101.0]},
        index=pd.MultiIndex.from_tuples(
            [("CU", "2011-07-01"), ("CU", "2011-07-04")],
            names=["underlying_symbol", "date"],
        ),
    )
    prices = normalize_rqdata_prices(raw_prices)
    assert prices[["underlying_symbol", "date", "close"]].to_dict("records") == [
        {"underlying_symbol": "CU", "date": pd.Timestamp("2011-07-01"), "close": 100.0},
        {"underlying_symbol": "CU", "date": pd.Timestamp("2011-07-04"), "close": 101.0},
    ]

    nhci = normalize_nhci_history(
        pd.DataFrame({"date": ["2011-06-30", "2011-07-01"], "close": [99, 100]}),
        "2011-07-01",
    )
    assert nhci.to_dict("records") == [{"date": pd.Timestamp("2011-07-01"), "close": 100}]

    nhci_from_excel_columns = normalize_nhci_history(
        pd.DataFrame(
            {
                "日期": ["2011-07-02", "2011-07-01"],
                "指数点位": [101.5, 100.0],
                "涨跌幅": ["1.50%", "0.00%"],
            }
        ),
        "2011-07-01",
    )
    assert nhci_from_excel_columns.to_dict("records") == [
        {"date": pd.Timestamp("2011-07-01"), "close": 100.0},
        {"date": pd.Timestamp("2011-07-02"), "close": 101.5},
    ]


def test_incremental_price_requests_and_merge() -> None:
    cached = pd.DataFrame(
        {
            "underlying_symbol": ["A", "A", "B", "D"],
            "date": pd.to_datetime(["2011-09-29", "2011-09-30", "2011-09-29", "2011-10-01"]),
            "close": [99.0, 100.0, 200.0, 400.0],
        }
    )
    groups = group_incremental_requests(
        ["A", "B", "C", "D"], cached, "2011-07-01", "2011-10-01"
    )
    assert groups == {"2011-07-01": ["C"], "2011-09-30": ["B"], "2011-10-01": ["A"]}

    updates = pd.DataFrame(
        {
            "underlying_symbol": ["A", "A"],
            "date": pd.to_datetime(["2011-09-30", "2011-10-01"]),
            "close": [101.0, 102.0],
        }
    )
    merged = merge_rqdata_price_updates(cached, updates)
    assert merged.loc[merged["underlying_symbol"].eq("A") & merged["date"].eq("2011-09-30"), "close"].item() == 101.0
    assert merged.loc[merged["underlying_symbol"].eq("A") & merged["date"].eq("2011-10-01"), "close"].item() == 102.0
    assert len(merged) == len(cached) + 1


def test_log_returns_do_not_fill_missing_levels() -> None:
    levels = pd.DataFrame(
        {
            "date": pd.to_datetime(["2020-01-01", "2020-01-02", "2020-01-01", "2020-01-03"]),
            "underlying_symbol": ["A", "A", "B", "B"],
            "close": [100.0, 110.0, 100.0, 120.0],
        }
    )
    returns = compute_log_returns(levels)
    assert returns.loc[pd.Timestamp("2020-01-02"), "A"] == pytest.approx(np.log(1.1))
    assert np.isnan(returns.loc[pd.Timestamp("2020-01-02"), "B"])
    assert returns.loc[pd.Timestamp("2020-01-03"), "B"] == pytest.approx(np.log(1.2))


def test_monthly_environment_uses_product_and_pair_equal_means() -> None:
    dates = pd.bdate_range("2020-01-01", "2020-01-31")
    base = np.arange(1, len(dates) + 1, dtype=float)
    returns = pd.DataFrame({"A": base, "B": base * 2, "C": -base}, index=dates)
    sectors = pd.DataFrame(
        {"index_code": ["A", "B", "C"], "sector": ["grains", "grains", "metals"]}
    )
    result = compute_monthly_environment(
        returns,
        sectors,
        lookback_months=1,
        min_observations=len(dates),
        min_pair_observations=len(dates),
        annualization_days=1,
    )
    month = pd.Timestamp("2020-01-31")
    expected_volatility = returns.std(ddof=1).mean()
    assert result.metrics.loc[month, "volatility"] == pytest.approx(expected_volatility)
    assert result.metrics.loc[month, "correlation"] == pytest.approx(1.0)
    assert result.metrics.loc[month, "n_pairs"] == 3
    assert result.metrics.loc[month, "within_sector_correlation"] == pytest.approx(1.0)
    assert result.metrics.loc[month, "across_sector_correlation"] == pytest.approx(1.0)
    assert len(result.correlation_by_month[month]) == 3


def test_environment_dispatch_and_product_pair_identity() -> None:
    dates = pd.bdate_range("2020-01-01", "2020-02-28")
    base = np.arange(1.0, len(dates) + 1)
    returns = pd.DataFrame(
        {"A": base, "B": base * 2, "C": -base, "UNMAPPED": base * 0.5}, index=dates
    )
    sectors = pd.DataFrame(
        {"index_code": ["A", "B", "C"], "sector": ["grains", "grains", "metals"]}
    )
    result = compute_monthly_environment_by_aggregation(
        returns, sectors, "variety_equal", lookback_months=1, min_observations=1, min_pair_observations=1
    )
    month = pd.Timestamp("2020-02-29")
    assert result.metrics.loc[month, "n_varieties"] == 4
    ramp = compute_monthly_ramp_in_environment_by_aggregation(
        returns, sectors, "variety_equal", lookback_months=1, min_observations=1, min_pair_observations=1
    )
    assert ramp.ramp_result.metrics.loc[month, "valid_member_count"] == 4
    assert ramp.ramp_result.metrics.loc[month, "valid_pair_count"] == 6
    assert ramp.ramp_result.metrics.loc[month, "correlation"] == pytest.approx(1.0)
    assert ramp.ramp_result.metrics.loc[month, "correlation"] >= 0
    with pytest.raises(ValueError, match="Unsupported aggregation method"):
        compute_monthly_environment_by_aggregation(returns, sectors, "invalid")
    with pytest.raises(ValueError, match="breakpoint_analysis"):
        build_breakpoint_scenarios(SimpleNamespace(config={}))


def test_formal_context_uses_configured_ramp_in_environment(monkeypatch, tmp_path: Path) -> None:
    project = tmp_path
    (project / "config").mkdir()
    (project / "data").mkdir()
    config_path = project / "config" / "config.yaml"
    config_path.write_text("placeholder: true", encoding="utf-8")
    (project / "data" / "future.parquet").touch()
    (project / "data" / "dominant.parquet").touch()
    (project / "data" / "nhci.parquet").touch()
    pd.DataFrame({"index_code": ["A"], "sector": ["谷物"]}).to_csv(
        project / "config" / "commodity_sectors.csv", index=False
    )
    months = pd.date_range("2015-01-31", periods=2, freq="ME")
    fake_result = MonthlyEnvironmentResult(
        metrics=pd.DataFrame(
            {"volatility": [0.1, 0.2], "correlation": [0.2, 0.3]}, index=months
        ),
        volatility_by_month={month: pd.Series({"谷物": 0.1}) for month in months},
        correlation_by_month={month: pd.DataFrame() for month in months},
        exclusions_by_month={},
        sector_counts_by_month={},
    )
    fake_ramp = RampInEnvironmentResult(
        raw_result=fake_result,
        ramp_result=fake_result,
        variety_weights_by_month={},
        member_weights_by_month={},
        pair_weights_by_month={},
    )
    calls: list[tuple[object, ...]] = []

    def fake_config(_: object) -> dict[str, object]:
        return {
            "paths": {
                "future_instruments": "data/future.parquet",
                "dominant_prices": "data/dominant.parquet",
                "nhci": "data/nhci.parquet",
            },
            "start_date": "2011-07-01",
            "end_date": None,
            "aggregation_method": "sector_equal",
            "environment_lookback_months": 3,
            "min_observations": 40,
            "min_pair_observations": 40,
            "annualization_days": 252,
            "ramp_in_months": 6,
            "regime_threshold_lookback_months": 36,
            "regime_threshold_min_periods": 36,
            "validation_target_date": "2015-02-02",
        }

    def fake_compute(*args: object) -> RampInEnvironmentResult:
        calls.append(args)
        return fake_ramp

    future = pd.DataFrame({"underlying_symbol": ["A"]})
    dominant = pd.DataFrame(
        {
            "underlying_symbol": ["A", "A"],
            "date": pd.to_datetime(["2015-01-01", "2015-02-02"]),
            "close": [100.0, 101.0],
        }
    )
    monkeypatch.setattr("cta_research.research.load_config", fake_config)
    monkeypatch.setattr(
        "cta_research.research.pd.read_parquet",
        lambda path: future if Path(path).name == "future.parquet" else dominant,
    )
    monkeypatch.setattr(
        "cta_research.research.read_history_file",
        lambda _: pd.DataFrame({"date": ["2015-01-01"], "close": [100.0]}),
    )
    monkeypatch.setattr(
        "cta_research.research.normalize_nhci_history",
        lambda data, *_: pd.DataFrame({"date": pd.to_datetime(data["date"]), "close": data["close"]}),
    )
    monkeypatch.setattr(
        "cta_research.research.compute_log_returns",
        lambda _: pd.DataFrame({"A": [0.01]}, index=pd.to_datetime(["2015-01-01"])),
    )
    monkeypatch.setattr("cta_research.research.compute_monthly_ramp_in_environment_by_aggregation", fake_compute)
    monkeypatch.setattr(
        "cta_research.research.compute_sector_returns",
        lambda *_: pd.DataFrame({"谷物": [0.01]}, index=pd.to_datetime(["2015-01-01"])),
    )
    monkeypatch.setattr("cta_research.research.add_rolling_thresholds", lambda metrics, *_: metrics)
    monkeypatch.setattr(
        "cta_research.research.classify_monthly_regimes",
        lambda metrics: metrics.assign(regime="LV_LC"),
    )
    monkeypatch.setattr(
        "cta_research.research.map_monthly_regimes_to_daily",
        lambda *_: pd.DataFrame({"regime": ["LV_LC"]}, index=pd.to_datetime(["2015-01-01"])),
    )
    monkeypatch.setattr(
        "cta_research.research.validate_target_regime",
        lambda *_: pd.Series({"regime": "LV_LC"}),
    )
    monkeypatch.setattr(
        "cta_research.research.compress_regime_stages",
        lambda *_: pd.DataFrame(),
    )

    context = build_sector_research_context(config_path)

    assert len(calls) == 1
    assert calls[0][2:] == ("sector_equal", 3, 40, 40, 252, 6)
    assert context.ramp_in_result is fake_ramp
    assert context.environment_result is fake_result


def test_breakpoint_scenarios_fix_metric_window_and_scan_threshold_windows(monkeypatch) -> None:
    months = pd.date_range("2010-01-31", periods=72, freq="ME")
    fake_result = MonthlyEnvironmentResult(
        metrics=pd.DataFrame(
            {
                "volatility": np.linspace(0.1, 0.3, len(months)),
                "correlation": np.linspace(0.2, 0.4, len(months)),
            },
            index=months,
        ),
        volatility_by_month={date: pd.Series({"A": 0.1, "B": 0.2}) for date in months},
        correlation_by_month={
            date: pd.DataFrame({"sector_1": ["A"], "sector_2": ["B"], "abs_correlation": [0.3]})
            for date in months
        },
        exclusions_by_month={},
        sector_counts_by_month={},
    )
    calls = []

    fake_ramp = RampInEnvironmentResult(
        raw_result=fake_result,
        ramp_result=fake_result,
        variety_weights_by_month={date: pd.Series({"A": 1.0, "B": 1.0}) for date in months},
        member_weights_by_month={date: pd.Series({"A": 1.0, "B": 1.0}) for date in months},
        pair_weights_by_month={date: pd.Series(dtype="float64") for date in months},
    )

    def fake_compute(*args, **kwargs):
        calls.append(args[2])
        return fake_ramp

    monkeypatch.setattr("cta_research.research.compute_monthly_ramp_in_environment_by_aggregation", fake_compute)
    daily_dates = pd.date_range("2010-01-01", "2016-01-05", freq="D")
    context = SimpleNamespace(
        config={
            "breakpoint_analysis": {
                "aggregation_methods": ["sector_equal", "variety_equal"],
                "environment_lookback_months": 3,
                "ramp_in_months": 6,
                "min_observations": 0,
                "min_pair_observations": 0,
                "regime_threshold_lookback_months": [12, 36, 60],
            },
            "aggregation_method": "sector_equal",
            "environment_lookback_months": 6,
            "min_observations": 100,
            "min_pair_observations": 100,
            "annualization_days": 252,
        },
        environment_result=fake_result,
        daily_returns=pd.DataFrame(),
        sector_mapping=pd.DataFrame(),
        nhci=pd.Series(100.0, index=daily_dates),
    )

    results = build_breakpoint_scenarios(context)

    assert calls == ["sector_equal", "variety_equal"]
    assert set(results) == {
        ("sector_equal", 12),
        ("sector_equal", 36),
        ("sector_equal", 60),
        ("variety_equal", 12),
        ("variety_equal", 36),
        ("variety_equal", 60),
    }
    assert all(result.environment_lookback_months == 3 for result in results.values())
    assert all(result.lookback_months == 3 for result in results.values())
    for method in ("sector_equal", "variety_equal"):
        method_results = [result for (scenario_method, _), result in results.items() if scenario_method == method]
        assert len({id(result.environment_result) for result in method_results}) == 1
    for threshold_months in (12, 36, 60):
        result = results[("sector_equal", threshold_months)]
        valid_months = result.monthly_regimes.index[result.monthly_regimes["regime"].ne(INSUFFICIENT_HISTORY)]
        assert valid_months[0] == months[threshold_months]
    invalid_settings = context.config["breakpoint_analysis"].copy()
    invalid_settings["regime_threshold_lookback_months"] = [12, 12]
    invalid_context = SimpleNamespace(**context.__dict__)
    invalid_context.config = {**context.config, "breakpoint_analysis": invalid_settings}
    with pytest.raises(ValueError, match="must not contain duplicates"):
        build_breakpoint_scenarios(invalid_context)


def test_metric_lookback_scenarios_scan_windows_and_validate_config(monkeypatch) -> None:
    months = pd.date_range("2010-01-31", periods=72, freq="ME")
    fake_result = MonthlyEnvironmentResult(
        metrics=pd.DataFrame(
            {
                "volatility": np.linspace(0.1, 0.3, len(months)),
                "correlation": np.linspace(0.2, 0.4, len(months)),
            },
            index=months,
        ),
        volatility_by_month={date: pd.Series({"A": 0.1, "B": 0.2}) for date in months},
        correlation_by_month={
            date: pd.DataFrame({"sector_1": ["A"], "sector_2": ["B"], "abs_correlation": [0.3]})
            for date in months
        },
        exclusions_by_month={},
        sector_counts_by_month={},
    )
    fake_ramp = RampInEnvironmentResult(
        raw_result=fake_result,
        ramp_result=fake_result,
        variety_weights_by_month={},
        member_weights_by_month={},
        pair_weights_by_month={},
    )
    calls = []

    def fake_compute(*args, **kwargs):
        calls.append((args[2], args[3]))
        return fake_ramp

    monkeypatch.setattr("cta_research.research.compute_monthly_ramp_in_environment_by_aggregation", fake_compute)
    context = SimpleNamespace(
        config={
            "metric_lookback_analysis": {
                "aggregation_method": "sector_equal",
                "environment_lookback_months": [1, 3, 6],
                "regime_threshold_lookback_months": 36,
                "ramp_in_months": 6,
                "min_observations": 0,
                "min_pair_observations": 0,
            },
            "annualization_days": 252,
        },
        daily_returns=pd.DataFrame(),
        sector_mapping=pd.DataFrame(),
        nhci=pd.Series(100.0, index=pd.date_range("2010-01-01", "2016-01-05", freq="D")),
    )
    results = build_metric_lookback_scenarios(context)
    assert list(results) == [1, 3, 6]
    assert calls == [("sector_equal", 1), ("sector_equal", 3), ("sector_equal", 6)]
    assert all(result.regime_threshold_lookback_months == 36 for result in results.values())
    assert all(result.lookback_months == window for window, result in results.items())

    for key, value in (
        ("environment_lookback_months", []),
        ("environment_lookback_months", [1, 1]),
        ("environment_lookback_months", [0, 3]),
        ("regime_threshold_lookback_months", 0),
    ):
        settings = context.config["metric_lookback_analysis"].copy()
        settings[key] = value
        invalid_context = SimpleNamespace(**context.__dict__)
        invalid_context.config = {**context.config, "metric_lookback_analysis": settings}
        with pytest.raises(ValueError, match="metric_lookback_analysis"):
            build_metric_lookback_scenarios(invalid_context)
    invalid_settings = context.config["metric_lookback_analysis"].copy()
    invalid_settings["aggregation_method"] = "variety_equal"
    invalid_context = SimpleNamespace(**context.__dict__)
    invalid_context.config = {**context.config, "metric_lookback_analysis": invalid_settings}
    with pytest.raises(ValueError, match="sector_equal"):
        build_metric_lookback_scenarios(invalid_context)


def test_sector_environment_aggregates_equal_weight_sector_portfolios() -> None:
    dates = pd.bdate_range("2020-01-01", "2020-01-31")
    returns = pd.DataFrame(
        {
            "A": np.arange(1.0, len(dates) + 1),
            "B": np.arange(3.0, len(dates) + 3),
            "C": -np.arange(1.0, len(dates) + 1),
            "UNKNOWN": 1.0,
        },
        index=dates,
    )
    sectors = pd.DataFrame(
        {"index_code": ["A", "B", "C"], "sector": ["grains", "grains", "metals"]}
    )

    sector_returns = compute_sector_returns(returns, sectors)
    result = compute_monthly_sector_environment(
        returns,
        sectors,
        lookback_months=1,
        min_observations=len(dates),
        min_pair_observations=len(dates),
        annualization_days=1,
    )
    month = pd.Timestamp("2020-01-31")

    assert list(sector_returns.columns) == ["grains", "metals"]
    assert sector_returns.loc[dates[0], "grains"] == pytest.approx(2.0)
    assert result.metrics.loc[month, "n_sectors"] == 2
    assert result.metrics.loc[month, "n_pairs"] == 1
    assert result.metrics.loc[month, "correlation"] == pytest.approx(1.0)
    assert result.sector_counts_by_month[month].to_dict() == {"grains": 2, "metals": 1}
    assert "unmapped:UNKNOWN" in result.exclusions_by_month[month]


def test_sector_environment_allows_one_sector_without_pairs() -> None:
    dates = pd.bdate_range("2020-01-01", "2020-01-31")
    returns = pd.DataFrame({"A": np.arange(1.0, len(dates) + 1)}, index=dates)
    sectors = pd.DataFrame({"index_code": ["A"], "sector": ["grains"]})

    result = compute_monthly_sector_environment(
        returns,
        sectors,
        lookback_months=1,
        min_observations=len(dates),
        min_pair_observations=len(dates),
        annualization_days=1,
    )

    month = pd.Timestamp("2020-01-31")
    assert result.metrics.loc[month, "n_sectors"] == 1
    assert result.metrics.loc[month, "n_pairs"] == 0
    assert np.isnan(result.metrics.loc[month, "correlation"])


def test_sector_volatility_contribution_shares_normalize_valid_sectors() -> None:
    volatility_by_month = {
        pd.Timestamp("2020-01-31"): pd.Series({"谷物": 2.0, "黑色": 1.0}),
        pd.Timestamp("2020-02-29"): pd.Series({"谷物": 4.0, "黑色": np.nan, "化工": 2.0}),
        pd.Timestamp("2020-03-31"): pd.Series({"谷物": np.nan, "黑色": np.nan}),
    }

    shares = compute_sector_volatility_contribution_shares(volatility_by_month)

    assert shares.index.tolist() == [pd.Timestamp("2020-01-31"), pd.Timestamp("2020-02-29")]
    assert shares.columns[0] == "谷物"
    assert set(shares.columns) == {"谷物", "黑色", "化工"}
    assert shares.loc[pd.Timestamp("2020-01-31"), "谷物"] == pytest.approx(2 / 3)
    assert shares.loc[pd.Timestamp("2020-01-31"), "黑色"] == pytest.approx(1 / 3)
    assert pd.isna(shares.loc[pd.Timestamp("2020-02-29"), "黑色"])
    assert shares.sum(axis=1).tolist() == pytest.approx([1.0, 1.0])

    weighted = compute_sector_volatility_contribution_shares(
        {pd.Timestamp("2020-01-31"): pd.Series({"谷物": 2.0, "黑色": 1.0})},
        {pd.Timestamp("2020-01-31"): pd.Series({"谷物": 0.25, "黑色": 1.0})},
    )
    assert weighted.loc[pd.Timestamp("2020-01-31"), "谷物"] == pytest.approx(1 / 3)
    assert weighted.loc[pd.Timestamp("2020-01-31"), "黑色"] == pytest.approx(2 / 3)
    assert weighted.sum(axis=1).iloc[0] == pytest.approx(1.0)


def test_monthly_environment_excludes_partial_final_month() -> None:
    dates = pd.bdate_range("2020-01-01", "2020-02-03")
    returns = pd.DataFrame(
        {
            "A": np.sin(np.arange(len(dates)) / 5),
            "B": np.cos(np.arange(len(dates)) / 7),
        },
        index=dates,
    )
    sectors = pd.DataFrame({"index_code": ["A", "B"], "sector": ["grains", "metals"]})

    result = compute_monthly_environment(
        returns,
        sectors,
        lookback_months=1,
        min_observations=1,
        min_pair_observations=1,
        annualization_days=1,
    )

    assert result.metrics.index.tolist() == [pd.Timestamp("2020-01-31")]


def test_linear_ramp_weights_use_calendar_months_without_restarting() -> None:
    dates = pd.date_range("2020-01-31", periods=8, freq="ME")
    weights = compute_linear_ramp_weights(
        {
            dates[0]: ["A", "B"],
            dates[1]: ["A", "B"],
            dates[2]: ["A"],
            dates[3]: ["A", "C"],
            dates[4]: ["A", "C"],
            dates[5]: ["A", "C"],
            dates[6]: ["A", "C"],
            dates[7]: ["A", "C"],
        }
    )
    assert weights[dates[0]]["A"] == pytest.approx(1 / 6)
    assert weights[dates[1]]["A"] == pytest.approx(2 / 6)
    assert weights[dates[3]]["C"] == pytest.approx(1 / 6)
    assert weights[dates[7]]["C"] == pytest.approx(5 / 6)
    assert weights[dates[7]]["A"] == pytest.approx(1.0)


def test_ramp_in_environment_weights_new_varieties_and_pairs() -> None:
    dates = pd.bdate_range("2020-01-01", periods=140)
    returns = pd.DataFrame(
        {
            "A": np.sin(np.arange(len(dates)) / 5),
            "B": np.cos(np.arange(len(dates)) / 7),
            "C": np.sin(np.arange(len(dates)) / 11),
        },
        index=dates,
    )
    sectors = pd.DataFrame(
        {"index_code": ["A", "B", "C"], "sector": ["grains", "grains", "metals"]}
    )
    result = compute_monthly_ramp_in_environment_by_aggregation(
        returns,
        sectors,
        "variety_equal",
        lookback_months=1,
        min_observations=1,
        min_pair_observations=1,
        annualization_days=1,
    )
    first_month = next(iter(result.variety_weights_by_month))
    assert result.variety_weights_by_month[first_month].eq(1 / 6).all()
    pair_weights = result.pair_weights_by_month[first_month]
    assert pair_weights.eq(1 / 36).all()
    assert result.ramp_result.metrics.loc[first_month, "correlation"] == pytest.approx(
        result.raw_result.metrics.loc[first_month, "correlation"]
    )
    assert result.ramp_result.metrics["correlation"].dropna().between(0, 1).all()
    last_month = max(result.variety_weights_by_month)
    assert result.ramp_result.metrics.loc[last_month, "volatility"] == pytest.approx(
        result.raw_result.metrics.loc[last_month, "volatility"]
    )


def test_ramp_in_sector_returns_renormalize_daily_valid_weights() -> None:
    dates = pd.to_datetime(["2020-01-02", "2020-01-03"])
    returns = pd.DataFrame({"A": [1.0, 2.0], "B": [3.0, np.nan]}, index=dates)
    mapping = pd.DataFrame({"index_code": ["A", "B"], "sector": ["grains", "grains"]})
    weighted = _compute_ramp_in_sector_returns(
        returns,
        mapping,
        {pd.Timestamp("2020-01-31"): pd.Series({"A": 1.0, "B": 0.5})},
    )
    assert weighted.loc[dates[0], "grains"] == pytest.approx((1.0 + 3.0 * 0.5) / 1.5)
    assert weighted.loc[dates[1], "grains"] == pytest.approx(2.0)


def test_ramp_in_sector_environment_applies_variety_and_sector_weights() -> None:
    dates = pd.bdate_range("2020-01-01", periods=220)
    returns = pd.DataFrame(
        {
            "A": np.sin(np.arange(len(dates)) / 5),
            "B": np.cos(np.arange(len(dates)) / 7),
            "C": np.where(np.arange(len(dates)) < 90, np.nan, np.sin(np.arange(len(dates)) / 3)),
        },
        index=dates,
    )
    sectors = pd.DataFrame(
        {"index_code": ["A", "B", "C"], "sector": ["grains", "grains", "metals"]}
    )
    result = compute_monthly_ramp_in_environment_by_aggregation(
        returns,
        sectors,
        "sector_equal",
        lookback_months=1,
        min_observations=1,
        min_pair_observations=1,
        annualization_days=1,
    )
    c_month = next(date for date, weights in result.variety_weights_by_month.items() if "C" in weights)
    s2_month = next(date for date, weights in result.member_weights_by_month.items() if "metals" in weights)
    assert result.variety_weights_by_month[c_month]["C"] == pytest.approx(1 / 6)
    assert result.member_weights_by_month[s2_month]["metals"] == pytest.approx(1 / 6)
    pair_weights = result.pair_weights_by_month[s2_month]
    assert pair_weights.iloc[0] == pytest.approx(
        result.member_weights_by_month[s2_month].prod()
    )
    assert result.ramp_result.metrics["correlation"].dropna().between(0, 1).all()


def test_regime_agreement_and_scenario_plots() -> None:
    dates = pd.bdate_range("2020-01-01", periods=5)
    first = pd.DataFrame(
        {
            "regime": [INSUFFICIENT_HISTORY, "LV_LC", "LV_LC", "HV_LC", "HV_LC"],
            "effective_monthly_regime_date": pd.NaT,
        },
        index=dates,
    )
    second = first.copy()
    second["regime"] = [INSUFFICIENT_HISTORY, "LV_LC", "HV_HC", "HV_LC", "HV_LC"]
    agreement = compute_regime_agreement_matrix({"场景 A": first, "场景 B": second})
    assert agreement.loc["场景 A", "场景 B"] == pytest.approx(3 / 4)

    diagnostics = pd.DataFrame(
        {
            "raw_volatility": [0.2, 0.3],
            "ramp_volatility": [0.2, 0.25],
            "raw_correlation": [0.4, 0.5],
            "ramp_correlation": [0.4, 0.45],
            "ramp_volatility_threshold": [0.25, 0.26],
            "ramp_correlation_threshold": [0.42, 0.43],
            "member_set_changed": [False, True],
        },
        index=pd.to_datetime(["2020-01-31", "2020-02-29"]),
    )
    diagnostics_by_scenario = {
        ("sector_equal", 12): diagnostics,
        ("variety_equal", 12): diagnostics,
        ("sector_equal", 36): diagnostics,
        ("variety_equal", 36): diagnostics,
        ("sector_equal", 60): diagnostics,
        ("variety_equal", 60): diagnostics,
    }
    metric_figure = plot_breakpoint_scenario_comparison(diagnostics_by_scenario)
    assert len(metric_figure.axes) == 6
    assert [axis.get_title() for axis in metric_figure.axes[:3]] == [
        "中位数回看1年",
        "中位数回看3年",
        "中位数回看5年",
    ]
    assert {text.get_text() for text in metric_figure.legends[0].get_texts()} == {
        "板块等权·原始",
        "板块等权·线性渐进",
        "板块等权·阈值",
        "品种等权·原始",
        "品种等权·线性渐进",
        "品种等权·阈值",
    }
    timeseries_figure = plot_breakpoint_metric_timeseries(diagnostics_by_scenario, "volatility")
    assert len(timeseries_figure.axes) == 6
    assert tuple(timeseries_figure.get_size_inches()) == pytest.approx((18.0, 8.0))
    assert isinstance(timeseries_figure.axes[0].yaxis.get_major_formatter(), PercentFormatter)
    assert timeseries_figure.axes[0].get_title() == "中位数回看1年"
    assert timeseries_figure.axes[2].get_title() == "中位数回看5年"
    assert all(
        timeseries_figure.axes[0].get_shared_x_axes().joined(timeseries_figure.axes[0], axis)
        for axis in timeseries_figure.axes
    )
    assert all(
        timeseries_figure.axes[0].get_shared_y_axes().joined(timeseries_figure.axes[0], axis)
        for axis in timeseries_figure.axes
    )
    assert len(timeseries_figure.axes[0].lines) == 2
    assert len(timeseries_figure.axes[0].collections) == 2
    correlation_figure = plot_breakpoint_metric_timeseries(diagnostics_by_scenario, "correlation")
    assert not isinstance(correlation_figure.axes[0].yaxis.get_major_formatter(), PercentFormatter)

    lookback_diagnostics = {
        window: diagnostics.assign(
            ramp_volatility_threshold=0.25,
            ramp_correlation_threshold=0.42,
        )
        for window in (1, 3, 6)
    }
    lookback_stages = pd.DataFrame(
        {"regime": ["LV_LC"], "start_date": [dates[1]], "end_date": [dates[-1]]}
    )
    lookback_nhci = pd.Series(np.arange(100.0, 105.0), index=dates)
    lookback_volatility = plot_metric_lookback_timeseries(lookback_diagnostics, "volatility")
    assert len(lookback_volatility.axes) == 3
    assert isinstance(lookback_volatility.axes[0].yaxis.get_major_formatter(), PercentFormatter)
    assert [axis.get_title() for axis in lookback_volatility.axes] == [
        "指标回看1个月",
        "指标回看3个月",
        "指标回看6个月",
    ]
    assert all(
        lookback_volatility.axes[0].get_shared_x_axes().joined(lookback_volatility.axes[0], axis)
        for axis in lookback_volatility.axes
    )
    assert all(
        lookback_volatility.axes[0].get_shared_y_axes().joined(lookback_volatility.axes[0], axis)
        for axis in lookback_volatility.axes
    )
    lookback_correlation = plot_metric_lookback_timeseries(lookback_diagnostics, "correlation")
    assert not isinstance(lookback_correlation.axes[0].yaxis.get_major_formatter(), PercentFormatter)
    lookback_regimes = plot_metric_lookback_regime_comparison(
        lookback_nhci, {window: lookback_stages for window in (1, 3, 6)}
    )
    assert len(lookback_regimes.axes) == 3
    lookback_performance = pd.DataFrame(
        {
            "n_observations": [2] * 12,
            "annualized_return": [0.1] * 12,
            "annualized_sharpe": [0.5] * 12,
        },
        index=pd.MultiIndex.from_product(
            [[1, 3, 6], FORMAL_REGIMES], names=["environment_lookback_months", "regime"]
        ),
    )
    assert len(plot_metric_lookback_performance(lookback_performance).axes) == 4

    duration_dates = pd.date_range("2020-01-31", periods=6, freq="ME")
    duration_diagnostics = pd.DataFrame(
        {
            "ramp_volatility": [1.0, 1.0, np.nan, 2.0, 2.0, 1.0],
            "ramp_volatility_threshold": [1.0] * 6,
            "ramp_correlation": [0.2, 0.3, np.nan, 0.1, 0.1, 0.4],
            "ramp_correlation_threshold": [0.2] * 6,
        },
        index=duration_dates,
    )
    duration_summary = compute_breakpoint_threshold_duration_summary({("sector_equal", 12): duration_diagnostics})
    volatility_summary = duration_summary.loc[("sector_equal", 12, "volatility")]
    assert volatility_summary["valid_months"] == 5
    assert volatility_summary["low_months"] == 3
    assert volatility_summary["high_months"] == 2
    assert volatility_summary["low_share"] + volatility_summary["high_share"] == pytest.approx(1.0)
    assert volatility_summary["max_low_run_months"] == 2
    assert volatility_summary["max_high_run_months"] == 2
    correlation_summary = duration_summary.loc[("sector_equal", 12, "correlation")]
    assert correlation_summary["max_low_run_months"] == 2
    assert correlation_summary["max_high_run_months"] == 1
    stages = pd.DataFrame(
        {
            "regime": ["LV_LC"],
            "start_date": [dates[1]],
            "end_date": [dates[-1]],
        }
    )
    nhci = pd.Series(np.arange(100.0, 105.0), index=dates)
    regime_figure = plot_nhci_regime_scenario_comparison(
        nhci, {key: stages for key in diagnostics_by_scenario}
    )
    assert len(regime_figure.axes) == 6
    performance = pd.DataFrame(
        {
            "n_observations": [2] * 24,
            "annualized_return": [0.1] * 24,
            "annualized_sharpe": [0.5] * 24,
        },
        index=pd.MultiIndex.from_product(
            [["sector_equal", "variety_equal"], [12, 36, 60], ["LV_LC", "HV_LC", "LV_HC", "HV_HC"]],
            names=["aggregation_method", "regime_threshold_lookback_months", "regime"],
        ),
    )
    performance_figure = plot_nhci_scenario_performance(performance)
    assert len(performance_figure.axes) == 4

    cumulative = pd.DataFrame(
        {
            "LV_LC": [1.0, 1.1, 1.05],
            "HV_LC": [1.0, 0.98, 1.02],
            "LV_HC": [1.0, 1.03, 1.08],
            "HV_HC": [1.0, 0.95, 0.9],
        }
    )
    cumulative_figure = plot_regime_cumulative_returns(
        cumulative,
        title="测试场景",
        sharex=True,
        sharey=True,
        logy=True,
    )
    assert len(cumulative_figure.axes) == 4
    assert cumulative_figure._suptitle.get_text() == "测试场景"
    assert all(line.get_ydata()[0] == pytest.approx(1.0) for line in (axis.lines[0] for axis in cumulative_figure.axes))
    assert all(
        cumulative_figure.axes[0].get_shared_y_axes().joined(cumulative_figure.axes[0], axis)
        for axis in cumulative_figure.axes[1:]
    )
    assert all(
        cumulative_figure.axes[0].get_shared_x_axes().joined(cumulative_figure.axes[0], axis)
        for axis in cumulative_figure.axes[1:]
    )
    assert all(axis.get_yscale() == "log" for axis in cumulative_figure.axes)
    expected_end_values = [cumulative[regime].iloc[-1] for regime in FORMAL_REGIMES]
    assert [axis.lines[1].get_ydata()[-1] for axis in cumulative_figure.axes] == pytest.approx(expected_end_values)


def test_thresholds_are_strictly_lagged_and_equal_values_are_low() -> None:
    index = pd.date_range("2012-01-31", periods=61, freq="ME")
    environment = pd.DataFrame(
        {
            "volatility": np.r_[np.arange(1.0, 61.0), 30.5],
            "correlation": np.r_[np.arange(101.0, 161.0), 130.5],
        },
        index=index,
    )
    classified = classify_monthly_regimes(add_rolling_thresholds(environment))
    assert classified.iloc[:60]["regime"].eq(INSUFFICIENT_HISTORY).all()
    assert classified.iloc[60]["volatility_threshold"] == pytest.approx(30.5)
    assert classified.iloc[60]["correlation_threshold"] == pytest.approx(130.5)
    assert classified.iloc[60]["regime"] == "LV_LC"


def test_daily_mapping_uses_next_trading_day_and_target_audit() -> None:
    monthly = pd.DataFrame(
        {
            "volatility": [0.2, 0.3],
            "correlation": [0.1, 0.2],
            "volatility_threshold": [0.15, 0.25],
            "correlation_threshold": [0.15, 0.15],
            "regime": ["HV_LC", "HV_HC"],
        },
        index=pd.to_datetime(["2017-01-31", "2017-02-28"]),
    )
    dates = pd.bdate_range("2017-01-30", "2017-03-03")
    daily = map_monthly_regimes_to_daily(monthly, dates)
    assert daily.loc["2017-01-31", "regime"] == INSUFFICIENT_HISTORY
    assert daily.loc["2017-02-01", "regime"] == "HV_LC"
    assert daily.loc["2017-02-17", "effective_monthly_regime_date"] == pd.Timestamp("2017-01-31")
    assert daily.loc["2017-03-01", "regime"] == "HV_HC"
    audit = validate_target_regime("2017-02-17", daily, monthly)
    assert audit["regime"] == "HV_LC"
    assert audit["effective_monthly_regime_date"] < audit["target_date"]


def test_cta_returns_do_not_bridge_missing_months() -> None:
    nav = pd.DataFrame(
        {
            "date": pd.to_datetime(
                ["2020-01-30", "2020-01-31", "2020-02-28", "2020-04-30", "2020-01-31", "2020-02-28"]
            ),
            "product_id": ["CTA_A", "CTA_A", "CTA_A", "CTA_A", "CTA_B", "CTA_B"],
            "nav": [1.0, 1.1, 1.21, 1.5, 2.0, 2.2],
        }
    )
    monthly_nav, returns = compute_cta_monthly_returns(nav)
    assert monthly_nav.loc["2020-01-31", "CTA_A"] == pytest.approx(1.1)
    assert returns.loc["2020-02-29", "CTA_A"] == pytest.approx(0.1)
    assert np.isnan(returns.loc["2020-03-31", "CTA_A"])
    assert np.isnan(returns.loc["2020-04-30", "CTA_A"])

    regimes = pd.Series(
        ["LV_LC", "HV_LC", "HV_HC", "LV_HC"],
        index=pd.date_range("2020-01-31", periods=4, freq="ME"),
    )
    statistics = compute_cta_regime_statistics(returns, regimes)
    assert statistics.loc[("CTA_A", "HV_LC"), "n_months"] == 1
    cumulative = compute_regime_cumulative_nav(returns, regimes)
    assert np.isnan(cumulative.loc["2020-03-31", ("CTA_A", "HV_LC")])


def test_stage_summary_and_all_figures_smoke() -> None:
    dates = pd.bdate_range("2020-01-30", periods=6)
    monthly = pd.DataFrame(
        {
            "volatility": [0.1],
            "correlation": [0.2],
            "volatility_threshold": [0.15],
            "correlation_threshold": [0.15],
            "within_sector_correlation": [0.25],
            "across_sector_correlation": [0.15],
            "regime": ["LV_HC"],
        },
        index=[pd.Timestamp("2020-01-31")],
    )
    daily = map_monthly_regimes_to_daily(monthly, dates)
    nhci = pd.Series(np.arange(100.0, 106.0), index=dates)
    stages = compress_regime_stages(daily, nhci, monthly)
    assert stages["n_trading_days"].sum() == len(dates)

    cumulative = pd.DataFrame(
        {
            ("CTA_A", "LV_LC"): [1.0, 1.1],
            ("CTA_A", "HV_LC"): [1.0, 1.0],
            ("CTA_A", "LV_HC"): [1.0, 1.0],
            ("CTA_A", "HV_HC"): [1.0, 1.0],
        },
        index=pd.date_range("2020-01-31", periods=2, freq="ME"),
    )
    cumulative.columns.names = ["product_id", "regime"]
    environment_figure = plot_market_environment(
        monthly, "volatility", "volatility_threshold", "Figure 1", "Volatility"
    )
    assert environment_figure.axes[0].get_xlabel() == "日期"
    assert {line.get_label() for line in environment_figure.axes[0].lines} == {"年化波动率", "滚动中位数"}
    assert plot_market_environment(monthly, "correlation", "correlation_threshold", "Figure 2", "Correlation")
    assert plot_sector_correlation(monthly)
    river = plot_sector_volatility_river(
        pd.DataFrame(
            {
                "谷物": [0.5, 0.4],
                "黑色": [0.3, 0.35],
                "化工": [0.2, 0.25],
            },
            index=dates[:2],
        )
    )
    river_axes = river.axes[0]
    assert tuple(river.get_size_inches()) == pytest.approx((16.0, 10.0))
    assert river_axes.get_ylabel() == "贡献占比"
    assert river_axes.get_ylim() == pytest.approx((0.0, 1.0))
    assert [text.get_text() for text in river_axes.get_legend().get_texts()] == ["谷物", "黑色", "化工"]
    assert river_axes.get_xlim()[0] == matplotlib.dates.date2num(dates[0])
    combined_river = plot_sector_volatility_river(
        pd.DataFrame(
            {
                "谷物": [0.5, 0.4, 0.35],
                "黑色": [0.3, 0.35, np.nan],
                "化工": [0.2, 0.25, 0.65],
            },
            index=dates[:3],
        )
    )
    assert len(combined_river.axes) == 1
    assert combined_river.axes[0].get_xlim()[0] == matplotlib.dates.date2num(dates[0])
    assert combined_river.axes[0].get_xlim()[1] == matplotlib.dates.date2num(dates[2])
    assert [text.get_text() for text in combined_river.axes[0].get_legend().get_texts()] == ["谷物", "黑色", "化工"]
    figure = plot_nhci_regimes(nhci, stages)
    axes = figure.axes[0]
    assert axes.get_ylabel() == "指数点位"
    valid_start = stages.loc[stages["regime"].ne(INSUFFICIENT_HISTORY), "start_date"].min()
    assert axes.get_xlim()[0] == matplotlib.dates.date2num(valid_start)
    assert {line.get_label() for line in axes.lines} == {"南华商品指数"}
    assert "南华商品指数" not in {text.get_text() for text in axes.get_legend().get_texts()}
    fund_figure = plot_fund_vs_nhci(pd.Series([2.0, 2.2], index=dates[:2]), nhci, stages, "Fund A")
    assert {line.get_label() for line in fund_figure.axes[0].lines} == {"Fund A", "南华商品指数"}
    cumulative_figure = plot_regime_cumulative_returns(
        pd.DataFrame(
            {
                "LV_LC": [1.0, 1.1],
                "HV_LC": [1.0, 1.0],
                "LV_HC": [1.0, 1.0],
                "HV_HC": [1.0, 1.0],
            },
            index=dates[:2],
        )
    )
    assert len(cumulative_figure.axes) == 4
    assert {axis.get_xlabel() for axis in cumulative_figure.axes} == {""}
    assert [axis.get_ylabel() for axis in cumulative_figure.axes] == ["累计净值", "", "累计净值", ""]
    assert all(any(line.get_linestyle() == "--" for line in axis.lines) for axis in cumulative_figure.axes)
    assert all(axis.get_legend() is None for axis in cumulative_figure.axes)
    shared_cumulative = plot_regime_cumulative_returns(
        pd.DataFrame(
            {
                "LV_LC": [1.0, 1.1],
                "HV_LC": [1.0, 1.0],
                "LV_HC": [1.0, 1.0],
                "HV_HC": [1.0, 1.0],
            }
        ),
        sharex=True,
        sharey=True,
    )
    assert all(
        shared_cumulative.axes[0].get_shared_x_axes().joined(shared_cumulative.axes[0], axis)
        for axis in shared_cumulative.axes[1:]
    )
    assert all(
        shared_cumulative.axes[0].get_shared_y_axes().joined(shared_cumulative.axes[0], axis)
        for axis in shared_cumulative.axes[1:]
    )
    assert all(axis.get_yscale() == "linear" for axis in shared_cumulative.axes)
    assert plot_cta_regime_performance(cumulative, "CTA_A")


def test_environment_plots_start_at_first_valid_metric() -> None:
    index = pd.date_range("2020-01-31", periods=4, freq="ME")
    environment = pd.DataFrame(
        {
            "volatility": [np.nan, 0.2, 0.3, 0.4],
            "correlation": [np.nan, 0.1, 0.2, 0.3],
            "volatility_threshold": [np.nan, np.nan, 0.2, 0.2],
            "correlation_threshold": [np.nan, np.nan, 0.1, 0.1],
        },
        index=index,
    )

    figure = plot_market_environment(environment, "volatility", "volatility_threshold", "图1", "波动率")

    assert figure.axes[0].get_xlim()[0] == matplotlib.dates.date2num(index[1])
