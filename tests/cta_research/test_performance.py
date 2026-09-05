import numpy as np
import pandas as pd
import pytest

from cta_research import compute_regime_cumulative_returns, compute_series_regime_performance
from cta_research.performance import compute_regime_annualized_returns


def test_series_regime_performance_uses_explicit_frequency_and_excludes_history() -> None:
    dates = pd.date_range("2020-01-01", periods=6, freq="D")
    values = pd.Series([100.0, 110.0, 121.5, 121.5, 121.5, 121.5], index=dates, name="NHCI")
    daily_regimes = pd.DataFrame(
        {"regime": ["INSUFFICIENT_HISTORY", "LV_LC", "LV_LC", "HV_HC", "HV_HC", "HV_HC"]},
        index=dates,
    )

    result = compute_series_regime_performance(values, daily_regimes, periods_per_year=252)

    assert result.loc["LV_LC", "n_observations"] == 2
    assert result.loc["LV_LC", "annualized_return"] == pytest.approx(1.215**126 - 1)
    assert pd.notna(result.loc["LV_LC", "annualized_sharpe"])
    assert np.isnan(result.loc["HV_HC", "annualized_sharpe"])
    assert "INSUFFICIENT_HISTORY" not in result.index


def test_compute_regime_annualized_returns_excludes_insufficient_history() -> None:
    dates = pd.date_range("2020-01-01", periods=5, freq="7D")
    nav = pd.DataFrame(
        {
            "Fund": [1.0, 1.1, 1.21, 1.331, 1.5],
            "NHCI": [1.0, 0.9, 0.99, 1.089, 1.1979],
            "Flat": [1.0, 1.0, 1.0, 1.0, 1.0],
        },
        index=dates,
    )
    daily_regimes = pd.DataFrame(
        {"regime": ["INSUFFICIENT_HISTORY", "LV_LC", "LV_LC", "HV_HC", "HV_HC"]}, index=dates
    )

    result = compute_regime_annualized_returns(nav, daily_regimes)

    assert result.loc[("LV_LC", "Fund"), "n_observations"] == 2
    assert result.loc[("HV_HC", "NHCI"), "n_observations"] == 2
    assert result.loc[("HV_LC", "Fund"), "n_observations"] == 0
    assert np.isnan(result.loc[("HV_LC", "Fund"), "annualized_return"])
    assert {"annualized_volatility", "annualized_sharpe"}.issubset(result.columns)
    assert pd.notna(result.loc[("HV_HC", "Fund"), "annualized_sharpe"])
    assert np.isnan(result.loc[("HV_HC", "Flat"), "annualized_sharpe"])


@pytest.mark.parametrize(
    ("frequency", "periods_per_year"),
    [("D", 252.0), ("7D", 52.0), ("30D", 12.0), ("90D", 365.25 / 90)],
)
def test_series_regime_performance_infers_standard_frequency(
    frequency: str, periods_per_year: float
) -> None:
    dates = pd.date_range("2020-01-01", periods=4, freq=frequency)
    values = pd.Series(1.01 ** np.arange(4), index=dates, name="Fund")
    daily_regimes = pd.DataFrame({"regime": "LV_LC"}, index=dates)

    result = compute_series_regime_performance(values, daily_regimes)

    assert result.loc["LV_LC", "annualized_return"] == pytest.approx(1.01**periods_per_year - 1)


def test_aligned_series_share_one_inferred_annualization_frequency() -> None:
    dates = pd.date_range("2020-01-01", periods=6, freq="7D")
    values = pd.DataFrame(
        {
            "Fund": 1.01 ** np.arange(6),
            "NHCI": 1.01 ** np.arange(6),
        },
        index=dates,
    )
    daily_regimes = pd.DataFrame({"regime": "LV_LC"}, index=dates)

    result = compute_regime_annualized_returns(values, daily_regimes)

    assert result.loc[("LV_LC", "Fund"), "annualized_return"] == pytest.approx(
        result.loc[("LV_LC", "NHCI"), "annualized_return"]
    )
    assert result.loc[("LV_LC", "Fund"), "annualized_sharpe"] == pytest.approx(
        result.loc[("LV_LC", "NHCI"), "annualized_sharpe"]
    )


def test_compute_regime_cumulative_returns_keeps_only_active_state_returns() -> None:
    dates = pd.date_range("2020-01-01", periods=5, freq="D")
    values = pd.Series([100.0, 110.0, 121.0, 100.0, 105.0], index=dates)
    daily_regimes = pd.DataFrame(
        {"regime": ["INSUFFICIENT_HISTORY", "LV_LC", "HV_HC", "LV_LC", "HV_HC"]},
        index=dates,
    )

    result = compute_regime_cumulative_returns(values, daily_regimes)

    assert result.loc[dates[1], "LV_LC"] == 1.1
    assert result.loc[dates[3], "LV_LC"] == 1.1 * (100 / 121)
    assert result.loc[dates[2], "HV_HC"] == 1.1
    assert "INSUFFICIENT_HISTORY" not in result.columns
