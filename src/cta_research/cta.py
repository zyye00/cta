from __future__ import annotations

import numpy as np
import pandas as pd

from .regimes import FORMAL_REGIMES, INSUFFICIENT_HISTORY


def normalize_cta_nav(cta_data: pd.DataFrame) -> pd.DataFrame:
    """Normalize supported long or wide CTA NAV data to date-product-nav long form."""
    if cta_data.empty:
        raise ValueError("CTA NAV data is empty.")
    columns = {str(column).lower(): column for column in cta_data.columns}
    date_column = columns.get("date")
    if date_column is None:
        raise ValueError("CTA NAV data must contain a `date` column.")
    if {"product_id", "nav"}.issubset(columns):
        data = cta_data[[date_column, columns["product_id"], columns["nav"]]].copy()
        data.columns = ["date", "product_id", "nav"]
    else:
        value_columns = [column for column in cta_data.columns if column != date_column]
        if not value_columns:
            raise ValueError("Wide CTA NAV data has no product columns.")
        data = cta_data.melt(id_vars=date_column, value_vars=value_columns, var_name="product_id", value_name="nav")
        data = data.rename(columns={date_column: "date"})
    data["date"] = pd.to_datetime(data["date"], errors="coerce")
    data["product_id"] = data["product_id"].astype("string").str.strip()
    data["nav"] = pd.to_numeric(data["nav"], errors="coerce")
    data = data.dropna().sort_values(["product_id", "date"])
    if data["nav"].le(0).any():
        raise ValueError("CTA NAV values must be positive.")
    return data.drop_duplicates(["date", "product_id"], keep="last").reset_index(drop=True)


def compute_cta_monthly_returns(cta_nav: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Take each product's last monthly NAV and calculate returns without bridging missing months."""
    data = normalize_cta_nav(cta_nav)
    data["month"] = data["date"].dt.to_period("M").dt.to_timestamp("M")
    monthly_long = data.groupby(["month", "product_id"], observed=True, as_index=False).last()
    monthly_nav = monthly_long.pivot(index="month", columns="product_id", values="nav").sort_index()
    full_index = pd.date_range(monthly_nav.index.min(), monthly_nav.index.max(), freq="ME")
    monthly_nav = monthly_nav.reindex(full_index)
    monthly_nav.index.name = "date"
    return monthly_nav, monthly_nav.pct_change(fill_method=None)


def compute_cta_regime_statistics(
    monthly_returns: pd.DataFrame,
    monthly_regimes: pd.Series,
    annualization_months: int = 12,
) -> pd.DataFrame:
    """Compute separate state-conditional performance statistics for each CTA product."""
    regimes = monthly_regimes.copy()
    regimes.index = pd.to_datetime(regimes.index).to_period("M").to_timestamp("M")
    aligned = monthly_returns.join(regimes.rename("regime"), how="inner")
    rows: list[dict[str, object]] = []
    for product in monthly_returns.columns:
        for regime in FORMAL_REGIMES:
            values = aligned.loc[aligned["regime"].eq(regime), product].dropna()
            standard_deviation = values.std(ddof=1)
            annualized_return = (
                (1 + values).prod() ** (annualization_months / len(values)) - 1
                if len(values) and values.gt(-1).all()
                else np.nan
            )
            rows.append(
                {
                    "product_id": product,
                    "regime": regime,
                    "n_months": len(values),
                    "mean_monthly_return": values.mean(),
                    "annualized_return": annualized_return,
                    "annualized_volatility": standard_deviation * np.sqrt(annualization_months),
                    "annualized_sharpe": (
                        values.mean() / standard_deviation * np.sqrt(annualization_months)
                        if len(values) >= 2 and standard_deviation > 0
                        else np.nan
                    ),
                }
            )
    return pd.DataFrame(rows).set_index(["product_id", "regime"])


def compute_regime_cumulative_nav(
    monthly_returns: pd.DataFrame,
    monthly_regimes: pd.Series,
) -> pd.DataFrame:
    """Build state-attributed cumulative NAV while preserving missing CTA months."""
    regimes = monthly_regimes.copy()
    regimes.index = pd.to_datetime(regimes.index).to_period("M").to_timestamp("M")
    aligned = monthly_returns.join(regimes.rename("regime"), how="inner")
    outputs: dict[tuple[str, str], pd.Series] = {}
    for product in monthly_returns.columns:
        valid = aligned[product].notna() & aligned["regime"].ne(INSUFFICIENT_HISTORY)
        for regime in FORMAL_REGIMES:
            contribution = aligned[product].where(valid & aligned["regime"].eq(regime), 0.0)
            contribution = contribution.where(valid)
            outputs[(str(product), regime)] = (1 + contribution).cumprod()
    result = pd.DataFrame(outputs)
    result.columns = pd.MultiIndex.from_tuples(result.columns, names=["product_id", "regime"])
    return result
