from __future__ import annotations

from pathlib import Path

import pandas as pd

DATE_ALIASES = ("date", "交易日期", "日期", "trade_date")
NHCI_VALUE_ALIASES = ("close", "value", "指数", "指数值", "指数点位", "收盘", "收盘价")
FUND_NAV_VALUE_ALIASES = ("复权净值", "累计净值", "单位净值", "净值", "nav", "value")


def _find_column(columns: pd.Index, aliases: tuple[str, ...], label: str) -> str:
    normalized = {str(column).strip().lower(): column for column in columns}
    for alias in aliases:
        if alias.lower() in normalized:
            return str(normalized[alias.lower()])
    raise ValueError(f"Cannot identify the {label} column. Available columns: {list(columns)}")


def read_history_file(path: str | Path) -> pd.DataFrame:
    """Read a supported history file without changing its columns."""
    path = Path(path)
    readers = {
        ".csv": pd.read_csv,
        ".parquet": pd.read_parquet,
        ".xls": pd.read_excel,
        ".xlsx": pd.read_excel,
    }
    reader = readers.get(path.suffix.lower())
    if reader is None:
        raise ValueError(f"Unsupported input file type: {path.suffix}")
    return reader(path)


def normalize_nhci_history(
    raw: pd.DataFrame,
    start_date: str | pd.Timestamp,
    end_date: str | pd.Timestamp | None = None,
) -> pd.DataFrame:
    """Normalize a user-supplied NHCI history to date/close without analysis."""
    if not isinstance(raw, pd.DataFrame) or raw.empty:
        raise ValueError("External NHCI input is empty or is not a DataFrame.")
    date_column = _find_column(raw.columns, DATE_ALIASES, "date")
    value_column = _find_column(raw.columns, NHCI_VALUE_ALIASES, "close")
    history = raw[[date_column, value_column]].rename(columns={date_column: "date", value_column: "close"})
    history["date"] = pd.to_datetime(history["date"], errors="coerce")
    history["close"] = pd.to_numeric(history["close"], errors="coerce")
    history = history.dropna().sort_values("date")
    history = history.loc[history["date"].ge(pd.Timestamp(start_date))]
    if end_date:
        history = history.loc[history["date"].le(pd.Timestamp(end_date))]
    if history.empty:
        raise ValueError("External NHCI input has no rows in the configured date range.")
    if history.duplicated("date").any():
        raise ValueError("External NHCI input contains duplicate dates.")
    if history["close"].le(0).any():
        raise ValueError("External NHCI input contains non-positive close values.")
    return history.reset_index(drop=True)


def normalize_fund_nav_history(raw: pd.DataFrame) -> pd.DataFrame:
    """Normalize a private-fund NAV workbook to ``date`` and ``value``."""
    if not isinstance(raw, pd.DataFrame) or raw.empty:
        raise ValueError("Fund NAV input is empty or is not a DataFrame.")
    date_column = _find_column(raw.columns, DATE_ALIASES, "date")
    value_column = _find_column(raw.columns, FUND_NAV_VALUE_ALIASES, "fund NAV")
    history = raw[[date_column, value_column]].rename(columns={date_column: "date", value_column: "value"})
    history["date"] = pd.to_datetime(history["date"], errors="coerce")
    history["value"] = pd.to_numeric(history["value"], errors="coerce")
    history = history.dropna().drop_duplicates("date", keep="last").sort_values("date")
    if history.empty or history["value"].le(0).any():
        raise ValueError("Fund NAV input has no valid positive observations.")
    return history.reset_index(drop=True)
