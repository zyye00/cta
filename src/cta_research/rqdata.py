from __future__ import annotations

from pathlib import Path

import pandas as pd

DATE_ALIASES = ("date", "交易日期", "日期", "trade_date")
SYMBOL_ALIASES = ("underlying_symbol", "index_code", "order_book_id", "symbol", "代码")
COMMODITY_EXCHANGES = {"SHFE", "DCE", "CZCE", "INE", "GFEX"}


def _find_column(columns: pd.Index, aliases: tuple[str, ...], label: str) -> str:
    normalized = {str(column).strip().lower(): column for column in columns}
    for alias in aliases:
        if alias.lower() in normalized:
            return str(normalized[alias.lower()])
    raise ValueError(f"Cannot identify the {label} column. Available columns: {list(columns)}")


def normalize_rqdata_prices(raw: pd.DataFrame, symbol_column: str | None = None) -> pd.DataFrame:
    """Normalize RQData price output while retaining all raw fields."""
    if not isinstance(raw, pd.DataFrame) or raw.empty:
        raise ValueError("RQData price output is empty or is not a DataFrame.")
    data = raw.reset_index() if isinstance(raw.index, pd.MultiIndex) else raw.copy()
    symbol_column = symbol_column or _find_column(data.columns, SYMBOL_ALIASES, "symbol")
    date_column = _find_column(data.columns, DATE_ALIASES, "date")
    if "close" not in data.columns:
        raise ValueError(f"RQData price output has no close field: {list(data.columns)}")
    data = data.rename(columns={symbol_column: "underlying_symbol", date_column: "date"})
    data["underlying_symbol"] = data["underlying_symbol"].astype("string").str.strip().str.upper()
    data["date"] = pd.to_datetime(data["date"], errors="coerce")
    data["close"] = pd.to_numeric(data["close"], errors="coerce")
    data = data.dropna(subset=["underlying_symbol", "date"]).sort_values(["underlying_symbol", "date"])
    if data.duplicated(["underlying_symbol", "date"]).any():
        raise ValueError("RQData price output contains duplicate symbol-date observations.")
    return data.reset_index(drop=True)


def group_incremental_requests(
    symbols: list[str],
    cached: pd.DataFrame,
    start_date: str | pd.Timestamp,
    end_date: str | pd.Timestamp,
) -> dict[str, list[str]]:
    """Group symbols by the first date missing from their cached price history."""
    start = pd.Timestamp(start_date).normalize()
    end = pd.Timestamp(end_date).normalize()
    if end < start:
        return {}
    latest_dates = pd.Series(dtype="datetime64[ns]")
    if not cached.empty:
        required = {"underlying_symbol", "date"}
        if missing := required.difference(cached.columns):
            raise ValueError(f"Cached price history is missing columns: {sorted(missing)}")
        cache_symbols = cached["underlying_symbol"].astype("string").str.strip().str.upper()
        cache_dates = pd.to_datetime(cached["date"], errors="coerce")
        latest_dates = cache_dates.groupby(cache_symbols).max()
    groups: dict[str, list[str]] = {}
    for symbol in sorted({str(symbol).strip().upper() for symbol in symbols}):
        latest = latest_dates.get(symbol)
        request_start = start if pd.isna(latest) else max(start, latest + pd.Timedelta(days=1))
        if request_start <= end:
            groups.setdefault(request_start.date().isoformat(), []).append(symbol)
    return groups


def merge_rqdata_price_updates(cached: pd.DataFrame, updates: pd.DataFrame) -> pd.DataFrame:
    """Merge normalized RQData updates, keeping the newest row for each symbol-date."""
    if updates.empty:
        return cached.reset_index(drop=True)
    merged = pd.concat([cached, updates], ignore_index=True)
    merged = merged.drop_duplicates(["underlying_symbol", "date"], keep="last")
    return merged.sort_values(["underlying_symbol", "date"]).reset_index(drop=True)


def select_commodity_instruments(
    future_instruments: pd.DataFrame,
    exchanges: set[str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split RQData futures metadata into domestic commodities and exclusions."""
    required = {"underlying_symbol", "exchange"}
    if missing := required.difference(future_instruments.columns):
        raise ValueError(f"Future metadata is missing columns: {sorted(missing)}")
    exchanges = exchanges or COMMODITY_EXCHANGES
    data = future_instruments.copy()
    data["underlying_symbol"] = data["underlying_symbol"].astype("string").str.strip().str.upper()
    data["exchange"] = data["exchange"].astype("string").str.strip().str.upper()
    exchanges = {str(exchange).strip().upper() for exchange in exchanges}
    valid_symbol = data["underlying_symbol"].notna() & data["underlying_symbol"].ne("")
    valid_exchange = data["exchange"].isin(exchanges)
    reasons = pd.Series("non_commodity_exchange", index=data.index, dtype="string")
    reasons.loc[valid_exchange & valid_symbol] = pd.NA
    reasons.loc[~valid_symbol] = "missing_symbol"
    selected_mask = reasons.isna()
    selected = data.loc[selected_mask].copy()
    excluded = data.loc[~selected_mask].assign(exclusion_reason=reasons.loc[~selected_mask]).copy()
    return selected.reset_index(drop=True), excluded.reset_index(drop=True)


def save_rqdata_raw(
    output_dir: str | Path,
    future_instruments: pd.DataFrame,
    dominant_prices: pd.DataFrame,
    index_99_prices: pd.DataFrame | None = None,
) -> dict[str, Path]:
    """Persist RQData inputs with a parquet round-trip check."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    frames: dict[str, pd.DataFrame] = {
        "future_instruments": future_instruments,
        "dominant_prices": dominant_prices,
    }
    if index_99_prices is not None:
        frames["index_99_prices"] = index_99_prices
    paths: dict[str, Path] = {}
    for name, frame in frames.items():
        path = output_dir / f"{name}.parquet"
        frame.to_parquet(path, index=False)
        restored = pd.read_parquet(path)
        if len(restored) != len(frame) or list(restored.columns) != list(frame.columns):
            raise RuntimeError(f"Parquet round-trip verification failed for {path}.")
        paths[name] = path
    return paths
