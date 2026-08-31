from __future__ import annotations

import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import yaml

from .histories import normalize_fund_nav_history, read_history_file

FUND_ALIAS_PATTERN = re.compile(r"^基金 (\d+)$")


def _load_fund_name_mapping(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    with path.open(encoding="utf-8") as file:
        data = yaml.safe_load(file)
    if data is None:
        return {}
    valid_mapping = isinstance(data, dict) and all(
        isinstance(key, str) and isinstance(value, str) for key, value in data.items()
    )
    if not valid_mapping:
        raise ValueError("Fund name mapping must be a string-to-string YAML mapping.")
    mapping = {str(key): str(value) for key, value in data.items()}
    if len(mapping) != len(set(mapping.values())):
        raise ValueError("Fund name mapping contains duplicate aliases.")
    return mapping


def _save_fund_name_mapping(path: Path, mapping: dict[str, str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        yaml.safe_dump(mapping, file, allow_unicode=True, sort_keys=True)


def _next_fund_alias(mapping: dict[str, str]) -> str:
    numbers = [int(match.group(1)) for alias in mapping.values() if (match := FUND_ALIAS_PATTERN.match(alias))]
    return f"基金 {max(numbers, default=0) + 1:02d}"


def _normalize_fund_categories(categories: Mapping[str, Iterable[str]]) -> dict[str, tuple[str, ...]]:
    if not isinstance(categories, Mapping) or not categories:
        raise ValueError("fund_categories must be a non-empty mapping.")
    normalized: dict[str, tuple[str, ...]] = {}
    owners: dict[str, str] = {}
    for category, funds in categories.items():
        if not isinstance(category, str) or not category.strip() or isinstance(funds, (str, bytes)):
            raise ValueError("Each fund category must map to a non-empty list of aliases.")
        members = tuple(str(fund).strip() for fund in funds)
        if not members or any(not fund for fund in members):
            raise ValueError("Each fund category must map to a non-empty list of aliases.")
        for fund in members:
            if fund in owners:
                raise ValueError(f"Fund alias {fund!r} appears in multiple categories.")
            owners[fund] = category
        normalized[category] = members
    return normalized


@dataclass(frozen=True)
class FundHistoryResult:
    """Normalized fund NAVs and a manifest of accepted or excluded files."""

    nav: pd.DataFrame
    manifest: pd.DataFrame


def load_fund_histories(
    raw_dir: str | Path,
    exclude_paths: Iterable[str | Path] | str | Path = (),
    fund_categories: Mapping[str, Iterable[str]] | None = None,
) -> FundHistoryResult:
    """Discover and normalize private-fund NAV files using local stable aliases."""
    raw_dir = Path(raw_dir)
    categories = _normalize_fund_categories(fund_categories) if fund_categories is not None else None
    category_by_fund = (
        {fund: category for category, members in categories.items() for fund in members} if categories else {}
    )
    mapping_path = raw_dir / "fund_name_mapping.yaml"
    name_mapping = _load_fund_name_mapping(mapping_path)
    mapping_changed = False
    paths = [exclude_paths] if isinstance(exclude_paths, (str, Path)) else exclude_paths
    excluded = {Path(path).resolve() for path in paths}
    curves: dict[str, pd.Series] = {}
    manifest: list[dict[str, object]] = []
    for path in sorted(raw_dir.iterdir() if raw_dir.exists() else (), key=lambda item: item.name):
        if not path.is_file() or path.suffix.lower() not in {".xls", ".xlsx"}:
            continue
        if path.resolve() in excluded:
            manifest.append({"status": "excluded", "reason": "configured non-fund input"})
            continue
        try:
            history = normalize_fund_nav_history(read_history_file(path))
        except (ValueError, TypeError) as error:
            manifest.append({"status": "excluded", "reason": str(error)})
            continue
        fund_name = path.stem.rsplit("净值", maxsplit=1)[0] or path.stem
        if fund_name in curves:
            fund_name = path.stem
        if fund_name not in name_mapping:
            name_mapping[fund_name] = _next_fund_alias(name_mapping)
            mapping_changed = True
        alias = name_mapping[fund_name]
        if categories and alias not in category_by_fund:
            raise ValueError(f"Loaded fund alias {alias!r} is missing from fund_categories.")
        series = history.set_index("date")["value"].rename(alias)
        curves[alias] = series
        manifest.append(
            {
                "fund": alias,
                "category": category_by_fund.get(alias),
                "status": "accepted",
                "start_date": history["date"].min(),
                "end_date": history["date"].max(),
                "observations": len(history),
            }
        )
    if mapping_changed:
        _save_fund_name_mapping(mapping_path, name_mapping)
    nav = pd.concat(curves, axis=1, sort=True).sort_index(axis=1) if curves else pd.DataFrame()
    return FundHistoryResult(nav=nav, manifest=pd.DataFrame(manifest))
