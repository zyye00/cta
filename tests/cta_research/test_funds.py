from pathlib import Path

import pandas as pd
import pytest
import yaml

from cta_research import load_fund_histories
from cta_research.funds import _normalize_fund_categories


def test_load_fund_histories_discovers_excel_and_records_exclusions(tmp_path: Path) -> None:
    nhci_path = tmp_path / "NHCI.xlsx"
    fund_path = tmp_path / "基金甲净值20260803.xlsx"
    invalid_path = tmp_path / "说明.xlsx"
    dates = pd.date_range("2020-01-01", periods=3, freq="D")
    pd.DataFrame({"日期": dates, "指数点位": [100, 101, 102]}).to_excel(nhci_path, index=False)
    pd.DataFrame({"日期": dates[::-1], "单位净值": [1.02, 1.01, 1.0]}).to_excel(fund_path, index=False)
    pd.DataFrame({"说明": ["not a fund"]}).to_excel(invalid_path, index=False)

    result = load_fund_histories(tmp_path, exclude_paths=[nhci_path])

    assert list(result.nav.columns) == ["基金 01"]
    assert result.nav.index.is_monotonic_increasing
    assert set(result.manifest["status"]) == {"accepted", "excluded"}
    assert result.manifest.loc[result.manifest["reason"].eq("configured non-fund input"), "reason"].item() == (
        "configured non-fund input"
    )
    mapping = yaml.safe_load((tmp_path / "fund_name_mapping.yaml").read_text(encoding="utf-8"))
    assert mapping == {"基金甲": "基金 01"}

    second_fund = tmp_path / "基金乙净值20260803.xlsx"
    pd.DataFrame({"日期": dates, "单位净值": [1.0, 1.01, 1.02]}).to_excel(second_fund, index=False)
    rerun = load_fund_histories(tmp_path, exclude_paths=[nhci_path])

    assert list(rerun.nav.columns) == ["基金 01", "基金 02"]
    manifest_text = " ".join(rerun.manifest.astype("string").fillna("").to_numpy().ravel())
    assert "基金甲" not in manifest_text
    assert "基金乙" not in manifest_text


def test_fund_categories_validate_and_are_added_to_manifest(tmp_path: Path) -> None:
    dates = pd.date_range("2020-01-01", periods=3, freq="D")
    fund_path = tmp_path / "基金甲净值20260803.xlsx"
    pd.DataFrame({"日期": dates, "单位净值": [1.0, 1.01, 1.02]}).to_excel(fund_path, index=False)

    result = load_fund_histories(
        tmp_path,
        fund_categories={"截面": ["基金 01"], "趋势": ["基金 02"]},
    )

    assert result.manifest.loc[result.manifest["status"].eq("accepted"), "category"].item() == "截面"
    assert _normalize_fund_categories({"截面": ["基金 01"], "趋势": ["基金 02"]}) == {
        "截面": ("基金 01",),
        "趋势": ("基金 02",),
    }
    with pytest.raises(ValueError, match="multiple categories"):
        _normalize_fund_categories({"截面": ["基金 01"], "趋势": ["基金 01"]})


def test_loaded_fund_missing_category_fails(tmp_path: Path) -> None:
    dates = pd.date_range("2020-01-01", periods=3, freq="D")
    pd.DataFrame({"日期": dates, "单位净值": [1.0, 1.01, 1.02]}).to_excel(
        tmp_path / "基金甲净值20260803.xlsx", index=False
    )
    with pytest.raises(ValueError, match="missing from fund_categories"):
        load_fund_histories(tmp_path, fund_categories={"截面": ["基金 02"]})
