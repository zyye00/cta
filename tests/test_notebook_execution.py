import json
from io import StringIO
from pathlib import Path
from shutil import copy, copytree

import nbformat
import numpy as np
import pandas as pd
import yaml
from nbconvert.preprocessors import ExecutePreprocessor


def _read_saved_html_tables(notebook_path: str) -> list[pd.DataFrame]:
    notebook = nbformat.read(notebook_path, as_version=4)
    tables: list[pd.DataFrame] = []
    for cell in notebook.cells:
        for output in cell.get("outputs", []):
            html = output.get("data", {}).get("text/html")
            if html:
                tables.extend(pd.read_html(StringIO("".join(html) if isinstance(html, list) else html)))
    return tables


def _write_fake_rqdatac(project: Path) -> None:
    (project / "rqdatac.py").write_text(
        '''import json
from pathlib import Path

import pandas as pd

CALLS_PATH = Path("rqdatac_calls.json")
CODES = ["A", "AG", "AL", "AU", "C", "CU", "M", "RB"]


def _record(call: dict[str, object]) -> None:
    calls = json.loads(CALLS_PATH.read_text()) if CALLS_PATH.exists() else []
    calls.append(call)
    CALLS_PATH.write_text(json.dumps(calls), encoding="utf-8")


def init() -> None:
    _record({"kind": "init"})


def all_instruments(type: str) -> pd.DataFrame:
    _record({"kind": "all_instruments", "type": type})
    return pd.DataFrame({"underlying_symbol": CODES, "exchange": "SHFE"})


class _Futures:
    def get_dominant_price(self, symbols: list[str] | str, **kwargs: object) -> pd.DataFrame:
        symbols = [symbols] if isinstance(symbols, str) else symbols
        _record({"kind": "dominant", "symbols": symbols, **kwargs})
        dates = pd.bdate_range(kwargs["start_date"], kwargs["end_date"])
        index = pd.MultiIndex.from_product([symbols, dates], names=["underlying_symbol", "date"])
        return pd.DataFrame({"close": range(1, len(index) + 1)}, index=index)


futures = _Futures()


def get_price(ids: list[str] | str, **kwargs: object) -> pd.DataFrame:
    ids = [ids] if isinstance(ids, str) else ids
    _record({"kind": "index_99", "ids": ids, **kwargs})
    dates = pd.bdate_range(kwargs["start_date"], kwargs["end_date"])
    index = pd.MultiIndex.from_product([ids, dates], names=["order_book_id", "date"])
    return pd.DataFrame({"close": range(1, len(index) + 1)}, index=index)
''',
        encoding="utf-8",
    )


def test_download_notebook_resolves_explicit_rqdata_end_date() -> None:
    notebook = nbformat.read("notebooks/01_download.ipynb", as_version=4)
    source = "\n".join(cell.source for cell in notebook.cells if cell.cell_type == "code")
    assert 'END_DATE = config["end_date"] or date.today().isoformat()' in source
    assert source.count('"start_date": request_start') == 2
    assert source.count('"end_date": END_DATE') == 2


def test_report_tables_match_saved_notebook_outputs() -> None:
    report = Path("REPORT.md").read_text(encoding="utf-8")
    nhci_table = next(
        table
        for table in _read_saved_html_tables("notebooks/02_analysis.ipynb")
        if {"样本数", "年化收益", "夏普比率"}.issubset(table.columns)
    )
    for row in nhci_table.itertuples(index=False, name=None):
        state, observations, annualized_return, sharpe = row
        assert f"| {state} | {observations} | {annualized_return} | {sharpe:.2f} |" in report

    config = yaml.safe_load(Path("config/config.yaml").read_text(encoding="utf-8"))
    categories = {
        fund: category for category, funds in config["fund_categories"].items() for fund in funds
    }
    fund_tables = [
        table
        for table in _read_saved_html_tables("notebooks/03_fund_analysis.ipynb")
        if any(str(column).startswith("基金 ") and str(column).endswith(" 夏普") for column in table.columns)
    ]
    assert len(fund_tables) == len(categories)
    for table in fund_tables:
        sharpe_column = next(
            str(column)
            for column in table.columns
            if str(column).startswith("基金 ") and str(column).endswith(" 夏普")
        )
        fund = sharpe_column.removesuffix(" 夏普")
        sharpes = ["—" if pd.isna(value) else f"{value:.2f}" for value in table[sharpe_column]]
        assert f"| {categories[fund]}—{fund} | {' | '.join(sharpes)} |" in report


def test_notebooks_execute_top_to_bottom_with_synthetic_data(tmp_path: Path) -> None:
    project = tmp_path / "cta"
    raw_dir = project / "data"
    raw_dir.mkdir(parents=True)
    (project / "notebooks").mkdir()
    copytree("config", project / "config")
    copytree("src", project / "src")
    config_path = project / "config" / "config.yaml"
    config_path.write_text(
        config_path.read_text().replace("end_date: null", 'end_date: "2019-01-02"'), encoding="utf-8"
    )
    _write_fake_rqdatac(project)
    for notebook_name in (
        "01_download.ipynb",
        "02_analysis.ipynb",
        "03_fund_analysis.ipynb",
        "04_sector_volatility_contributions.ipynb",
    ):
        copy(Path("notebooks") / notebook_name, project / "notebooks" / notebook_name)

    dates = pd.bdate_range("2011-07-01", "2018-12-31")
    codes = ["A", "AG", "AL", "AU", "C", "CU", "M", "RB"]
    generator = np.random.default_rng(7)
    common = generator.normal(0.0001, 0.005, len(dates))
    histories = []
    for position, code in enumerate(codes):
        returns = common * (0.25 + position / 20) + generator.normal(0.0001, 0.008, len(dates))
        histories.append(
            pd.DataFrame(
                {
                    "date": dates,
                    "underlying_symbol": code,
                    "close": 1_000 * np.exp(np.cumsum(returns)),
                }
            )
        )
    variety_indices = pd.concat(histories, ignore_index=True)
    nhci_returns = common + generator.normal(0.0001, 0.003, len(dates))
    nhci = pd.DataFrame({"date": dates, "close": 1_000 * np.exp(np.cumsum(nhci_returns))})
    future_instruments = pd.DataFrame(
        {
            "underlying_symbol": codes,
            "exchange": "SHFE",
        }
    )
    index_99 = variety_indices.copy()
    index_99["underlying_symbol"] = index_99["underlying_symbol"] + "99"
    initial_rows = len(variety_indices)
    future_instruments.to_parquet(raw_dir / "future_instruments.parquet", index=False)
    variety_indices.to_parquet(raw_dir / "dominant_prices.parquet", index=False)
    index_99.to_parquet(raw_dir / "index_99_prices.parquet", index=False)
    nhci.rename(columns={"date": "日期", "close": "指数点位"}).assign(
        涨跌幅="0.00%"
    ).to_excel(raw_dir / "NHCI-20260803.xlsx", index=False)
    for name, start, scale in (("基金甲", "2017-01-03", 1.0), ("基金乙", "2018-01-02", 1.5)):
        fund_dates = pd.bdate_range(start, "2019-01-02")
        pd.DataFrame(
            {
                "日期": fund_dates[::-1],
                "单位净值": scale * np.exp(np.linspace(0, 0.2, len(fund_dates)))[::-1],
                "累计净值": scale * np.exp(np.linspace(0, 0.2, len(fund_dates)))[::-1],
                "复权净值": scale * np.exp(np.linspace(0, 0.2, len(fund_dates)))[::-1],
                "涨跌幅": "0.00%",
            }
        ).to_excel(raw_dir / f"{name}净值20260803.xlsx", index=False)

    for notebook_name in (
        "01_download.ipynb",
        "02_analysis.ipynb",
        "03_fund_analysis.ipynb",
        "04_sector_volatility_contributions.ipynb",
    ):
        notebook = nbformat.read(project / "notebooks" / notebook_name, as_version=4)
        processor = ExecutePreprocessor(timeout=180, kernel_name="python3")
        processor.preprocess(notebook, {"metadata": {"path": str(project)}})
        errors = [
            output
            for cell in notebook.cells
            if cell.cell_type == "code"
            for output in cell.get("outputs", [])
            if output.output_type == "error"
        ]
        assert not errors
        if notebook_name == "03_fund_analysis.ipynb":
            serialized = json.dumps(notebook, ensure_ascii=False)
            assert "基金甲" not in serialized
            assert "基金乙" not in serialized
            assert "基金 01" in serialized
        if notebook_name == "02_analysis.ipynb":
            source = "\n".join(cell.source for cell in notebook.cells if cell.cell_type == "code")
            assert source.index("sys.path.insert") < source.index("from cta_research import")
            assert "Path(cta_research.__file__).resolve().is_relative_to" in source
            assert 'nhci_table.columns = ["样本数", "年化收益", "夏普比率"]' in source
            assert "compute_monthly_sector_environment" not in source
            assert "compute_cta" not in source
            assert "sharex=True" in source
            assert "sharey=True" in source
        if notebook_name == "03_fund_analysis.ipynb":
            source = "\n".join(cell.source for cell in notebook.cells if cell.cell_type == "code")
            assert source.index("sys.path.insert") < source.index("from cta_research import")
            assert "Path(cta_research.__file__).resolve().is_relative_to" in source
            assert "plot_fund_vs_nhci" in source
            assert "plot_market_environment" not in source
            assert "compute_cta" not in source
            assert "截面类产品" in serialized
            assert "趋势类产品" in serialized
            assert serialized.index("截面类产品") < serialized.index("趋势类产品")
            assert 'fund_categories = config["fund_categories"]' in source
            assert 'category": category' in source
            assert "compute_regime_annualized_returns" in source
            assert "periods_per_year=config" not in source
        if notebook_name == "04_sector_volatility_contributions.ipynb":
            source = "\n".join(cell.source for cell in notebook.cells if cell.cell_type == "code")
            assert source.index("sys.path.insert") < source.index("from cta_research import")
            assert "Path(cta_research.__file__).resolve().is_relative_to" in source
            assert "compute_sector_volatility_contribution_shares" in source
            assert "plot_sector_volatility_river" in source
    formal_config = yaml.safe_load(
        (project / "config" / "config.yaml").read_text(encoding="utf-8")
    )
    assert formal_config["environment_lookback_months"] == 3
    assert formal_config["aggregation_method"] == "sector_equal"
    assert formal_config["annualization_days"] == 252
    assert formal_config["min_observations"] == 40
    assert formal_config["min_pair_observations"] == 40
    assert formal_config["ramp_in_months"] == 6
    assert formal_config["regime_threshold_lookback_months"] == 36
    assert formal_config["regime_threshold_min_periods"] == 36
    removed_config_keys = {
        "data_source",
        "regime_threshold_method",
        "regime_threshold_include_current",
        "monthly_anchor",
        "daily_regime_effective",
        "cta_file",
        "cta_product_for_figure",
        "cta_annualization_months",
        "export_figures",
        "export_large_tables",
    }
    assert removed_config_keys.isdisjoint(formal_config)
    expected_figures = [
        "figure_1_volatility.png",
        "figure_2_correlation.png",
        "figure_3_nhci_regimes.png",
        "figure_4_regime_returns.png",
        "figure_5_fund_01.png",
        "figure_6_fund_02.png",
        "sector_volatility_contribution_river.png",
    ]
    assert all((project / "figures" / name).is_file() for name in expected_figures)
    dominant_after = pd.read_parquet(raw_dir / "dominant_prices.parquet")
    index_99_after = pd.read_parquet(raw_dir / "index_99_prices.parquet")
    assert len(dominant_after) == initial_rows + 2 * len(codes)
    assert len(index_99_after) == initial_rows + 2 * len(codes)
    assert dominant_after["date"].max() == pd.Timestamp("2019-01-02")
    calls = pd.read_json(project / "rqdatac_calls.json")
    price_calls = calls.loc[calls["kind"].isin(["dominant", "index_99"])]
    assert set(price_calls["start_date"]) == {"2019-01-01"}
