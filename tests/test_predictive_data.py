from __future__ import annotations

from hashlib import sha256
from io import BytesIO
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

import pandas as pd

import agri_macro_quant.research.predictive_data as predictive_data
from agri_macro_quant.research.predictive_data import (
    ASSET_SPECS,
    build_usda_features,
    download_usda_wasde,
)


WASDE_COLUMNS = [
    "WasdeNumber",
    "ReportDate",
    "ReportTitle",
    "Attribute",
    "ReliabilityProjection",
    "Commodity",
    "Region",
    "MarketYear",
    "ProjEstFlag",
    "AnnualQuarterFlag",
    "Value",
    "Unit",
    "ReleaseDate",
    "ReleaseTime",
    "ForecastYear",
    "ForecastMonth",
]


def _one_row_csv(release_date: str) -> bytes:
    row = {
        "WasdeNumber": "481",
        "ReportDate": "April 2010",
        "ReportTitle": "U.S. Feed Grain and Corn Supply and Use",
        "Attribute": "Production",
        "ReliabilityProjection": "",
        "Commodity": "Corn",
        "Region": "United States",
        "MarketYear": "2009/10",
        "ProjEstFlag": "Proj.",
        "AnnualQuarterFlag": "Annual",
        "Value": "13000",
        "Unit": "Million Bushels",
        "ReleaseDate": release_date,
        "ReleaseTime": "08:30:00",
        "ForecastYear": "2010",
        "ForecastMonth": "4",
    }
    return pd.DataFrame([row], columns=WASDE_COLUMNS).to_csv(index=False).encode()


def _archive(member: str, csv_payload: bytes) -> bytes:
    output = BytesIO()
    with ZipFile(output, "w", compression=ZIP_DEFLATED) as archive:
        archive.writestr(member, csv_payload)
    return output.getvalue()


def test_archive_falls_back_to_pinned_replay_and_validates_coverage(
    tmp_path: Path, monkeypatch
) -> None:
    member = "official.csv"
    payload = _archive(member, _one_row_csv("2010-04-09"))
    spec = {
        "start": "2010-04-01",
        "end": "2010-04-30",
        "url": "https://www.usda.gov/official.zip",
        "fallback_url": "https://web.archive.org/official.zip",
        "sha256": sha256(payload).hexdigest(),
        "member": member,
        "output": "wasde_2010_04_to_2010_04.csv",
    }
    monkeypatch.setattr(predictive_data, "USDA_WASDE_ARCHIVES", (spec,))

    def fake_download(url: str, **_: object) -> bytes:
        if "usda.gov" in url:
            raise RuntimeError("blocked edge")
        return payload

    monkeypatch.setattr(predictive_data, "_download", fake_download)
    files, failures = download_usda_wasde(
        tmp_path, start="2010-04-01", end="2010-04-30", workers=1
    )

    assert failures == []
    assert [path.name for path in files] == ["wasde_2010_04_to_2010_04.csv"]
    assert files[0].read_bytes() == _one_row_csv("2010-04-09")


def test_monthly_download_probes_official_v2_name(tmp_path: Path, monkeypatch) -> None:
    payload = _one_row_csv("2026-05-13")
    calls: list[str] = []
    monkeypatch.setattr(predictive_data, "USDA_WASDE_ARCHIVES", ())

    def fake_download(url: str, **_: object) -> bytes:
        calls.append(url)
        if url.endswith("2026-05-V2.csv"):
            return payload
        raise RuntimeError("not found")

    monkeypatch.setattr(predictive_data, "_download", fake_download)
    files, failures = download_usda_wasde(
        tmp_path, start="2026-05-01", end="2026-05-31", workers=1
    )

    assert failures == []
    assert len(files) == 1
    assert calls[0].endswith("2026-05.csv")
    assert calls[1].endswith("2026-05-V2.csv")


def test_consolidated_csv_keeps_each_release_and_parses_mixed_dates(
    tmp_path: Path,
) -> None:
    spec = ASSET_SPECS["ZC"]
    rows: list[dict[str, object]] = []
    for number, release_date in enumerate(("01/12/2021", "2021-02-09"), start=1):
        for market_year, offset in (("2019/20", 0.0), ("2020/21", 100.0)):
            for region, title, values in (
                (
                    "United States",
                    spec["us_title"],
                    {
                        "Production": 14_000.0 + offset + number,
                        "Yield per Harvested Acre": 170.0 + number,
                        "Ending Stocks": 1_500.0 + number,
                        "Use, Total": 14_500.0 + number,
                    },
                ),
                (
                    "World",
                    spec["world_title"],
                    {
                        "Production": 1_100.0 + offset + number,
                        "Ending Stocks": 280.0 + number,
                        "Domestic Total": 1_150.0 + number,
                    },
                ),
            ):
                for attribute, value in values.items():
                    rows.append(
                        {
                            "WasdeNumber": 600 + number,
                            "ReportDate": release_date,
                            "ReportTitle": title,
                            "Attribute": attribute,
                            "ReliabilityProjection": "",
                            "Commodity": spec["commodity"],
                            "Region": region,
                            "MarketYear": market_year,
                            "ProjEstFlag": "Proj.",
                            "AnnualQuarterFlag": "Annual",
                            "Value": value,
                            "Unit": "test",
                            "ReleaseDate": release_date,
                            "ReleaseTime": "12:00:00",
                            "ForecastYear": 2021,
                            "ForecastMonth": number,
                        }
                    )
    path = tmp_path / "wasde_consolidated.csv"
    pd.DataFrame(rows, columns=WASDE_COLUMNS).to_csv(path, index=False)

    features = build_usda_features([path])

    assert features["release_date"].tolist() == [
        pd.Timestamp("2021-01-12"),
        pd.Timestamp("2021-02-09"),
    ]
    assert features["market_year_start"].tolist() == [2020.0, 2020.0]
    assert features["us_production_revision"].iloc[1] > 0.0
