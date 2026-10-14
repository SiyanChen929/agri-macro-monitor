from __future__ import annotations

from dataclasses import replace
import json

import numpy as np
import pandas as pd

from agri_macro_quant.alignment import PITAligner, align_point_in_time
from agri_macro_quant.config import load_config
from agri_macro_quant.data_models import SeriesFrequency, SeriesSpec
from agri_macro_quant.features import point_in_time_asof
from agri_macro_quant.ingestion import ingest_all, load_snapshot, save_snapshot
from agri_macro_quant.pipeline import _source_label


def test_release_is_unavailable_before_intraday_publication(
    observation_frame,
) -> None:
    observations = observation_frame(
        {
            "observation_date": "2024-01-01",
            "publication_date": "2024-01-03 18:00:00",
            "value": 72.0,
        }
    )
    spec = SeriesSpec(
        "crop_condition",
        SeriesFrequency.WEEKLY,
        max_staleness_days=10,
    )
    calendar = pd.date_range("2024-01-02", "2024-01-05", freq="D")

    result = PITAligner(
        {spec.series_id: spec},
        calendar_frequency="D",
        decision_time_utc="17:00",
    ).align(observations, calendar)

    assert result.data.loc["2024-01-02":"2024-01-03", "crop_condition"].isna().all()
    assert result.data.loc[pd.Timestamp("2024-01-04"), "crop_condition"] == 72.0
    assert result.availability.loc[pd.Timestamp("2024-01-04"), "crop_condition"] == pd.Timestamp(
        "2024-01-03 18:00:00"
    )
    assert result.data.attrs["point_in_time_aligned"] is True


def test_revision_replaces_only_its_vintage_and_old_revision_cannot_displace_new_observation(
    observation_frame,
) -> None:
    observations = observation_frame(
        {
            "observation_date": "2024-01-01",
            "publication_date": "2024-01-02",
            "value": 10.0,
            "vintage_id": "jan-01-initial",
        },
        {
            "observation_date": "2024-01-03",
            "publication_date": "2024-01-04",
            "value": 20.0,
            "vintage_id": "jan-03-initial",
        },
        {
            "observation_date": "2024-01-01",
            "publication_date": "2024-01-02",
            "revision_date": "2024-01-05",
            "value": 99.0,
            "vintage_id": "jan-01-revised",
        },
        {
            "observation_date": "2024-01-03",
            "publication_date": "2024-01-04",
            "revision_date": "2024-01-06",
            "value": 21.0,
            "vintage_id": "jan-03-revised",
        },
    )
    spec = SeriesSpec(
        "crop_condition",
        SeriesFrequency.WEEKLY,
        max_staleness_days=30,
    )

    result = align_point_in_time(
        observations,
        {spec.series_id: spec},
        calendar=pd.date_range("2024-01-01", "2024-01-07", freq="D"),
        calendar_frequency="D",
    )

    expected = pd.Series(
        [np.nan, 10.0, 10.0, 20.0, 20.0, 21.0, 21.0],
        index=result.data.index,
        name="crop_condition",
    )
    pd.testing.assert_series_equal(result.data["crop_condition"], expected)
    assert result.observation_dates.loc[pd.Timestamp("2024-01-05"), "crop_condition"] == pd.Timestamp(
        "2024-01-03"
    )


def test_stale_release_is_masked_after_configured_age_limit(
    observation_frame,
) -> None:
    observations = observation_frame(
        {
            "observation_date": "2024-02-01",
            "publication_date": "2024-02-02",
            "value": 28.5,
        },
        series_id="exporter_stocks_to_use",
    )
    spec = SeriesSpec(
        "exporter_stocks_to_use",
        SeriesFrequency.MONTHLY,
        max_staleness_days=2,
    )
    result = align_point_in_time(
        observations,
        {spec.series_id: spec},
        calendar=pd.date_range("2024-02-01", "2024-02-06", freq="D"),
        calendar_frequency="D",
    )

    values = result.data[spec.series_id]
    assert values.loc["2024-02-02":"2024-02-04"].eq(28.5).all()
    assert values.loc["2024-02-05":].isna().all()
    assert result.age_days.loc[pd.Timestamp("2024-02-05"), spec.series_id] == 3.0
    assert result.metadata.loc[spec.series_id, "stale"]
    assert pd.isna(result.metadata.loc[spec.series_id, "latest_value"])


def test_point_in_time_asof_honours_revision_timestamp_and_staleness() -> None:
    records = pd.DataFrame(
        {
            "series": ["yield", "yield"],
            "observation_date": ["2024-03-01", "2024-03-01"],
            "publication_date": ["2024-03-02", "2024-03-02"],
            "revision_date": ["2024-03-02", "2024-03-04"],
            "value": [100.0, 96.0],
        }
    )
    calendar = pd.date_range("2024-03-01", "2024-03-06", freq="D")

    aligned = point_in_time_asof(records, calendar, max_age_days={"yield": 4})

    assert pd.isna(aligned.loc[pd.Timestamp("2024-03-01"), "yield"])
    assert aligned.loc[pd.Timestamp("2024-03-02"), "yield"] == 100.0
    assert aligned.loc[pd.Timestamp("2024-03-03"), "yield"] == 100.0
    assert aligned.loc[pd.Timestamp("2024-03-04"), "yield"] == 96.0
    assert aligned.loc[pd.Timestamp("2024-03-05"), "yield"] == 96.0
    assert pd.isna(aligned.loc[pd.Timestamp("2024-03-06"), "yield"])
    assert pd.isna(aligned.loc[pd.Timestamp("2024-03-06"), "yield__data_age_days"])


def test_live_dispatcher_reads_configured_local_csv(tmp_path) -> None:
    location = tmp_path / "licensed_weather.csv"
    pd.DataFrame(
        {
            "observation_date": ["2024-01-01", "2024-01-02"],
            "value": [40.0, 42.0],
        }
    ).to_csv(location, index=False)
    spec = SeriesSpec(
        "weather_stress",
        SeriesFrequency.DAILY,
        source="local_csv",
        source_id=str(location),
        publication_lag_days=1,
        max_staleness_days=4,
        critical=True,
    )
    base = load_config()
    config = replace(
        base,
        series={spec.series_id: spec},
        data_sources={
            "sources": {"local_csv": {"enabled": True, "live": True}},
            "series": {spec.series_id: {"source": "local_csv"}},
        },
    )

    observations = ingest_all(
        config,
        "live",
        start="2024-01-01",
        end="2024-01-05",
    )

    assert observations["value"].tolist() == [40.0, 42.0]
    assert observations["publication_date"].tolist() == [
        pd.Timestamp("2024-01-02 23:59:59"),
        pd.Timestamp("2024-01-03 23:59:59"),
    ]
    assert observations.attrs["mode"] == "live"
    assert observations.attrs["ingestion_issues"] == []


def test_aligner_honours_configured_quality_status_threshold(observation_frame) -> None:
    observations = observation_frame(
        {
            "observation_date": "2024-01-01",
            "publication_date": "2024-01-02",
            "frequency": "daily",
            "value": 50.0,
        }
    )
    spec = SeriesSpec("crop_condition", SeriesFrequency.WEEKLY, max_staleness_days=10)
    result = PITAligner(
        {spec.series_id: spec},
        calendar_frequency="D",
        quality_config={"warning_score": 100.0, "failure_score": 0.0},
    ).align(observations, pd.date_range("2024-01-01", "2024-01-03", freq="D"))

    assert result.quality.score == 99.0
    assert result.quality.status == "warning"


def test_shared_canonical_csv_is_filtered_per_configured_series(tmp_path) -> None:
    location = tmp_path / "shared_canonical.csv"
    pd.DataFrame(
        {
            "series_id": ["weather_stress", "crop_condition"],
            "observation_date": ["2024-01-01", "2024-01-01"],
            "publication_date": ["2024-01-02", "2024-01-02"],
            "effective_date": ["2024-01-01", "2024-01-01"],
            "revision_date": ["2024-01-02", "2024-01-02"],
            "value": [40.0, 70.0],
            "source": ["licensed", "licensed"],
            "frequency": ["weekly", "weekly"],
            "unit": ["score_0_100", "percent_good_excellent"],
            "vintage_id": ["initial", "initial"],
        }
    ).to_csv(location, index=False)
    specs = {
        name: SeriesSpec(
            name,
            SeriesFrequency.WEEKLY,
            source="local_csv",
            source_id=str(location),
            max_staleness_days=10,
        )
        for name in ("weather_stress", "crop_condition")
    }
    base = load_config()
    config = replace(
        base,
        series=specs,
        data_sources={
            "sources": {"local_csv": {"enabled": True, "live": True}},
            "series": {
                name: {"source": "local_csv", "source_id": str(location)}
                for name in specs
            },
        },
    )

    observations = ingest_all(
        config,
        "live",
        start="2024-01-01",
        end="2024-01-05",
    )

    assert len(observations) == 2
    assert observations.groupby("series_id").size().to_dict() == {
        "crop_condition": 1,
        "weather_stress": 1,
    }


def test_cache_future_vintages_cannot_change_asof_quality_or_metadata(tmp_path) -> None:
    base = load_config()
    wheat_spec = base.series["wheat"]
    config = replace(base, series={"wheat": wheat_spec})
    rows = pd.DataFrame(
        {
            "series_id": ["wheat", "wheat"],
            "observation_date": ["2024-01-01", "2025-01-01"],
            "publication_date": ["2024-01-01 20:00:00", "2025-01-01 20:00:00"],
            "effective_date": ["2024-01-01", "2025-01-01"],
            "revision_date": ["2024-01-01 20:00:00", "2025-01-01 20:00:00"],
            "value": [5.0, 999.0],
            "source": ["governed_cache", "governed_cache"],
            "frequency": ["daily", "daily"],
            "unit": ["usd_per_bushel", "usd_per_bushel"],
            "vintage_id": ["known", "future-impossible"],
        }
    )
    prefix_path = tmp_path / "prefix.csv.gz"
    extended_path = tmp_path / "extended.csv.gz"
    save_snapshot(rows.iloc[:1], prefix_path)
    save_snapshot(rows, extended_path)

    results = []
    ingested = []
    for path in (prefix_path, extended_path):
        observations = ingest_all(
            config,
            "cache",
            start="2024-01-01",
            end="2024-01-02",
            cache_path=path,
        )
        ingested.append(observations)
        results.append(
            align_point_in_time(
                observations,
                config.series,
                start="2024-01-01",
                end="2024-01-02",
                as_of="2024-01-02",
                decision_time_utc=config.runtime.decision_time_utc,
                calendar_frequency="D",
            )
        )

    assert len(ingested[0]) == len(ingested[1]) == 1
    assert ingested[1].attrs["source_snapshot_rows"] == 2
    assert ingested[1].attrs["excluded_post_cutoff_rows"] == 1
    pd.testing.assert_frame_equal(results[0].data, results[1].data)
    pd.testing.assert_frame_equal(results[0].metadata, results[1].metadata)
    assert results[0].quality.score == results[1].quality.score
    assert results[0].quality.status == results[1].quality.status
    assert [issue.code for issue in results[0].quality.issues] == [
        issue.code for issue in results[1].quality.issues
    ]
    assert "impossible_value" not in [issue.code for issue in results[1].quality.issues]


def test_cache_replay_preserves_synthetic_demo_origin(tmp_path) -> None:
    base = load_config()
    config = replace(base, series={"wheat": base.series["wheat"]})
    path = tmp_path / "demo-origin.csv.gz"
    row = pd.DataFrame(
        {
            "series_id": ["wheat"],
            "observation_date": ["2024-01-01"],
            "publication_date": ["2024-01-01 20:00:00"],
            "effective_date": ["2024-01-01"],
            "revision_date": ["2024-01-01 20:00:00"],
            "value": [5.0],
            "source": ["demo"],
            "frequency": ["daily"],
            "unit": ["USD/bushel"],
            "vintage_id": ["fixture"],
        }
    )
    row.attrs.update({"mode": "demo", "synthetic": True})
    save_snapshot(row, path)

    replay = ingest_all(
        config,
        "cache",
        start="2024-01-01",
        end="2024-01-02",
        cache_path=path,
    )

    assert replay.attrs["mode"] == "cache"
    assert replay.attrs["origin_mode"] == "demo"
    assert replay.attrs["synthetic"] is True
    assert "DEMO-ORIGIN" in _source_label("cache", replay.attrs)


def test_legacy_snapshot_infers_synthetic_origin_from_rows(tmp_path) -> None:
    path = tmp_path / "legacy-demo.csv.gz"
    row = pd.DataFrame(
        {
            "series_id": ["wheat"],
            "observation_date": ["2024-01-01"],
            "publication_date": ["2024-01-01 20:00:00"],
            "effective_date": ["2024-01-01"],
            "revision_date": ["2024-01-01 20:00:00"],
            "value": [5.0],
            "source": ["demo"],
            "frequency": ["daily"],
            "unit": ["USD/bushel"],
            "vintage_id": ["fixture"],
        }
    )
    save_snapshot(row, path)
    manifest_path = path.with_name(f"{path.name}.manifest.json")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["metadata"] = {}  # simulate a checksum-valid legacy manifest
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    replay = load_snapshot(path)

    assert replay.attrs["synthetic"] is True
    assert replay.attrs["origin_mode"] == "demo"
    assert "DEMO-ORIGIN" in _source_label("cache", replay.attrs)


def test_live_canonical_csv_cannot_launder_demo_rows(tmp_path) -> None:
    path = tmp_path / "claimed-live.csv"
    row = pd.DataFrame(
        {
            "series_id": ["wheat"],
            "observation_date": ["2024-01-02"],
            "publication_date": ["2024-01-02 20:00:00"],
            "effective_date": ["2024-01-02"],
            "revision_date": ["2024-01-02 20:00:00"],
            "value": [5.0],
            "source": ["demo"],
            "frequency": ["daily"],
            "unit": ["USD/bushel"],
            "vintage_id": ["fixture"],
        }
    )
    row.to_csv(path, index=False)
    base = load_config()
    spec = SeriesSpec(
        "wheat",
        SeriesFrequency.DAILY,
        source="local_csv",
        source_id=str(path),
        critical=True,
    )
    config = replace(base, series={"wheat": spec})

    live = ingest_all(
        config,
        "live",
        start="2024-01-01",
        end="2024-01-03",
    )

    assert live.attrs["synthetic"] is True
    assert live.attrs["origin_mode"] == "mixed_with_synthetic"
    assert any(
        issue["code"] == "synthetic_rows_in_live_mode"
        for issue in live.attrs["ingestion_issues"]
    )
