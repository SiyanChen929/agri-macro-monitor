from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from agri_macro_quant.pipeline import run_pipeline
from agri_macro_quant.config import load_config
from agri_macro_quant.pipeline import _prepare_output_dir
from agri_macro_quant.provenance import (
    build_manifest,
    dataframe_hash,
    file_hash,
    source_tree_hash,
)


def test_source_tree_hash_is_deterministic_and_ignores_generated_bytecode(
    tmp_path: Path,
) -> None:
    source = tmp_path / "src" / "example"
    source.mkdir(parents=True)
    module = source / "model.py"
    module.write_text("VALUE = 1\n", encoding="utf-8")

    first = source_tree_hash(tmp_path)
    (source / "__pycache__").mkdir()
    (source / "__pycache__" / "model.cpython-312.pyc").write_bytes(b"generated")
    assert source_tree_hash(tmp_path) == first

    module.write_text("VALUE = 2\n", encoding="utf-8")
    assert source_tree_hash(tmp_path) != first


def test_manifest_records_runtime_source_and_data_lineage(tmp_path: Path) -> None:
    source = tmp_path / "src" / "example"
    source.mkdir(parents=True)
    (source / "model.py").write_text("VALUE = 1\n", encoding="utf-8")
    (tmp_path / "pyproject.toml").write_text(
        """[project]\nname = "example"\nversion = "0.0.1"\ndependencies = ["pandas>=2"]\n""",
        encoding="utf-8",
    )
    raw = pd.DataFrame(
        {
            "series_id": ["wheat"],
            "observation_date": [pd.Timestamp("2024-01-01")],
            "publication_date": [pd.Timestamp("2024-01-02")],
            "value": [6.25],
        }
    )
    aligned = pd.DataFrame(
        {"wheat": [6.25]}, index=pd.DatetimeIndex(["2024-01-02"], name="date")
    )
    availability = pd.DataFrame(
        {"wheat": [pd.Timestamp("2024-01-02T22:00:00")]}, index=aligned.index
    )
    raw_manifest = {"sha256": "a" * 64, "rows": 1}

    manifest = build_manifest(
        project_root=tmp_path,
        configuration={"runtime": {"seed": 7}},
        data=aligned,
        raw_observations=raw,
        availability=availability,
        raw_snapshot_manifest=raw_manifest,
        raw_snapshot_manifest_sha256="b" * 64,
        raw_snapshot_manifest_path="data/raw/observations.csv.gz.manifest.json",
        upstream_snapshot_manifest={"sha256": "c" * 64},
        upstream_snapshot_manifest_sha256="d" * 64,
        as_of="2024-01-02",
        mode="demo",
        seed=7,
    )

    assert manifest["source_tree_sha256"] == source_tree_hash(tmp_path)
    assert manifest["raw_observation_vintage_sha256"] == dataframe_hash(raw)
    assert manifest["raw_snapshot_file_sha256"] == "a" * 64
    assert manifest["raw_snapshot_manifest_sha256"] == "b" * 64
    assert manifest["upstream_snapshot_manifest_sha256"] == "d" * 64
    assert len(manifest["upstream_snapshot_manifest_content_sha256"]) == 64
    assert manifest["aligned_panel_sha256"] == dataframe_hash(aligned)
    assert manifest["aligned_availability_sha256"] == dataframe_hash(availability)
    assert manifest["data_snapshot_sha256"] == manifest["aligned_panel_sha256"]
    assert manifest["python_runtime"]["version"]
    assert manifest["direct_dependencies"]["pandas"]["installed_version"]
    assert set(manifest["git"]) == {"available", "commit", "dirty"}


def test_overwrite_removes_stale_statistical_artifacts_and_records_snapshot_links(
    tmp_path: Path,
) -> None:
    output = tmp_path / "named_run"
    output.mkdir()
    (output / ".agri_macro_quant_run.json").write_text(
        json.dumps(
            {
                "format": "agri_macro_quant_run_v1",
                "project_root": str(Path(__file__).resolve().parents[1]),
            }
        ),
        encoding="utf-8",
    )
    stale_statistical = output / "diagnostics" / "statistical_regime_metadata.json"
    stale_statistical.parent.mkdir(parents=True)
    stale_statistical.write_text('{"stale": true}', encoding="utf-8")
    stale_chart = output / "charts" / "obsolete_capability.png"
    stale_chart.parent.mkdir(parents=True)
    stale_chart.write_bytes(b"obsolete")
    sibling = tmp_path / "unrelated-output.txt"
    sibling.write_text("preserve me", encoding="utf-8")

    result = run_pipeline(
        mode="demo",
        start="2024-01-02",
        as_of="2024-06-28",
        output_dir=output,
        overwrite=True,
        run_statistical=False,
        record_research_log=False,
    )

    assert result.statistical_regimes is None
    assert not stale_statistical.exists()
    assert not stale_chart.exists()
    assert not list((output / "diagnostics").glob("statistical_regime_*"))
    assert sibling.read_text(encoding="utf-8") == "preserve me"

    persisted = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    raw_manifest_path = output / persisted["raw_snapshot_manifest_path"]
    raw_manifest = json.loads(raw_manifest_path.read_text(encoding="utf-8"))
    assert persisted["raw_snapshot_manifest_sha256"] == file_hash(raw_manifest_path)
    assert persisted["raw_snapshot_file_sha256"] == raw_manifest["sha256"]
    assert persisted["raw_observation_vintage_sha256"] == dataframe_hash(
        result.observations
    )
    assert persisted["aligned_panel_sha256"] == dataframe_hash(result.market)
    assert len(persisted["aligned_availability_sha256"]) == 64
    project_root = Path(__file__).resolve().parents[1]
    assert persisted["source_tree_sha256"] == source_tree_hash(project_root)


def test_overwrite_refuses_an_unmarked_arbitrary_directory(tmp_path: Path) -> None:
    output = tmp_path / "not_a_managed_run"
    output.mkdir()
    sentinel = output / "keep.txt"
    sentinel.write_text("do not delete", encoding="utf-8")

    with pytest.raises(ValueError, match="unmarked directory"):
        _prepare_output_dir(load_config(), output, overwrite=True)

    assert sentinel.read_text(encoding="utf-8") == "do not delete"


def test_default_output_directories_are_atomically_unique(tmp_path: Path) -> None:
    config = load_config(
        overrides={"paths": {"reports_dir": str(tmp_path / "reports")}}
    )

    first = _prepare_output_dir(config, None, overwrite=False)
    second = _prepare_output_dir(config, None, overwrite=False)

    assert first != second
    assert (first / ".agri_macro_quant_run.json").is_file()
    assert (second / ".agri_macro_quant_run.json").is_file()
