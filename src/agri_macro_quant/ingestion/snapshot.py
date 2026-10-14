"""Portable canonical CSV snapshots with a reproducibility manifest."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import tempfile
from typing import Any

import pandas as pd

from ..data_models import ObservationFrame, normalize_observation_frame


def _manifest_path(path: Path) -> Path:
    return path.with_name(f"{path.name}.manifest.json")


def infer_synthetic_provenance(frame: pd.DataFrame) -> bool:
    """Conservatively detect synthetic/demo rows from canonical provenance."""

    if "source" not in frame.columns:
        return False
    sources = frame["source"].fillna("").astype(str).str.lower()
    return bool(sources.str.contains(r"demo|synthetic", regex=True).any())


def save_snapshot(
    frame: pd.DataFrame,
    path: str | Path,
    *,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Atomically write a canonical snapshot and return its audit manifest."""

    destination = Path(path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    source_attrs = dict(getattr(frame, "attrs", {}))
    canonical = normalize_observation_frame(frame)
    timestamp_columns = ["observation_date", "publication_date", "effective_date", "revision_date"]
    serializable = canonical.copy()
    for column in timestamp_columns:
        serializable[column] = serializable[column].dt.strftime("%Y-%m-%dT%H:%M:%S")

    suffix = ".csv.gz" if destination.name.endswith(".gz") else ".csv"
    with tempfile.NamedTemporaryFile(
        mode="wb", suffix=suffix, dir=destination.parent, delete=False
    ) as handle:
        temporary = Path(handle.name)
    try:
        serializable.to_csv(temporary, index=False, compression="gzip" if suffix.endswith(".gz") else None)
        payload = temporary.read_bytes()
        digest = hashlib.sha256(payload).hexdigest()
        temporary.replace(destination)
    finally:
        if temporary.exists():
            temporary.unlink()

    available = canonical[["publication_date", "revision_date", "effective_date"]].max(axis=1)
    provenance_metadata = {
        key: source_attrs[key]
        for key in (
            "mode",
            "origin_mode",
            "synthetic",
            "ingestion_issues",
            "timestamp_quality",
            "warning",
        )
        if key in source_attrs
    }
    if "origin_mode" not in provenance_metadata and "mode" in provenance_metadata:
        provenance_metadata["origin_mode"] = provenance_metadata["mode"]
    row_synthetic = infer_synthetic_provenance(canonical)
    if row_synthetic:
        if provenance_metadata.get("synthetic") is False:
            provenance_metadata["warning"] = (
                "row-level source provenance indicates demo/synthetic data; "
                "a contradictory synthetic=false declaration was overridden"
            )
        provenance_metadata["synthetic"] = True
        provenance_metadata.setdefault("origin_mode", "demo")
    provenance_metadata.update(metadata or {})
    # Caller metadata cannot downgrade row-level synthetic provenance.
    if row_synthetic:
        provenance_metadata["synthetic"] = True
        if provenance_metadata.get("origin_mode") not in {"demo", "mixed_with_synthetic"}:
            provenance_metadata["origin_mode"] = "mixed_with_synthetic"
    manifest = {
        "format": "canonical_observation_csv_v1",
        "path": str(destination),
        "sha256": digest,
        "rows": int(len(canonical)),
        "series": sorted(canonical["series_id"].dropna().astype(str).unique().tolist()),
        "min_observation_date": canonical["observation_date"].min().isoformat() if len(canonical) else None,
        "max_observation_date": canonical["observation_date"].max().isoformat() if len(canonical) else None,
        "max_available_at": available.max().isoformat() if len(canonical) else None,
        "metadata": provenance_metadata,
    }
    manifest_destination = _manifest_path(destination)
    manifest_destination.write_text(
        json.dumps(
            manifest,
            indent=2,
            sort_keys=True,
            default=str,
            allow_nan=False,
        ),
        encoding="utf-8",
    )
    return manifest


def load_snapshot(path: str | Path, *, verify: bool = True) -> ObservationFrame:
    """Load a prior snapshot and optionally verify its SHA-256 manifest."""

    source = Path(path).expanduser().resolve()
    if not source.exists():
        raise FileNotFoundError(f"Snapshot does not exist: {source}")
    manifest_path = _manifest_path(source)
    manifest: dict[str, Any] = {}
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if verify:
        if not manifest:
            raise ValueError(f"Snapshot manifest is missing: {manifest_path}")
        actual = hashlib.sha256(source.read_bytes()).hexdigest()
        if actual != manifest.get("sha256"):
            raise ValueError("Snapshot checksum does not match its manifest")
    frame = pd.read_csv(source)
    result = normalize_observation_frame(frame)
    metadata = manifest.get("metadata", {}) if isinstance(manifest, dict) else {}
    if not isinstance(metadata, dict):
        metadata = {}
    origin_mode = metadata.get("origin_mode", metadata.get("mode"))
    row_synthetic = infer_synthetic_provenance(result)
    synthetic = bool(metadata.get("synthetic", origin_mode == "demo")) or row_synthetic
    if row_synthetic and origin_mode not in {"demo", "mixed_with_synthetic"}:
        origin_mode = "mixed_with_synthetic" if origin_mode else "demo"
    result.attrs.update(
        {
            "snapshot_manifest": manifest,
            "snapshot_path": str(source),
            "origin_mode": origin_mode,
            "synthetic": synthetic,
            "ingestion_issues": metadata.get("ingestion_issues", []),
            "timestamp_quality": metadata.get("timestamp_quality"),
            "provenance_warning": (
                "row-level source provenance indicates demo/synthetic data"
                if row_synthetic and not bool(metadata.get("synthetic", False))
                else metadata.get("warning")
            ),
        }
    )
    return result
