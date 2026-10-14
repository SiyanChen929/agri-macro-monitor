#!/usr/bin/env python3
"""Download the public inputs for the predictive experiment.

Fetches (1) USDA WAOB historical WASDE CSVs, (2) NOAA NCEI daily summaries for
the preregistered crop-belt station baskets, and (3) Yahoo continuous grain
futures proxies, then writes parsed point-in-time feature files and a hashed
data-lineage table. Nothing is redistributed with this repository; run this
script once before ``agri-macro predictive-experiment``.

Usage (from the repository root, after ``pip install -e .``)::

    python scripts/download_data.py
    python scripts/download_data.py --data-dir data/predictive_experiment_v1 --refresh
"""

from __future__ import annotations

import argparse
from pathlib import Path

from agri_macro_quant.research.predictive_data import acquire_predictive_data
from agri_macro_quant.research.predictive_runner import load_predictive_config


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--experiment-config",
        type=Path,
        default=Path("config/predictive_experiment.yaml"),
    )
    parser.add_argument(
        "--data-dir", type=Path, default=Path("data/predictive_experiment_v1")
    )
    parser.add_argument(
        "--refresh", action="store_true", help="re-download files already cached"
    )
    args = parser.parse_args(argv)

    loaded = load_predictive_config(args.experiment_config)
    bundle = acquire_predictive_data(
        args.data_dir,
        start=loaded.start_date,
        end=loaded.as_of_date,
        refresh=args.refresh,
    )
    print(bundle.lineage.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
