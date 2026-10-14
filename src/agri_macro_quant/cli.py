"""Command-line interface for scheduled and interactive research runs."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys

import pandas as pd

from .alignment import align_point_in_time
from .config import load_config
from .ingestion import ingest_all
from .pipeline import run_pipeline


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="agri-macro",
        description="Point-in-time agricultural supply-shock regime research platform",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    run = subparsers.add_parser("run", help="execute the full research workflow")
    run.add_argument("--config", type=Path, default=None)
    run.add_argument("--mode", choices=["demo", "live", "cache"], default=None)
    run.add_argument("--start", default=None, help="first observation date (YYYY-MM-DD)")
    run.add_argument("--as-of", default=None, help="information cutoff (YYYY-MM-DD)")
    run.add_argument("--cache-path", type=Path, default=None)
    run.add_argument("--output-dir", type=Path, default=None)
    run.add_argument("--overwrite", action="store_true")
    run.add_argument("--fail-on-quality", action="store_true")
    run.add_argument("--skip-statistical", action="store_true")
    run.add_argument("--no-research-log", action="store_true")

    validate = subparsers.add_parser("validate-data", help="ingest, PIT-align, and validate only")
    validate.add_argument("--config", type=Path, default=None)
    validate.add_argument("--mode", choices=["demo", "live", "cache"], default=None)
    validate.add_argument("--start", default=None)
    validate.add_argument("--as-of", default=None)
    validate.add_argument("--cache-path", type=Path, default=None)

    show = subparsers.add_parser("show-config", help="print resolved configuration summary")
    show.add_argument("--config", type=Path, default=None)

    predictive = subparsers.add_parser(
        "predictive-experiment",
        help="execute the frozen real-data ZW/ZC/ZS prediction experiment",
    )
    predictive.add_argument(
        "--experiment-config",
        type=Path,
        default=Path("config/predictive_experiment.yaml"),
    )
    predictive.add_argument(
        "--data-dir", type=Path, default=Path("data/predictive_experiment_v1")
    )
    predictive.add_argument("--output-dir", type=Path, default=None)
    predictive.add_argument("--refresh-data", action="store_true")
    return parser


def _validate(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    if not config.runtime.strict_point_in_time:
        raise ValueError(
            "runtime.strict_point_in_time=false is unsupported: validation is always point-in-time strict"
        )
    start = pd.Timestamp(args.start or config.runtime.start_date).normalize()
    as_of = pd.Timestamp(args.as_of or config.runtime.end_date or pd.Timestamp.now()).normalize()
    observations = ingest_all(
        config,
        args.mode or config.runtime.mode,
        start=start,
        end=as_of,
        cache_path=args.cache_path,
    )
    result = align_point_in_time(
        observations,
        config.series,
        start=start,
        end=as_of,
        as_of=as_of,
        decision_time_utc=config.runtime.decision_time_utc,
        calendar_frequency=config.runtime.calendar,
        quality_config=asdict(config.quality),
    )
    print(json.dumps(result.quality.as_dict(), indent=2, sort_keys=True))
    return 0 if result.quality.passed else 2


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.command == "predictive-experiment":
            from .research.predictive_runner import (
                execute_predictive_experiment,
                load_predictive_config,
            )

            loaded = load_predictive_config(args.experiment_config)
            output = args.output_dir or Path("reports") / (
                f"{loaded.experiment_id}_{loaded.as_of_date.strftime('%Y%m%d')}"
            )
            experiment = execute_predictive_experiment(
                config_path=args.experiment_config,
                data_dir=args.data_dir,
                output_dir=output,
                refresh_data=bool(args.refresh_data),
            )
            print(
                json.dumps(
                    {
                        "status": "completed",
                        "conclusion": experiment.conclusion,
                        "output_dir": str(experiment.output_dir),
                        "report": str(experiment.report_html),
                        "primary_selected_model": experiment.primary.selected_model,
                        "raw_sensitivity_selected_model": (
                            experiment.raw_sensitivity.selected_model
                        ),
                    },
                    indent=2,
                    sort_keys=True,
                )
            )
            return 0
        if args.command == "show-config":
            config = load_config(args.config)
            summary = {
                "project_root": str(config.project_root),
                "mode": config.runtime.mode,
                "start_date": config.runtime.start_date,
                "calendar": config.runtime.calendar,
                "series": sorted(config.series),
                "reports_dir": str(config.paths.reports_dir),
            }
            print(json.dumps(summary, indent=2, sort_keys=True))
            return 0
        if args.command == "validate-data":
            return _validate(args)
        result = run_pipeline(
            config_path=args.config,
            mode=args.mode,
            start=args.start,
            as_of=args.as_of,
            cache_path=args.cache_path,
            output_dir=args.output_dir,
            overwrite=args.overwrite,
            fail_on_quality=args.fail_on_quality,
            run_statistical=not args.skip_statistical,
            record_research_log=not args.no_research_log,
        )
        latest = result.regimes.scores.iloc[-1]
        payload = {
            "status": "completed",
            "output_dir": str(result.output_dir),
            "report": str(result.reports["html"]),
            "quality": {"status": result.quality.status, "score": result.quality.score},
            "dominant_regime": str(latest.idxmax()),
            "dominant_score": float(latest.max()),
            "signals": int(len(result.signals)),
            "charts": len(result.reports["charts"]),
        }
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 0
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        print(f"agri-macro: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
