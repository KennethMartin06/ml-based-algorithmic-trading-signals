"""Shared argument parsing for the command-line scripts."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:  # allow `python scripts/<name>.py` as well as `python -m scripts.<name>`
    sys.path.insert(0, str(ROOT))

from src.experiment import build_config  # noqa: E402
from src.utils import load_config  # noqa: E402

DEFAULT_CONFIG = "config.yaml"


def base_parser(description: str) -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=description, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", default=DEFAULT_CONFIG, help="YAML configuration file (relative to the project root)")
    p.add_argument("--tickers", nargs="+", help="override the ticker universe, e.g. --tickers SPY (-> results_<TICKERS>/)")
    p.add_argument("--quick", action="store_true", help="fast smoke run: SPY only, no LSTM tuning, short walk-forward -> results_quick/")
    p.add_argument("--synthetic", action="store_true",
                   help="use a deterministic SYNTHETIC random walk instead of market data (development only) -> results_synthetic/")
    p.add_argument("--skip-lstm", action="store_true", help="disable the LSTM, e.g. if PyTorch is not installed (-> results_nolstm/)")
    p.add_argument("--results-dir", help="write outputs to this directory instead of the automatically chosen one")
    return p


def config_from_args(args) -> dict:
    cfg = load_config(args.config)
    cfg = build_config(cfg, quick=args.quick, synthetic=args.synthetic, skip_lstm=args.skip_lstm, tickers=args.tickers,
                       skip_walk_forward=getattr(args, "skip_walk_forward", False), results_dir=args.results_dir)
    if Path(args.config).name != DEFAULT_CONFIG:
        cfg["_is_default_full_run"] = False
    return cfg
