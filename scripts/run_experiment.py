"""Full pipeline: data -> features -> targets -> training/tuning -> evaluation -> backtests ->
walk-forward -> robustness -> figures -> Markdown report.

    python -m scripts.run_experiment                 # full experiment (config.yaml) -> results/
    python -m scripts.run_experiment --quick         # SPY smoke run -> results_quick/
    python -m scripts.run_experiment --tickers SPY   # single asset, full settings
    python -m scripts.run_experiment --skip-walk-forward
    python -m scripts.run_experiment --synthetic --quick   # offline dev run, labelled SYNTHETIC
"""

from __future__ import annotations

try:
    from scripts._cli import base_parser, config_from_args
except ImportError:
    from _cli import base_parser, config_from_args

from src.experiment import run_experiment
from src.utils import get_logger


def main() -> None:
    parser = base_parser(__doc__)
    parser.add_argument("--skip-walk-forward", action="store_true", help="skip the walk-forward stage")
    args = parser.parse_args()
    cfg = config_from_args(args)
    run_experiment(cfg, skip_walk_forward=args.skip_walk_forward, logger=get_logger())


if __name__ == "__main__":
    main()
