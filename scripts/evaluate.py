"""Stage 2 - evaluate saved holdout predictions: metrics, thresholds (validation), backtests,
robustness analyses and figures. Requires `python -m scripts.train` to have run first.

    python -m scripts.evaluate [--tickers SPY] [--quick]
"""

from __future__ import annotations

try:
    from scripts._cli import base_parser, config_from_args
except ImportError:
    from _cli import base_parser, config_from_args

from src.experiment import OutputPaths, evaluate_stage, load_data
from src.utils import get_logger


def main() -> None:
    args = base_parser(__doc__).parse_args()
    cfg = config_from_args(args)
    frames, _ = load_data(cfg)  # may redirect to a *_synthetic_fallback directory
    paths = OutputPaths(cfg["experiment"]["results_dir"])
    missing = [t for t in cfg["data"]["tickers"] if not (paths.predictions / f"{t}_holdout_predictions.csv").exists()]
    if missing:
        raise SystemExit(f"No saved predictions for {missing}; run `python -m scripts.train` first.")
    evaluate_stage(cfg, frames, paths, get_logger())


if __name__ == "__main__":
    main()
