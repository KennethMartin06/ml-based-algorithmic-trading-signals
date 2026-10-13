"""Stage 1 - train and tune all holdout models, write validation/test predictions.

    python -m scripts.train [--tickers SPY] [--quick] [--skip-lstm]

Outputs: results/predictions/<T>_holdout_predictions.csv, results/metrics/tuning/*,
results/metrics/feature_importance/*, results/metrics/best_params.json, LSTM history.
"""

from __future__ import annotations

try:
    from scripts._cli import base_parser, config_from_args
except ImportError:
    from _cli import base_parser, config_from_args

from src.experiment import OutputPaths, load_data, reset_outputs, train_stage
from src.utils import get_logger, save_json, set_global_seed


def main() -> None:
    args = base_parser(__doc__).parse_args()
    cfg = config_from_args(args)
    set_global_seed(cfg["experiment"]["seed"])
    frames, manifest = load_data(cfg)  # may redirect to a *_synthetic_fallback directory
    reset_outputs(cfg["experiment"]["results_dir"])  # stage 1 starts a new run: no stale artefacts
    paths = OutputPaths(cfg["experiment"]["results_dir"])
    save_json(manifest, paths.metrics / "data_manifest.json")
    train_stage(cfg, frames, paths, get_logger())


if __name__ == "__main__":
    main()
