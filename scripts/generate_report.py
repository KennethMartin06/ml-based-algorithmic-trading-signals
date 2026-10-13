"""Rebuild results/reports/experiment_report.md from the saved metric files (no retraining).

    python -m scripts.generate_report [--quick]
"""

from __future__ import annotations

try:
    from scripts._cli import base_parser, config_from_args
except ImportError:
    from _cli import base_parser, config_from_args

from src.reporting import generate_report
from src.utils import get_logger


def main() -> None:
    args = base_parser(__doc__).parse_args()
    cfg = config_from_args(args)
    path = generate_report(cfg["experiment"]["results_dir"], cfg)
    get_logger().info("Report written to %s", path)


if __name__ == "__main__":
    main()
