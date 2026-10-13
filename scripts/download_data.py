"""Download (or load from cache) the configured universe and write a data manifest.

    python -m scripts.download_data                 # uses config.yaml
    python -m scripts.download_data --tickers SPY   # subset
    python -m scripts.download_data --refresh       # ignore the cache and download again
"""

from __future__ import annotations

try:
    from scripts._cli import base_parser, config_from_args
except ImportError:  # executed as `python scripts/download_data.py`
    from _cli import base_parser, config_from_args

from src.data_loader import load_universe
from src.utils import get_logger, resolve_path, save_json


def main() -> None:
    parser = base_parser(__doc__)
    parser.add_argument("--refresh", action="store_true", help="re-download even if a cached CSV exists")
    args = parser.parse_args()
    cfg = config_from_args(args)
    if args.refresh:
        cfg["data"]["use_cache"] = False
    log = get_logger()
    frames, manifest = load_universe(cfg, force_synthetic=cfg.get("_synthetic", False))
    # written next to the raw data, never into results/ (the experiment writes its own manifest)
    save_json(manifest, resolve_path(cfg["data"]["raw_dir"]) / "download_manifest.json")
    for m in manifest:
        log.info("%-5s %-9s rows=%d %s..%s sha256=%s", m["ticker"], m["source"], m["rows"], m["first_date"], m["last_date"], m["sha256"][:12])


if __name__ == "__main__":
    main()
