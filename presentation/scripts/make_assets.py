"""Generate the deck's background images and the price-trace motif data.

Outputs (all under presentation/assets/):
  backgrounds/bg_content.jpg  dark ink background with a faint cool glow (content slides)
  backgrounds/bg_hero.jpg     the same with a dot grid and a stronger glow (title / closing slides)
  data/spy_trace.csv          SPY adjusted close 2010-2025, every 5th session, normalised to 1.0
                              (geometry for the price-line motif; derived from data/raw, which is
                              gitignored, so the small derived file is kept with the deck)

Run from the project root:  python presentation/scripts/make_assets.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
ASSETS = ROOT / "presentation" / "assets"
W, H = 2560, 1440
BASE = np.array([10, 14, 20], dtype=np.float64)  # #0A0E14


def _glow(cx: float, cy: float, radius: float, color, strength: float) -> np.ndarray:
    """Soft radial glow (Gaussian falloff) as an additive RGB layer."""
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float64)
    d2 = ((xx - cx * W) ** 2 + (yy - cy * H) ** 2) / (radius * W) ** 2
    falloff = np.exp(-d2 * 2.2) * strength
    return falloff[..., None] * (np.array(color, dtype=np.float64) - BASE)


def _finish(img: np.ndarray, path: Path) -> None:
    rng = np.random.default_rng(7)
    img = img + rng.normal(0.0, 0.9, img.shape)  # dither so the glow does not band
    Image.fromarray(np.clip(img, 0, 255).astype(np.uint8)).save(path, quality=93, subsampling=0)


def backgrounds() -> None:
    out = ASSETS / "backgrounds"
    out.mkdir(parents=True, exist_ok=True)
    base = np.broadcast_to(BASE, (H, W, 3)).copy()

    content = base + _glow(0.92, -0.08, 0.55, (24, 52, 84), 0.55) + _glow(0.02, 1.08, 0.45, (48, 36, 16), 0.35)
    _finish(content, out / "bg_content.jpg")

    hero = base + _glow(0.88, 0.05, 0.65, (26, 60, 98), 0.75) + _glow(0.10, 1.05, 0.55, (70, 50, 18), 0.45)
    yy, xx = np.mgrid[0:H, 0:W]
    step = 48
    dots = (((xx % step) - step / 2) ** 2 + ((yy % step) - step / 2) ** 2) <= 2.2**2
    fade = np.clip((xx / W - 0.25) / 0.6, 0, 1) * np.clip(1.2 - yy / H, 0, 1)  # dots fade towards the text side
    hero += (dots * fade * 0.07)[..., None] * (255 - hero)
    _finish(hero, out / "bg_hero.jpg")


def trace() -> None:
    raw = ROOT / "data" / "raw" / "SPY_2010-01-01_2025-12-31.csv"
    out = ASSETS / "data"
    out.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(raw, parse_dates=["Date"])
    px = df["Adj Close"].astype(float)
    sub = df.iloc[::5].copy()
    if sub.index[-1] != df.index[-1]:
        sub = pd.concat([sub, df.iloc[[-1]]])
    sub = pd.DataFrame({"date": sub["Date"].dt.strftime("%Y-%m-%d"), "value": (px.loc[sub.index] / px.iloc[0]).round(5)})
    sub.to_csv(out / "spy_trace.csv", index=False)


if __name__ == "__main__":
    backgrounds()
    trace()
    print("assets written to", ASSETS)
