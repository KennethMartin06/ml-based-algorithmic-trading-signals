"""Render a deck through PowerPoint itself (COM) and run layout checks.

Outputs (into presentation/preview/ unless --out is given):
  slide-NN.png       every slide at 1920 x 1080, final (fully built) state
  contact_sheet.jpg  all slides on one page
  <deck>.pdf         static PDF export (optional, --pdf)
  layout_report.json text-overflow and off-slide checks, measured by PowerPoint's own text layout

PowerPoint renders with the real fonts, so a text box flagged here genuinely overflows in PowerPoint. A static
image cannot show motion; see verify_motion.py for that.

Usage:  python presentation/scripts/render_preview.py presentation/final_presentation.pptx [--pdf] [--out DIR]
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import tempfile
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[2]
PT_PER_IN = 72.0


def _iter_shapes(shapes):
    for shp in shapes:
        yield shp
        if shp.Type == 6:  # msoGroup
            for i in range(1, shp.GroupItems.Count + 1):
                yield shp.GroupItems.Item(i)


def layout_checks(pres) -> list[dict]:
    issues = []
    sw, sh = pres.PageSetup.SlideWidth, pres.PageSetup.SlideHeight
    for s in pres.Slides:
        for shp in _iter_shapes(s.Shapes):
            if shp.Type == 6:
                continue
            if shp.Left < -1 or shp.Top < -1 or shp.Left + shp.Width > sw + 1 or shp.Top + shp.Height > sh + 1:
                issues.append({"slide": s.SlideIndex, "shape": shp.Name, "issue": "outside slide"})
            if not shp.HasTextFrame or not shp.TextFrame2.HasText:
                continue
            tf = shp.TextFrame2
            tr = tf.TextRange
            avail_h = shp.Height - tf.MarginTop - tf.MarginBottom
            avail_w = shp.Width - tf.MarginLeft - tf.MarginRight
            over_h = tr.BoundHeight - avail_h
            over_w = tr.BoundWidth - avail_w if not tf.WordWrap else 0
            if over_h > 1.0 or over_w > 1.0:
                issues.append({"slide": s.SlideIndex, "shape": shp.Name, "issue": "text overflow",
                               "text": tr.Text[:70], "over_height_pt": round(over_h, 1),
                               "over_width_pt": round(over_w, 1)})
    return issues


def contact_sheet(pngs: list[Path], out: Path, cols: int = 4) -> None:
    thumbs = [Image.open(p).convert("RGB").resize((480, 270), Image.LANCZOS) for p in pngs]
    rows = (len(thumbs) + cols - 1) // cols
    pad, lab = 16, 26
    sheet = Image.new("RGB", (cols * (480 + pad) + pad, rows * (270 + pad + lab) + pad), (20, 24, 30))
    d = ImageDraw.Draw(sheet)
    try:
        font = ImageFont.truetype("consola.ttf", 16)
    except OSError:
        font = ImageFont.load_default()
    for i, im in enumerate(thumbs):
        r, c = divmod(i, cols)
        x, y = pad + c * (480 + pad), pad + r * (270 + pad + lab)
        d.text((x, y), f"{i + 1:02d}", fill=(255, 181, 71), font=font)
        sheet.paste(im, (x, y + lab))
    sheet.save(out, quality=88)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("deck")
    ap.add_argument("--out", default=str(ROOT / "presentation" / "preview"))
    ap.add_argument("--pdf", action="store_true")
    args = ap.parse_args()
    deck = Path(args.deck).resolve()
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)

    import win32com.client as w

    app = w.gencache.EnsureDispatch("PowerPoint.Application")
    pres = app.Presentations.Open(str(deck), True, False, False)  # read-only, no window
    try:
        tmp = Path(tempfile.mkdtemp())
        pres.Export(str(tmp), "PNG", 1920, 1080)
        for old in out.glob("slide-*.png"):
            old.unlink()
        exported = {int(re.search(r"(\d+)\.png$", p.name, re.I).group(1)): p for p in tmp.iterdir()}
        pngs = []
        for i in range(1, pres.Slides.Count + 1):
            dst = out / f"slide-{i:02d}.png"
            shutil.move(str(exported[i]), dst)
            pngs.append(dst)
        shutil.rmtree(tmp, ignore_errors=True)
        contact_sheet(pngs, out / "contact_sheet.jpg")
        if args.pdf:
            pres.SaveCopyAs(str(out / (deck.stem + ".pdf")), 32)  # ppSaveAsPDF
        issues = layout_checks(pres)
    finally:
        pres.Close()
    (out / "layout_report.json").write_text(json.dumps({"deck": deck.name, "issues": issues}, indent=1),
                                            encoding="utf-8")
    print(f"rendered {len(pngs)} slides to {out}; layout issues: {len(issues)}")
    for it in issues:
        print("  ", json.dumps(it, ensure_ascii=True))


if __name__ == "__main__":
    main()
