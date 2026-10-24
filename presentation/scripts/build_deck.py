"""Build the static deck with python-pptx: theme, layouts, slides, native charts and speaker notes.

Every number on a slide is computed here from the saved results in ``results/metrics`` (or, for the live-demo
slide, from the capture documented in ``presentation/assets/screenshots/CAPTURE_NOTES.md``). Claims that depend
on the data are guarded by assertions, so the build fails instead of printing a sentence the data no longer
supports.

Motion is NOT written here. Each slide declares its transition and animation steps in a plan
(``presentation/build/motion_plan.json``), and ``apply_motion.py`` applies that plan through PowerPoint itself.

Run from the project root:  python presentation/scripts/build_deck.py
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from lxml import etree
from pptx import Presentation
from pptx.chart.data import CategoryChartData, XyChartData
from pptx.dml.color import RGBColor
from pptx.enum.chart import (
    XL_AXIS_CROSSES,
    XL_CHART_TYPE,
    XL_LABEL_POSITION,
    XL_LEGEND_POSITION,
    XL_MARKER_STYLE,
    XL_TICK_LABEL_POSITION,
    XL_TICK_MARK,
)
from pptx.enum.dml import MSO_LINE_DASH_STYLE
from pptx.enum.shapes import MSO_CONNECTOR, MSO_SHAPE, PP_PLACEHOLDER
from pptx.enum.text import MSO_ANCHOR, MSO_AUTO_SIZE, PP_ALIGN
from pptx.opc.constants import RELATIONSHIP_TYPE as RT
from pptx.oxml.ns import qn
from pptx.util import Emu, Pt

ROOT = Path(__file__).resolve().parents[2]
PRES = ROOT / "presentation"
ASSETS = PRES / "assets"
SHOTS = ASSETS / "screenshots"
BUILD = PRES / "build"
M = ROOT / "results" / "metrics"

# ---------------------------------------------------------------------------------------------
# design tokens
# ---------------------------------------------------------------------------------------------
INK, SURF, SURF2 = "0A0E14", "111823", "172231"
LINE, LINE2 = "243245", "3A4D63"
TEXT, TEXT2, MUTED = "EAF0F6", "AAB6C3", "8291A2"
AMBER, CYAN, VIOLET, TEAL, CORAL, SLATE = "FFB547", "4CC9F0", "A78BFA", "3DDBB4", "FF6B6B", "6E7D8C"
TICKER_COLORS = {"SPY": CYAN, "AAPL": VIOLET, "MSFT": TEAL}

HEAD, BODY, LIGHT, MONO, SERIF = "Segoe UI Semibold", "Segoe UI", "Segoe UI Light", "Consolas", "Cambria"

SW, SH = 13.333, 7.5
X0, CW = 0.7, 11.933
XR = X0 + CW

A_NS = "http://schemas.openxmlformats.org/drawingml/2006/main"
P_NS = "http://schemas.openxmlformats.org/presentationml/2006/main"
R_NS = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"

TICKERS = ["SPY", "AAPL", "MSFT"]
# Numbers shown by the one live run captured for slide 16 (assets/screenshots/CAPTURE_NOTES.md).
LIVE = {"seconds": 8.2, "test_days": 1205, "test_end_label": "21 Oct 2026", "sharpe": 0.68, "bh_sharpe": 0.71,
        "acc_gap_pp": 1.9}
ML = ["Logistic Regression", "Random Forest", "XGBoost", "LSTM", "Ensemble (soft vote)"]
SHORT = {"Logistic Regression": "LR", "Random Forest": "RF", "XGBoost": "XGB", "LSTM": "LSTM", "Ensemble (soft vote)": "Ens"}
KEYS = {"Logistic Regression": "logistic_regression", "Random Forest": "random_forest", "XGBoost": "xgboost",
        "LSTM": "lstm", "Ensemble (soft vote)": "ensemble"}


def emu(v: float) -> Emu:
    return Emu(int(round(v * 914400)))


def rgb(h: str) -> RGBColor:
    return RGBColor.from_string(h)


def pct(x: float, d: int = 1) -> str:
    return f"{x * 100:.{d}f}%"


def signed(x: float, d: int = 2) -> str:
    s = f"{abs(x):.{d}f}"
    return ("+" if x > 0 else "−" if x < 0 else "") + s


# ---------------------------------------------------------------------------------------------
# facts: everything quoted on the slides
# ---------------------------------------------------------------------------------------------


def count_tests() -> int:
    out = subprocess.run([sys.executable, "-m", "pytest", "--collect-only", "-q"], cwd=ROOT, capture_output=True,
                         text=True, timeout=600)
    m = re.search(r"(\d+) tests? collected", out.stdout)
    if not m:
        raise RuntimeError("could not count tests:\n" + out.stdout[-2000:] + out.stderr[-2000:])
    return int(m.group(1))


class Facts:
    def __init__(self) -> None:
        rd = lambda n: pd.read_csv(M / n)  # noqa: E731
        self.trade = rd("trading_metrics_test.csv")
        cls = rd("classification_metrics.csv")
        self.cls = cls[cls.split == "test"]
        self.boot = rd("bootstrap_confidence_intervals.csv")
        self.wf = rd("walk_forward_trading.csv")
        self.cost = rd("robustness_cost_sensitivity.csv")
        self.var = rd("robustness_variants.csv")
        self.vol = rd("robustness_vol_target.csv")
        self.cdist = rd("class_distribution.csv")
        self.fdict = rd("feature_dictionary.csv")
        self.run = json.loads((M / "run_summary.json").read_text())
        self.ds = json.loads((M / "dataset_summary.json").read_text())
        self.params = json.loads((M / "best_params.json").read_text())
        self.thr = json.loads((M / "selected_thresholds.json").read_text())
        self.folds = json.loads((M / "walk_forward_folds.json").read_text())
        assert self.run["synthetic_data"] is False and self.run["is_default_full_run"] is True
        self._derive()

    def _derive(self) -> None:
        t, b = self.trade, self.boot
        bh = t[t.strategy == "Buy & Hold"].set_index("ticker").sharpe
        conf = t[t.strategy.str.endswith("(conf.)")].copy()
        conf["bh"] = conf.ticker.map(bh)
        self.bh_sharpe = bh
        self.n_ml = len(conf)
        self.n_higher = int((conf.sharpe > conf.bh + 1e-9).sum())
        self.n_identical = int(((conf.sharpe - conf.bh).abs() <= 1e-9).sum())
        assert (conf[(conf.sharpe - conf.bh).abs() <= 1e-9].exposure == 1.0).all()  # "invested every test day"

        ds_ = b[(b.statistic == "sharpe_minus_buy_and_hold") & b.strategy.str.endswith("(conf.)")].copy()
        ds_["model"] = ds_.strategy.str.replace(" (conf.)", "", regex=False)
        self.dsharpe = ds_
        higher = conf[conf.sharpe > conf.bh + 1e-9]
        h_ci = ds_.merge(higher[["ticker", "strategy"]], on=["ticker", "strategy"])
        assert len(h_ci) == self.n_higher and (h_ci.low <= 0).all()  # "every interval includes zero"
        self.sharpe_reliable = ds_[(ds_.low > 0) | (ds_.high < 0)]
        assert (self.sharpe_reliable.high < 0).all()  # the only reliable differences are worse than B&H

        acc = self.cls[self.cls.model.isin(ML)]
        self.acc_min, self.acc_max = acc.accuracy.min(), acc.accuracy.max()
        maj = self.cls[self.cls.model == "Majority class"].set_index("ticker").accuracy
        self.maj_min, self.maj_max, self.maj_spy = maj.min(), maj.max(), maj["SPY"]
        assert (self.cls[self.cls.model == "Majority class"].pred_share_up == 1.0).all()  # "always predict Up"
        am = t[t.strategy.str.endswith("(argmax)")].copy()
        am["bh"] = am.ticker.map(bh)
        self.n_argmax = len(am)
        self.argmax_higher = am[am.sharpe > am.bh + 1e-9][["ticker", "strategy", "sharpe", "bh"]].values.tolist()
        da = b[b.statistic == "accuracy_minus_majority_class"].copy()
        self.dacc = da
        self.acc_reliable = da[(da.low > 0) | (da.high < 0)]
        assert len(self.acc_reliable) == 1, self.acc_reliable
        r = self.acc_reliable.iloc[0]
        self.acc_rel_label, self.acc_rel_est = f"{r.ticker} {r.strategy}", r.estimate

        pers = t[(t.strategy == "Naive persistence") & (t.ticker == "SPY")].iloc[0]
        self.pers_spy_cum, self.pers_spy_trades = pers.cumulative_return, int(pers.n_trades)

        w = self.wf
        wbh = w[w.strategy == "Buy & Hold"].set_index("ticker").sharpe
        wma = w[w.strategy == "MA Crossover (10/50)"].set_index("ticker").sharpe
        wml = w[w.strategy.str.endswith("(conf.)")].copy()
        wml["bh"] = wml.ticker.map(wbh)
        self.wf_beat, self.wf_n = int((wml.sharpe > wml.bh + 1e-9).sum()), len(wml)
        wspy = w[w.ticker == "SPY"].set_index("strategy").cumulative_return
        assert all(wspy["Buy & Hold"] > wspy[k] for k in wspy.index if k.endswith("(conf.)"))  # slide 15 caption
        best = wml.loc[wml.groupby("ticker").sharpe.idxmax()].set_index("ticker")
        self.wf_table = {tk: (wbh[tk], wma[tk], best.loc[tk, "sharpe"], best.loc[tk, "strategy"]) for tk in TICKERS}

        c = self.cost

        def beat_at(bps: int) -> tuple[int, int]:
            s = c[(c.cost_bps == bps) & c.strategy.str.endswith("(conf.)")]
            bhc = c[(c.cost_bps == bps) & (c.strategy == "Buy & Hold")].set_index("ticker").sharpe
            return int((s.sharpe > s.ticker.map(bhc) + 1e-9).sum()), len(s)

        self.cost_bps = sorted(int(x) for x in c.cost_bps.unique())
        self.cost_beat_lo, self.cost_beat_hi = beat_at(self.cost_bps[0]), beat_at(self.cost_bps[-1])
        self.pers_cost = {tk: c[(c.strategy == "Naive persistence") & (c.ticker == tk)].sort_values("cost_bps")
                          .sharpe.tolist() for tk in TICKERS}
        self.bh_cost_spy = c[(c.strategy == "Buy & Hold") & (c.ticker == "SPY")].sort_values("cost_bps").sharpe.tolist()
        assert all(v[0] > 0 for v in self.pers_cost.values()) and all(v[-1] < 0 for v in self.pers_cost.values())

        v = self.vol
        fixed = v[v.sizing.str.startswith("fixed")].set_index(["ticker", "strategy"]).max_drawdown
        tgt = v[v.sizing.str.startswith("vol")].set_index(["ticker", "strategy"]).max_drawdown
        red = 1 - tgt / fixed
        self.vol_red = (red.min(), red.max())
        dsh = v[v.sizing.str.startswith("vol")].set_index(["ticker", "strategy"]).sharpe - \
            v[v.sizing.str.startswith("fixed")].set_index(["ticker", "strategy"]).sharpe
        assert dsh.abs().max() < 0.1  # "at a similar Sharpe"
        self.vol_label = v[v.sizing.str.startswith("vol")].sizing.iloc[0].replace("vol target ", "")

        var = self.var
        vnames = list(var.variant.unique())
        default_v = next(x for x in vnames if x.startswith("default"))
        moc_v = next(x for x in vnames if "market-on-close" in x)
        bin_v = next(x for x in vnames if "binary" in x)

        def vrow(variant: str, tk: str, strat: str) -> float:
            return float(var[(var.variant == variant) & (var.ticker == tk) & (var.strategy == strat)].sharpe.iloc[0])

        def vbeat(variant: str):
            s = var[(var.variant == variant) & (var.kind == "ml_confidence")]
            bhv = var[(var.variant == variant) & (var.strategy == "Buy & Hold")].set_index("ticker").sharpe
            win = s[s.sharpe > s.ticker.map(bhv) + 1e-9]
            return len(win), len(s), set(zip(win.ticker, win.strategy))

        self.moc_from = vrow(default_v, "AAPL", "Random Forest (conf.)")
        self.moc_to = vrow(moc_v, "AAPL", "Random Forest (conf.)")
        vb = {k: vbeat(x) for k, x in (("default", default_v), ("moc", moc_v), ("binary", bin_v))}
        self.var_n = vb["default"][1]
        self.var_lo = min(vb["moc"][0], vb["binary"][0])
        self.var_hi = max(vb["moc"][0], vb["binary"][0])
        assert vb["moc"][2] != vb["default"][2] and vb["binary"][2] != vb["default"][2]  # "the winners change"

        bb = b[(b.statistic == "sharpe") & (b.strategy == "Buy & Hold")].set_index("ticker")
        self.bh_ci_spy = (bb.loc["SPY", "estimate"], bb.loc["SPY", "low"], bb.loc["SPY", "high"])

        cd = self.cdist[(self.cdist.ticker == "SPY") & (self.cdist.split == "test")].set_index("class").frequency
        self.spy_test_share = {k: float(cd[k]) for k in ("Up", "Down", "Neutral")}

        vals = [x for tk in self.thr.values() for x in tk.values()]
        self.thr_min, self.thr_max = min(vals), max(vals)
        grid = self.run["config"]["signals"]["threshold_grid"]
        self.grid_n, self.grid_min, self.grid_max = len(grid), min(grid), max(grid)
        self.tau_spy_xgb = self.thr["SPY"]["xgboost"]

        bt = self.run["config"]["backtest"]
        self.fee_bps, self.slip_bps = bt["fee_bps"], bt["slippage_bps"]
        self.cost_total = self.fee_bps + self.slip_bps
        self.capital = bt["initial_capital"]
        self.neutral = self.run["config"]["target"]["neutral_threshold"]

        sp = self.ds["SPY"]["split"]
        self.split = sp
        self.env = self.run["environment"]
        self.minutes = round(self.run["total_seconds"] / 60)
        self.seed = self.run["config"]["experiment"]["seed"]
        boot_cfg = self.run["config"]["robustness"]["bootstrap"]
        self.n_boot, self.block = boot_cfg["n_boot"], boot_cfg["block_len"]
        self.fgroups = self.fdict.groupby("group").size().to_dict()
        self.n_features = len(self.fdict)
        assert self.n_features == self.ds["SPY"]["n_features"]
        self.n_figures = len(list((ROOT / "results" / "figures").glob("*.png")))
        self.n_notebooks = len(list((ROOT / "notebooks").glob("*.ipynb")))
        self.n_tests = count_tests()
        self.wf_years = sorted({f["test_year"] for f in self.folds})


# ---------------------------------------------------------------------------------------------
# low-level drawing helpers
# ---------------------------------------------------------------------------------------------


def _strip_style(shape) -> None:
    st = shape._element.find(qn("p:style"))
    if st is not None:
        shape._element.remove(st)


def _alpha(parent, alpha: float | None) -> None:
    """Set the alpha of the srgbClr inside ``parent`` (a solidFill element)."""
    clr = parent.find(qn("a:srgbClr"))
    for a in clr.findall(qn("a:alpha")):
        clr.remove(a)
    if alpha is not None and alpha < 1:
        etree.SubElement(clr, qn("a:alpha")).set("val", str(int(alpha * 100000)))


def _effects(spPr, *, glow: tuple[str, float, float] | None = None, shadow: bool = False) -> None:
    lst = etree.SubElement(spPr, qn("a:effectLst"))
    if glow:
        color, rad, alpha = glow
        g = etree.SubElement(lst, qn("a:glow"), rad=str(int(rad * 12700)))
        c = etree.SubElement(g, qn("a:srgbClr"), val=color)
        etree.SubElement(c, qn("a:alpha"), val=str(int(alpha * 100000)))
    if shadow:
        s = etree.SubElement(lst, qn("a:outerShdw"), blurRad="381000", dist="114300", dir="5400000", algn="t",
                             rotWithShape="0")
        c = etree.SubElement(s, qn("a:srgbClr"), val="000000")
        etree.SubElement(c, qn("a:alpha"), val="60000")


def rect(shapes, x, y, w, h, *, fill=None, alpha=None, line=None, line_w=0.75, line_alpha=None, r=0.0, name=None,
         kind=None):
    kind = kind or (MSO_SHAPE.ROUNDED_RECTANGLE if r else MSO_SHAPE.RECTANGLE)
    s = shapes.add_shape(kind, emu(x), emu(y), emu(w), emu(h))
    _strip_style(s)
    if r and kind == MSO_SHAPE.ROUNDED_RECTANGLE:
        s.adjustments[0] = min(0.5, r / min(w, h))
    if fill:
        s.fill.solid()
        s.fill.fore_color.rgb = rgb(fill)
        _alpha(s._element.spPr.find(qn("a:solidFill")), alpha)
    else:
        s.fill.background()
    if line:
        s.line.color.rgb = rgb(line)
        s.line.width = Pt(line_w)
        _alpha(s._element.spPr.find(qn("a:ln")).find(qn("a:solidFill")), line_alpha)
    else:
        s.line.fill.background()
    tf = s.text_frame
    tf.margin_left = tf.margin_right = tf.margin_top = tf.margin_bottom = 0
    if name:
        s.name = name
    return s


ALIGN = {"l": PP_ALIGN.LEFT, "c": PP_ALIGN.CENTER, "r": PP_ALIGN.RIGHT}
ANCHOR = {"t": MSO_ANCHOR.TOP, "m": MSO_ANCHOR.MIDDLE, "b": MSO_ANCHOR.BOTTOM}


def _bullet(p, color: str) -> None:
    pPr = p._p.get_or_add_pPr()
    pPr.set("marL", str(int(0.24 * 914400)))
    pPr.set("indent", str(int(-0.24 * 914400)))
    etree.SubElement(etree.SubElement(pPr, qn("a:buClr")), qn("a:srgbClr"), val=color)
    etree.SubElement(pPr, qn("a:buSzPct"), val="100000")
    etree.SubElement(pPr, qn("a:buFont"), typeface="Arial")
    etree.SubElement(pPr, qn("a:buChar"), char="•")


def fill(tf, paras, *, size=16, color=TEXT2, font=BODY, bold=False, italic=False, align="l", after=0, line=None,
         spc=None) -> None:
    d = dict(size=size, color=color, font=font, bold=bold, italic=italic, spc=spc)
    if isinstance(paras, (str, tuple, dict)):
        paras = [paras]
    for i, item in enumerate(paras):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        popts = item if isinstance(item, dict) else {}
        runs = popts.get("runs", popts.get("text")) if isinstance(item, dict) else item
        if isinstance(runs, (str, tuple)):
            runs = [runs]
        p.alignment = ALIGN[popts.get("align", align)]
        p.space_after = Pt(popts.get("after", after))
        if popts.get("before"):
            p.space_before = Pt(popts["before"])
        ls = popts.get("line", line)
        if ls:
            p.line_spacing = ls
        if popts.get("bullet"):
            _bullet(p, popts["bullet"])
        for run in runs:
            txt, o = (run, {}) if isinstance(run, str) else run
            o = {**d, **{k: v for k, v in popts.items() if k in d}, **o}
            r = p.add_run()
            r.text = txt
            f = r.font
            f.size, f.name, f.bold, f.italic = Pt(o["size"]), o["font"], o["bold"], o["italic"]
            f.color.rgb = rgb(o["color"])
            rPr = r._r.get_or_add_rPr()
            if o.get("spc"):
                rPr.set("spc", str(int(o["spc"] * 100)))
            if o.get("baseline"):
                rPr.set("baseline", str(o["baseline"]))


def text(shapes, x, y, w, h, paras, *, anchor="t", name=None, wrap=True, inset=0.0, **kw):
    tb = shapes.add_textbox(emu(x), emu(y), emu(w), emu(h))
    tf = tb.text_frame
    tf.word_wrap = wrap
    tf.auto_size = MSO_AUTO_SIZE.NONE
    tf.margin_left = tf.margin_right = tf.margin_top = tf.margin_bottom = emu(inset)
    tf.vertical_anchor = ANCHOR[anchor]
    fill(tf, paras, **kw)
    if name:
        tb.name = name
    return tb


def shape_text(s, paras, *, anchor="m", **kw) -> None:
    tf = s.text_frame
    tf.word_wrap = True
    tf.auto_size = MSO_AUTO_SIZE.NONE
    tf.vertical_anchor = ANCHOR[anchor]
    fill(tf, paras, **kw)


def line(shapes, x1, y1, x2, y2, *, color=LINE2, w=1.0, head=False, name=None, alpha=None, dash=False):
    c = shapes.add_connector(MSO_CONNECTOR.STRAIGHT, emu(x1), emu(y1), emu(x2), emu(y2))
    _strip_style(c)
    c.line.color.rgb = rgb(color)
    c.line.width = Pt(w)
    ln = c.line._get_or_add_ln()
    _alpha(ln.find(qn("a:solidFill")), alpha)
    if dash:
        c.line.dash_style = MSO_LINE_DASH_STYLE.DASH
    if head:
        etree.SubElement(ln, qn("a:tailEnd"), type="triangle", w="med", len="med")
    if name:
        c.name = name
    return c


def polyline(shapes, pts, *, color, w, name=None, alpha=None, glow=None, head=False):
    fb = shapes.build_freeform(int(emu(pts[0][0])), int(emu(pts[0][1])), scale=1.0)
    fb.add_line_segments([(int(emu(x)), int(emu(y))) for x, y in pts[1:]], close=False)
    s = fb.convert_to_shape()
    _strip_style(s)
    s.fill.background()
    s.line.color.rgb = rgb(color)
    s.line.width = Pt(w)
    ln = s._element.spPr.find(qn("a:ln"))
    _alpha(ln.find(qn("a:solidFill")), alpha)
    ln.set("cap", "rnd")
    etree.SubElement(ln, qn("a:round"))
    if head:
        etree.SubElement(ln, qn("a:tailEnd"), type="triangle", w="med", len="med")
    if glow:
        _effects(s._element.spPr, glow=glow)
    if name:
        s.name = name
    return s


def card(shapes, x, y, w, h, name, *, accent=False, fill_=SURF, alpha=0.88, r=0.12):
    g = shapes.add_group_shape()
    g.name = name
    rect(g.shapes, x, y, w, h, fill=fill_, alpha=alpha, line=AMBER if accent else LINE,
         line_alpha=0.75 if accent else None, line_w=1.0 if accent else 0.75, r=r, name=f"{name}_bg")
    return g


def picture(shapes, path: Path, x, y, w, h, *, name, alt, radius_pct=2.2):
    pic = shapes.add_picture(str(path), emu(x), emu(y), emu(w), emu(h))
    pic.auto_shape_type = MSO_SHAPE.ROUNDED_RECTANGLE
    av = pic._element.spPr.find(qn("a:prstGeom")).find(qn("a:avLst"))
    etree.SubElement(av, qn("a:gd"), name="adj", fmla=f"val {int(radius_pct * 1000)}")
    pic.line.color.rgb = rgb(LINE2)
    pic.line.width = Pt(1)
    _effects(pic._element.spPr, shadow=True)
    pic.name = name
    pic._element.nvPicPr.cNvPr.set("descr", alt)
    return pic


def equation(shapes, x, y, w, h, lines, *, size=24, name=None, align="c"):
    """lines: list of lists of (text, style) where style in {'v' (italic var), 's' (subscript), 'o' (operator)}."""
    paras = []
    for ln_ in lines:
        runs = []
        for txt, st in ln_:
            o = {"font": SERIF, "color": TEXT, "size": size}
            if st in ("v", "vs"):
                o["italic"] = True
            if st in ("s", "vs"):
                o["baseline"] = -25000
                o["size"] = size
            runs.append((txt, o))
        paras.append({"runs": runs, "after": 6})
    return text(shapes, x, y, w, h, paras, anchor="m", name=name, align=align)


# ---------------------------------------------------------------------------------------------
# charts
# ---------------------------------------------------------------------------------------------


def _chart_frame(chart, *, legend: bool, legend_pos=XL_LEGEND_POSITION.TOP) -> None:
    chart.font.name = BODY
    chart.font.size = Pt(12)
    chart.font.color.rgb = rgb(TEXT2)
    chart.has_title = False
    chart.has_legend = legend
    if legend:
        chart.legend.position = legend_pos
        chart.legend.include_in_layout = False
        chart.legend.font.size = Pt(12)
        chart.legend.font.color.rgb = rgb(TEXT2)
    cs = chart._chartSpace
    sp = etree.fromstring(f'<c:spPr xmlns:c="http://schemas.openxmlformats.org/drawingml/2006/chart" '
                          f'xmlns:a="{A_NS}"><a:noFill/><a:ln><a:noFill/></a:ln></c:spPr>')
    old = cs.find(qn("c:spPr"))
    if old is not None:
        cs.remove(old)
    cs.find(qn("c:chart")).addnext(sp)
    pa = cs.find(qn("c:chart")).find(qn("c:plotArea"))
    psp = etree.fromstring(f'<c:spPr xmlns:c="http://schemas.openxmlformats.org/drawingml/2006/chart" '
                           f'xmlns:a="{A_NS}"><a:noFill/><a:ln><a:noFill/></a:ln></c:spPr>')
    ext = pa.find(qn("c:extLst"))
    (ext.addprevious(psp) if ext is not None else pa.append(psp))


def _axes(chart, *, vmin, vmax, major, fmt, cat_low=True, cat_size=12) -> None:
    va, ca = chart.value_axis, chart.category_axis
    va.minimum_scale, va.maximum_scale, va.major_unit = vmin, vmax, major
    va.has_major_gridlines = True
    gl = va.major_gridlines.format.line
    gl.color.rgb = rgb(LINE)
    gl.width = Pt(0.75)
    va.format.line.fill.background()
    va.major_tick_mark = va.minor_tick_mark = XL_TICK_MARK.NONE
    va.tick_labels.number_format = fmt
    va.tick_labels.number_format_is_linked = False
    va.tick_labels.font.size = Pt(11)
    va.tick_labels.font.color.rgb = rgb(MUTED)
    ca.format.line.color.rgb = rgb(LINE2)
    ca.format.line.width = Pt(1.25)
    ca.major_tick_mark = ca.minor_tick_mark = XL_TICK_MARK.NONE
    ca.tick_labels.font.size = Pt(cat_size)
    ca.tick_labels.font.color.rgb = rgb(TEXT2)
    if cat_low:
        ca.tick_label_position = XL_TICK_LABEL_POSITION.LOW


def _err_bars(series, plus, minus, color="9AA8B7", w=1.0) -> None:
    eb = etree.Element(qn("c:errBars"))
    etree.SubElement(eb, qn("c:errBarType"), val="both")
    etree.SubElement(eb, qn("c:errValType"), val="cust")
    etree.SubElement(eb, qn("c:noEndCap"), val="0")
    for tag, vals in (("c:plus", plus), ("c:minus", minus)):
        lit = etree.SubElement(etree.SubElement(eb, qn(tag)), qn("c:numLit"))
        etree.SubElement(lit, qn("c:formatCode")).text = "General"
        etree.SubElement(lit, qn("c:ptCount"), val=str(len(vals)))
        for i, v in enumerate(vals):
            etree.SubElement(etree.SubElement(lit, qn("c:pt"), idx=str(i)), qn("c:v")).text = repr(float(max(v, 0.0)))
    ln = etree.SubElement(etree.SubElement(eb, qn("c:spPr")), qn("a:ln"), w=str(int(w * 12700)))
    etree.SubElement(etree.SubElement(ln, qn("a:solidFill")), qn("a:srgbClr"), val=color)
    series._element.find(qn("c:cat")).addprevious(eb)


def _series_fill(series, color: str) -> None:
    if hasattr(series, "invert_if_negative"):
        series.invert_if_negative = False
    series.format.fill.solid()
    series.format.fill.fore_color.rgb = rgb(color)
    series.format.line.fill.background()


# ---------------------------------------------------------------------------------------------
# motion plan (applied later through PowerPoint COM by apply_motion.py)
# ---------------------------------------------------------------------------------------------


class Motion:
    def __init__(self) -> None:
        self.slides: list[dict] = []

    def slide(self, transition: str = "fade", duration: float = 0.7) -> None:
        self.cur = {"transition": transition, "duration": duration, "steps": []}
        self.slides.append(self.cur)

    def add(self, shape: str, effect: str = "rise", trigger: str = "after", delay: float = 0.0, dur: float = 0.5,
            **kw) -> None:
        self.cur["steps"].append({"shape": shape, "effect": effect, "trigger": trigger, "delay": round(delay, 3),
                                  "dur": dur, **kw})

    def stagger(self, shapes: list[str], effect: str = "rise", start: str = "after", gap: float = 0.12,
                dur: float = 0.5, first_delay: float = 0.0, **kw) -> None:
        for i, s in enumerate(shapes):
            self.add(s, effect, start if i == 0 else "with", first_delay + (0 if i == 0 else gap * i), dur, **kw)


# ---------------------------------------------------------------------------------------------
# the deck
# ---------------------------------------------------------------------------------------------


class Deck:
    def __init__(self) -> None:
        self.prs = Presentation()
        self.prs.slide_width, self.prs.slide_height = emu(SW), emu(SH)
        self.motion = Motion()
        self.notes: list[str] = []
        self._theme()
        self._master_and_layouts()

    # -- theme, master, layouts -----------------------------------------------------------------
    def _theme(self) -> None:
        part = self.prs.slide_master.part.part_related_by(RT.THEME)
        root = etree.fromstring(part.blob)
        ns = {"a": A_NS}
        root.set("name", "ML Signals")
        cs = root.find(".//a:clrScheme", ns)
        cs.set("name", "ML Signals Dark")
        colors = dict(dk1=INK, lt1=TEXT, dk2=SURF, lt2=TEXT2, accent1=AMBER, accent2=CYAN, accent3=VIOLET,
                      accent4=TEAL, accent5=CORAL, accent6=SLATE, hlink=CYAN, folHlink=VIOLET)
        for tag, val in colors.items():
            el = cs.find(f"a:{tag}", ns)
            for ch in list(el):
                el.remove(ch)
            etree.SubElement(el, qn("a:srgbClr"), val=val)
        fs = root.find(".//a:fontScheme", ns)
        fs.set("name", "ML Signals")
        fs.find("a:majorFont/a:latin", ns).set("typeface", HEAD)
        fs.find("a:minorFont/a:latin", ns).set("typeface", BODY)
        part._blob = etree.tostring(root, xml_declaration=True, encoding="UTF-8", standalone=True)

    def _set_bg(self, element, part, image: Path) -> None:
        _, rid = part.get_or_add_image_part(str(image))
        cSld = element.find(qn("p:cSld"))
        old = cSld.find(qn("p:bg"))
        if old is not None:
            cSld.remove(old)
        bg = etree.fromstring(
            f'<p:bg xmlns:p="{P_NS}" xmlns:a="{A_NS}" xmlns:r="{R_NS}"><p:bgPr><a:blipFill dpi="0" rotWithShape="1">'
            f'<a:blip r:embed="{rid}"/><a:srcRect/><a:stretch><a:fillRect/></a:stretch></a:blipFill><a:effectLst/>'
            f"</p:bgPr></p:bg>")
        cSld.insert(0, bg)

    @staticmethod
    def _place(ph, x, y, w, h) -> None:
        ph.left, ph.top, ph.width, ph.height = emu(x), emu(y), emu(w), emu(h)

    def _layout_text(self, layout, name, x, y, w, h, para_xml: str) -> None:
        tree = layout.shapes._spTree
        nid = max(int(e.get("id")) for e in tree.iter(qn("p:cNvPr"))) + 1
        sp = etree.fromstring(
            f'<p:sp xmlns:p="{P_NS}" xmlns:a="{A_NS}"><p:nvSpPr><p:cNvPr id="{nid}" name="{name}"/>'
            f'<p:cNvSpPr txBox="1"/><p:nvPr userDrawn="1"/></p:nvSpPr><p:spPr><a:xfrm>'
            f'<a:off x="{int(emu(x))}" y="{int(emu(y))}"/><a:ext cx="{int(emu(w))}" cy="{int(emu(h))}"/></a:xfrm>'
            f'<a:prstGeom prst="rect"><a:avLst/></a:prstGeom><a:noFill/></p:spPr><p:txBody>'
            f'<a:bodyPr wrap="none" lIns="0" tIns="0" rIns="0" bIns="0" anchor="t"/><a:lstStyle/>{para_xml}</p:txBody></p:sp>')
        tree.append(sp)

    def _master_and_layouts(self) -> None:
        master = self.prs.slide_master
        clr = master._element.find(qn("p:clrMap"))
        for k, v in dict(bg1="dk1", tx1="lt1", bg2="dk2", tx2="lt2").items():
            clr.set(k, v)
        self._set_bg(master._element, master.part, ASSETS / "backgrounds" / "bg_content.jpg")
        for ph in master.placeholders:
            if ph.placeholder_format.type == PP_PLACEHOLDER.TITLE:
                self._place(ph, X0, 0.82, CW, 0.8)
                bp = ph.text_frame._txBody.find(qn("a:bodyPr"))
                for k, v in dict(anchor="t", lIns="0", tIns="0", rIns="0", bIns="0").items():
                    bp.set(k, v)
                for ch in list(bp):
                    bp.remove(ch)
                etree.SubElement(bp, qn("a:noAutofit"))
            elif ph.placeholder_format.type == PP_PLACEHOLDER.BODY:
                self._place(ph, X0, 1.95, CW, 4.7)
        styles = master._element.find(qn("p:txStyles"))
        t1 = styles.find(qn("p:titleStyle")).find(qn("a:lvl1pPr"))
        t1.set("algn", "l")
        t1.find(qn("a:defRPr")).set("sz", "3600")
        b1 = styles.find(qn("p:bodyStyle")).find(qn("a:lvl1pPr"))
        b1.find(qn("a:defRPr")).set("sz", "1800")

        keep = {"Title Slide": "Hero", "Title Only": "Content (title only)", "Title and Content": "Title and content",
                "Blank": "Blank"}
        layouts = master.slide_layouts
        for lay in list(layouts):
            if lay.name not in keep:
                layouts.remove(lay)
        for lay in layouts:
            new_name = keep[lay.name]
            lay._element.find(qn("p:cSld")).set("name", new_name)
            if new_name == "Hero":
                self.hero_layout = lay
                self._set_bg(lay._element, lay.part, ASSETS / "backgrounds" / "bg_hero.jpg")
                for ph in list(lay.placeholders):
                    if ph.placeholder_format.type == PP_PLACEHOLDER.CENTER_TITLE:
                        self._place(ph, X0, 1.3, 10.0, 1.95)
                    elif ph.placeholder_format.type == PP_PLACEHOLDER.SUBTITLE:
                        self._place(ph, X0, 3.3, 9.5, 0.6)
                    else:
                        ph._element.getparent().remove(ph._element)
                continue
            for ph in list(lay.placeholders):
                t = ph.placeholder_format.type
                if t == PP_PLACEHOLDER.TITLE:
                    self._place(ph, X0, 0.82, CW, 0.8)
                elif t in (PP_PLACEHOLDER.BODY, PP_PLACEHOLDER.OBJECT):
                    self._place(ph, X0, 1.95, CW, 4.7)
                else:  # date / footer / slide-number placeholders from the stock 4:3 template
                    ph._element.getparent().remove(ph._element)
            if new_name == "Blank":
                continue
            if new_name == "Content (title only)":
                self.content_layout = lay
            self._layout_text(
                lay, "Footer", X0, 7.0, 6.0, 0.22,
                f'<a:p><a:r><a:rPr lang="en-US" sz="1000" spc="150"><a:solidFill><a:srgbClr val="{MUTED}"/>'
                f'</a:solidFill><a:latin typeface="{MONO}"/></a:rPr><a:t>ML TRADING SIGNALS · MIT MANIPAL · 2026</a:t>'
                f"</a:r></a:p>")
            self._layout_text(
                lay, "Slide number", XR - 1.0, 7.0, 1.0, 0.22,
                f'<a:p><a:pPr algn="r"/><a:fld id="{{B6F15528-21DE-4FAA-801E-634DDDAF4B2B}}" type="slidenum">'
                f'<a:rPr lang="en-US" sz="1000"><a:solidFill><a:srgbClr val="{MUTED}"/></a:solidFill>'
                f'<a:latin typeface="{MONO}"/></a:rPr><a:t>‹#›</a:t></a:fld></a:p>')

    # -- slide factories ------------------------------------------------------------------------
    def content(self, eyebrow: str, title: str, *, transition="fade", duration=0.7):
        s = self.prs.slides.add_slide(self.content_layout)
        s.shapes.title.text_frame.text = title
        s.shapes.title.name = "title"
        text(s.shapes, X0, 0.5, 9.0, 0.26, eyebrow, size=11, color=AMBER, font=MONO, spc=2, name="eyebrow")
        self.motion.slide(transition, duration)
        return s

    def hero(self, *, transition="fade", duration=0.9):
        s = self.prs.slides.add_slide(self.hero_layout)
        self.motion.slide(transition, duration)
        return s

    def note(self, slide, txt: str) -> None:
        slide.notes_slide.notes_text_frame.text = txt
        self.notes.append(txt)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.prs.save(path)


# ---------------------------------------------------------------------------------------------
# shared motif: the SPY price trace (real data, see make_assets.py)
# ---------------------------------------------------------------------------------------------


def trace_points(x0, x1, top, bottom):
    df = pd.read_csv(ASSETS / "data" / "spy_trace.csv")
    v = np.log(df["value"].to_numpy())
    t = np.linspace(0, 1, len(v))
    xs = x0 + t * (x1 - x0)
    ys = bottom - (v - v.min()) / (v.max() - v.min()) * (bottom - top)
    return list(zip(xs, ys))


def draw_trace(slide, x0, x1, top, bottom, *, alpha=None, w=2.25):
    pts = trace_points(x0, x1, top, bottom)
    polyline(slide.shapes, pts, color=AMBER, w=w, name="!!trace", alpha=alpha, glow=(AMBER, 6, 0.28 if alpha else 0.4))
    ex, ey = pts[-1]
    d = 0.17 if not alpha else 0.12
    dot = rect(slide.shapes, ex - d / 2, ey - d / 2, d, d, fill=AMBER, kind=MSO_SHAPE.OVAL, name="!!dot",
               alpha=alpha)
    _effects(dot._element.spPr, glow=(AMBER, 8, 0.45))


# ---------------------------------------------------------------------------------------------
# slides
# ---------------------------------------------------------------------------------------------


def build(F: Facts) -> Deck:  # noqa: C901 - one function per slide would only scatter the layout constants
    D = Deck()
    mo = D.motion

    # 1 ---------------------------------------------------------------------------------- title
    s = D.hero()
    ttl = s.shapes.title
    ttl.name = "title"
    ttl.text_frame.clear()
    ttl.text_frame.margin_left = ttl.text_frame.margin_right = ttl.text_frame.margin_top = 0
    ttl.text_frame.vertical_anchor = MSO_ANCHOR.TOP
    fill(ttl.text_frame, ["Machine-Learning", "Trading Signals"], size=60, color=TEXT, font=HEAD, line=0.92)
    sub = s.placeholders[1]
    sub.name = "subtitle"
    sub.text_frame.clear()
    sub.text_frame.margin_left = sub.text_frame.margin_top = 0
    sub.text_frame.vertical_anchor = MSO_ANCHOR.TOP
    fill(sub.text_frame, "A comparative study against technical baseline strategies", size=24, color=TEXT2,
         font=LIGHT)
    text(s.shapes, X0, 0.8, 11.5, 0.3,
         "MINI-PROJECT  ·  B.TECH CSE (AI & ML)  ·  MANIPAL INSTITUTE OF TECHNOLOGY", size=12, color=AMBER,
         font=MONO, spc=1.5, name="eyebrow")
    text(s.shapes, X0, 4.3, 6.8, 1.3, [
        {"runs": [("Ishan Singh  ·  Kenneth Martin", {"font": HEAD, "size": 18, "color": TEXT})], "after": 6},
        {"runs": [("Mentors   ", {"font": MONO, "size": 11, "color": MUTED}),
                  ("Dr. Narendra V G  ·  Dr. Ashwath Rao B", {"size": 14, "color": TEXT2})], "after": 4},
        {"runs": [("School of Computer Engineering  ·  2026", {"size": 12, "color": MUTED})]},
    ], name="people")
    draw_trace(s, X0, XR, 4.75, 6.85)
    text(s.shapes, XR - 4.5, 7.0, 4.5, 0.22, "SPY · daily adjusted close · 2010–2025 (data used in this study)",
         size=10, color=MUTED, align="r", name="trace_caption")
    mo.stagger(["eyebrow", "title", "subtitle"], "rise", gap=0.15, dur=0.6)
    mo.add("!!trace", "wipe", "after", dur=1.6, direction="left")
    mo.add("trace_caption", "fade", "with", delay=0.6, dur=0.6)
    mo.add("!!dot", "zoom", "after", dur=0.35)
    mo.add("people", "rise", "with", delay=0.1, dur=0.6)
    D.note(s, (
        "Good morning. We are Ishan Singh and Kenneth Martin, and our mini-project asks a simple question: can machine "
        "learning produce better trading signals than the simple rules traders have used for decades? The amber line "
        "is real data: the daily SPY price from 2010 to 2025, the same data our models were trained and tested on. "
        "Over the next fifteen minutes we will show how we built the system, how we tested it honestly, a live demo, "
        "and what the evidence actually says."))

    # 2 ---------------------------------------------------------------------------------- big idea
    s = D.content("THE BIG IDEA", "Can a model learn when to be in the market?", transition="morph", duration=1.4)
    for i, (nm, label, code, desc, accent) in enumerate([
        ("card_rule", "RULE-BASED", ["if SMA(10) > SMA(50):  long", "else:                  cash"],
         "Written by hand and fixed forever: the classic moving-average crossover baseline.", False),
        ("card_learned", "LEARNED", ["if P(Up | 42 features) ≥ τ:  long", "else:                        cash"],
         "Five model families learn P(Up) from 2011–2019 data; the threshold τ is chosen on 2020–2021.", True),
    ]):
        x = X0 if i == 0 else 7.03
        g = card(s.shapes, x, 2.0, 5.6, 3.0, nm, accent=accent)
        text(g.shapes, x + 0.4, 2.3, 4.8, 0.3, label, size=12, color=AMBER if accent else MUTED, font=MONO, spc=2)
        text(g.shapes, x + 0.4, 2.85, 4.8, 0.9, [{"text": c_, "after": 2} for c_ in code], size=17, color=TEXT,
             font=MONO)
        text(g.shapes, x + 0.4, 3.95, 4.8, 0.8, desc, size=14, color=TEXT2, line=1.1)
    vs = rect(s.shapes, 6.415, 3.24, 0.5, 0.5, fill=INK, line=LINE2, kind=MSO_SHAPE.OVAL, name="vs")
    shape_text(vs, "vs", size=13, color=MUTED, align="c")
    text(s.shapes, X0, 5.22, CW, 0.66, [
        {"runs": [("Same data · same execution · same costs — so any difference comes from ", {}),
                  ("the signal itself.", {"color": AMBER, "font": HEAD})], "after": 4},
        {"runs": [("Tested on SPY (an S&P 500 ETF), Apple (AAPL) and Microsoft (MSFT): daily prices 2010–2025 from "
                   "Yahoo Finance.", {"size": 12.5, "color": MUTED})]}], size=17, color=TEXT, name="statement")
    draw_trace(s, X0, XR, 5.98, 6.72, alpha=0.55, w=1.75)
    mo.add("card_rule", "rise", "after", delay=0.15, dur=0.55)
    mo.add("vs", "zoom", "with", delay=0.35, dur=0.4)
    mo.add("card_learned", "rise", "with", delay=0.5, dur=0.55)
    mo.add("statement", "fade", "after", delay=0.1, dur=0.6)
    D.note(s, (
        "Here is the core idea. On the left is a rule a human writes once: buy when the 10-day average is above the "
        "50-day average. On the right, instead of a fixed rule, a model looks at 42 features and estimates the "
        "probability that the next move is up, and we invest only when that probability clears a threshold tau. "
        "The important design decision is fairness: both approaches get exactly the same data, the same execution "
        "timing and the same trading costs, so any difference really comes from the signal. We test this on three "
        "liquid instruments: SPY, an exchange-traded fund that tracks the S&P 500, plus Apple and Microsoft, using "
        "daily prices from 2010 to 2025."))

    # 3 ---------------------------------------------------------------------------------- problem
    s = D.content("01 · THE PROBLEM", "Four easy ways to fool yourself in a backtest")
    traps = [
        ("Look-ahead bias", "Features or labels that quietly use tomorrow's data, so the backtest trades on "
                            "information nobody had.", "→ too-good accuracy"),
        ("Leaky validation", "Random K-fold on a time series lets the model train on the future of the days it "
                             "is tested on.", "→ optimistic CV scores"),
        ("Free trading", "Ignoring fees, slippage and when orders actually fill. High-turnover strategies die on "
                         "costs.", "→ paper-only profits"),
        ("Weak baselines", f"A model at {F.acc_max * 100:.0f}% sounds better than a coin flip, but 'always "
                           f"predict Up' already scores {pct(F.maj_spy)} on SPY.", "→ base-rate 'skill'"),
    ]
    names = []
    for i, (head, body, tag) in enumerate(traps):
        x = X0 + i * (2.78 + 0.271)
        g = card(s.shapes, x, 2.0, 2.78, 3.55, f"trap_{i + 1}")
        text(g.shapes, x + 0.3, 2.3, 1.0, 0.3, f"0{i + 1}", size=13, color=AMBER, font=MONO)
        text(g.shapes, x + 0.3, 2.72, 2.2, 0.45, head, size=19, color=TEXT, font=HEAD)
        text(g.shapes, x + 0.3, 3.28, 2.2, 1.65, body, size=14, color=TEXT2, line=1.1)
        text(g.shapes, x + 0.3, 5.03, 2.2, 0.3, tag, size=11, color=MUTED, font=MONO)
        names.append(f"trap_{i + 1}")
    text(s.shapes, X0, 5.9, CW, 0.4, "Each trap inflates results. The framework was designed to close all four.",
         size=16, color=TEXT2, name="bottom")
    mo.stagger(names, "rise", gap=0.12, dur=0.5)
    mo.add("bottom", "fade", "after", dur=0.5)
    D.note(s, (
        "Why is this hard? Because a trading backtest is very easy to get wrong in ways that make results look "
        "better than they are. First, look-ahead bias, where some feature or label secretly uses future data. Second, "
        "random cross-validation on time series, which lets the model learn from the future of the days it is "
        f"tested on. Third, ignoring trading costs and execution timing. And fourth, weak baselines: a model at "
        f"{F.acc_max * 100:.0f} percent sounds better than a coin flip, but simply predicting 'up' every day already "
        f"scores {pct(F.maj_spy)} on SPY. "
        "Our framework was designed to close all four of these traps."))

    # 4 ---------------------------------------------------------------------------------- why it matters
    s = D.content("02 · WHY IT MATTERS", "Small evaluation choices flip the conclusion")
    text(s.shapes, X0, 1.95, 7.2, 0.3, "Naive persistence — Sharpe ratio vs. one-way cost (bps), test 2022–2025",
         size=13, color=TEXT2, name="chart_head")
    cd = XyChartData()
    for name_, vals in [(f"Persistence · {tk}", F.pers_cost[tk]) for tk in TICKERS] + [("Buy-and-hold · SPY",
                                                                                         F.bh_cost_spy)]:
        sr = cd.add_series(name_)
        for b_, v_ in zip(F.cost_bps, vals):
            sr.add_data_point(b_, v_)
    gf = s.shapes.add_chart(XL_CHART_TYPE.XY_SCATTER_LINES, emu(X0), emu(2.3), emu(7.2), emu(3.85), cd)
    gf.name = "chart_cost"
    ch = gf.chart
    _chart_frame(ch, legend=True, legend_pos=XL_LEGEND_POSITION.RIGHT)
    _axes(ch, vmin=-5, vmax=1, major=1, fmt="0")
    xa = ch.category_axis  # on an XY chart this is the numeric x axis, drawn to scale
    xa.minimum_scale, xa.maximum_scale, xa.major_unit = 0, F.cost_bps[-1], 10
    xa.tick_labels.number_format, xa.tick_labels.number_format_is_linked = "0", False
    xa.has_major_gridlines = False
    ch.category_axis.has_title = True
    ch.category_axis.axis_title.text_frame.text = "one-way cost per unit turnover (bps)"
    f_ = ch.category_axis.axis_title.text_frame.paragraphs[0].runs[0].font
    f_.size, f_.color.rgb, f_.bold, f_.name = Pt(11), rgb(MUTED), False, BODY
    for ser, col in zip(ch.plots[0].series, [CYAN, VIOLET, TEAL, TEXT]):
        ser.smooth = False
        ser.format.line.color.rgb = rgb(col)
        ser.format.line.width = Pt(2.25)
        ser.marker.style = XL_MARKER_STYLE.CIRCLE
        ser.marker.size = 7
        ser.marker.format.fill.solid()
        ser.marker.format.fill.fore_color.rgb = rgb(col)
        ser.marker.format.line.color.rgb = rgb(INK)
    last = ch.plots[0].series[3]
    last.format.line.dash_style = MSO_LINE_DASH_STYLE.DASH
    last.marker.style = XL_MARKER_STYLE.NONE
    text(s.shapes, X0, 6.22, 7.2, 0.5, "Persistence (‘tomorrow repeats today’) is profitable before costs and deeply "
         "negative after them. AAPL and MSFT nearly coincide.", size=13, color=MUTED, name="chart_caption")
    est, lo, hi = F.bh_ci_spy
    for i, (nm, big, col, label) in enumerate([
        ("stat_exec", f"{F.moc_from:.2f} → {F.moc_to:.2f}", AMBER,
         "AAPL Random Forest Sharpe when fills move from the next open to the same day's close (sensitivity run)"),
        ("stat_noise", f"{signed(lo)} … {hi:.2f}", TEXT,
         f"95% bootstrap interval for SPY buy-and-hold's Sharpe ratio ({est:.2f}) over the four test years"),
    ]):
        y = 2.05 + i * 2.2
        g = s.shapes.add_group_shape()
        g.name = nm
        text(g.shapes, 8.35, y, 4.28, 0.9, big, size=46, color=col, font=LIGHT)
        text(g.shapes, 8.35, y + 0.95, 4.28, 0.85, label, size=14, color=TEXT2, line=1.1)
    rf = F.run["config"]["backtest"]["risk_free_rate"]
    text(s.shapes, 8.35, 6.18, 4.28, 0.55, f"Sharpe ratio = annualised mean return ÷ annualised volatility (risk-free "
         f"rate {rf:g}). 1 bp (basis point) = 0.01%.", size=11.5, color=MUTED, line=1.05, name="definitions")
    mo.add("chart_head", "fade", "after", dur=0.4)
    mo.add("chart_cost", "wipe", "with", delay=0.1, dur=1.2, direction="left")
    mo.add("chart_caption", "fade", "after", dur=0.4)
    mo.add("stat_exec", "rise", "after", delay=0.1, dur=0.55)
    mo.add("stat_noise", "rise", "after", delay=0.1, dur=0.55)
    mo.add("definitions", "fade", "after", dur=0.4)
    D.note(s, (
        "Here is why this matters, using our own results. The chart shows a naive 'tomorrow repeats today' strategy. "
        "A quick definition first: the Sharpe ratio is the annualised return divided by the annualised volatility, so "
        "it measures return per unit of risk, and a basis point is a hundredth of a percent. With zero costs this "
        "strategy actually has a positive Sharpe ratio on all three tickers; at realistic costs it becomes "
        "deeply negative. On the right: the same Random Forest on Apple goes from a Sharpe of "
        f"{F.moc_from:.2f} to {F.moc_to:.2f} just by changing the assumed fill price from the next open to the same "
        "day's close. And the 95 percent interval on buy-and-hold's own Sharpe ratio runs from "
        f"{signed(lo)} to {hi:.2f}. Four years of data is a small sample, so we need statistics, not single numbers."))

    # 5 ---------------------------------------------------------------------------------- objectives
    s = D.content("03 · OBJECTIVES", "What we set out to do")
    objectives = [
        ("Build moving-average crossover and buy-and-hold baselines in one backtester with realistic "
         "transaction costs.", "DELIVERED", TEAL),
        ("Engineer SMA, EMA, RSI, MACD and Bollinger-band features for XGBoost and LSTM directional signals.",
         "DELIVERED", TEAL),
        ("Compare ML signals with the baselines on predictive and risk-adjusted metrics across market "
         "conditions.", "DELIVERED", TEAL),
        ("Stretch goal: test whether news sentiment found by retrieval-augmented generation (RAG) makes the "
         "signals more robust.", "FUTURE WORK", AMBER),
    ]
    names = []
    for i, (txt, status, col) in enumerate(objectives):
        y = 1.95 + i * 1.12
        g = card(s.shapes, X0, y, CW, 0.95, f"obj_{i + 1}")
        text(g.shapes, X0 + 0.35, y, 0.6, 0.95, f"0{i + 1}", size=16, color=AMBER, font=MONO, anchor="m")
        text(g.shapes, X0 + 1.05, y, 8.65, 0.95, txt, size=16, color=TEXT, anchor="m", line=1.05)
        chip = rect(g.shapes, 10.6, y + 0.29, 1.75, 0.37, fill=col, alpha=0.12, line=col, line_alpha=0.8, r=0.18)
        shape_text(chip, status, size=11, color=col, font=MONO, align="c", spc=1)
        names.append(f"obj_{i + 1}")
    text(s.shapes, X0, 6.42, CW, 0.3, [[("BEYOND THE SYNOPSIS   ", {"font": MONO, "size": 10, "color": AMBER}),
          ("Logistic Regression · Random Forest · soft-vote ensemble · walk-forward evaluation · bootstrap CIs · "
           "live dashboard", {})]], size=12, color=MUTED, name="beyond")
    mo.stagger(names, "rise", gap=0.12)
    mo.add("beyond", "fade", "after", dur=0.5)
    D.note(s, (
        "These are the four objectives from our synopsis. The first three are delivered in full: technical "
        "baselines in one cost-aware backtester, technical-indicator features for XGBoost and an LSTM, and a "
        "comparison on both prediction and trading metrics. The fourth, a news-sentiment layer built with retrieval-augmented generation, or RAG, was a stretch "
        "goal. We deliberately did not implement it, because doing it correctly needs a time-stamped news archive "
        "for 2010 to 2025; fake or leaky sentiment would invalidate the whole comparison. We return to it in future "
        "scope. We also went beyond the synopsis with three extra models, walk-forward testing, bootstrap intervals "
        "and a live dashboard."))

    # 6 ---------------------------------------------------------------------------------- solution
    s = D.content("04 · OUR SOLUTION", "A leakage-safe, cost-aware comparison framework")
    fixes = [
        ("Look-ahead bias", "Point-in-time features. ", "Scale-invariant ratios only; tests recompute them on "
                                                       "truncated data and require identical values."),
        ("Leaky validation", "Chronological splits. ", "Train → validation → test with a 2-day purge; purged CV "
                                                      "for the tabular models, a purged hold-out slice for the LSTM."),
        ("Free trading", "Realistic execution. ", f"Decide at the close, fill at the next open, and pay "
                                                  f"{F.cost_total}\u00a0bps ({F.fee_bps}\u00a0fee + "
                                                  f"{F.slip_bps}\u00a0slippage) "
                                                  f"on every unit of turnover."),
        ("Weak baselines", "Baselines and intervals. ", "Majority class, persistence, MA crossover and "
                                                           "buy-and-hold, with block-bootstrap 95% CIs."),
    ]
    for i, (trap, lead, rest) in enumerate(fixes):
        y = 1.95 + i * 1.12
        g = card(s.shapes, X0, y, 3.15, 0.88, f"sol_trap_{i + 1}", fill_=SURF2)
        text(g.shapes, X0 + 0.3, y, 2.7, 0.88, [[(f"0{i + 1}  ", {"font": MONO, "size": 12, "color": AMBER}),
                                               (trap, {})]], size=15, color=TEXT2, anchor="m")
        line(s.shapes, 3.98, y + 0.44, 4.55, y + 0.44, color=AMBER, w=1.5, head=True, name=f"sol_arrow_{i + 1}")
        g2 = card(s.shapes, 4.65, y, 7.983, 0.88, f"sol_fix_{i + 1}", accent=True)
        text(g2.shapes, 4.95, y, 7.4, 0.88, [[(lead, {"font": HEAD, "color": TEXT}), (rest, {})]], size=15,
             color=TEXT2, anchor="m", line=1.05)
    text(s.shapes, X0, 6.4, CW, 0.35, "Every main result uses all four safeguards; sensitivity runs relax one at a "
         "time and are labelled.",
         size=14, color=MUTED, name="bottom")
    first = True
    for i in range(4):
        base = 0.0 if first else i * 0.3
        mo.add(f"sol_trap_{i + 1}", "rise", "after" if first else "with", delay=base, dur=0.45)
        mo.add(f"sol_arrow_{i + 1}", "wipe", "with", delay=base + 0.15, dur=0.35, direction="left")
        mo.add(f"sol_fix_{i + 1}", "rise", "with", delay=base + 0.25, dur=0.45)
        first = False
    mo.add("bottom", "fade", "after", dur=0.5)
    D.note(s, (
        "Our solution maps one safeguard to each trap. Against look-ahead bias, every feature is a point-in-time, "
        "scale-invariant ratio, and our tests recompute features on truncated data and require identical values. "
        "Against leaky validation, splits are strictly chronological with a two-day purge, and tuning uses purged "
        "time-series cross-validation for the tabular models and a purged hold-out slice for the LSTM. Against free "
        "trading, we decide at the close, fill at the next open, "
        f"and charge {F.cost_total} basis points on every unit of turnover. Against weak baselines, we compare "
        "with four baselines and report bootstrap confidence intervals."))

    # 7 ---------------------------------------------------------------------------------- architecture
    s = D.content("05 · ARCHITECTURE", "System architecture: from raw prices to decisions")
    nodes = [
        ("Data", "Yahoo Finance OHLCV, validated and adjusted", "data_loader.py"),
        ("Features", f"{F.n_features} point-in-time, scale-invariant indicators", "features.py"),
        ("Targets", "Down / Neutral / Up from next-open returns", "targets.py"),
        ("Split", "Chronological train / val / test + purge", "preprocessing.py"),
        ("Models", "LR · RF · XGBoost · LSTM · ensemble", "models.py · lstm_model.py"),
        ("Signal rule", "τ chosen on validation; walk-forward runner", "experiment.py"),
        ("Backtester", f"Next-open fills, {F.cost_total} bps per unit turnover", "backtester.py"),
        ("Evaluation", "Metrics, bootstrap CIs, calibration", "evaluation.py"),
    ]
    nw, gap, nh = 2.66, 0.4243, 1.42
    pos = []
    for i, (nm, det, mod) in enumerate(nodes):
        row, col = divmod(i, 4)
        x, y = X0 + col * (nw + gap), (1.95 if row == 0 else 3.95)
        pos.append((x, y))
        g = card(s.shapes, x, y, nw, nh, f"node_{i + 1}", accent=(nm == "Models"))
        text(g.shapes, x + 0.25, y + 0.2, nw - 0.5, 0.36, [[(f"0{i + 1}  ", {"font": MONO, "size": 12,
                                                                             "color": AMBER}),
                                                           (nm, {"font": HEAD, "size": 16, "color": TEXT})]])
        text(g.shapes, x + 0.25, y + 0.6, nw - 0.5, 0.5, det, size=12.5, color=TEXT2, line=1.05)
        text(g.shapes, x + 0.25, y + 1.1, nw - 0.5, 0.24, mod, size=10.5, color=MUTED, font=MONO)
    for i in (0, 1, 2, 4, 5, 6):
        (x, y), (x2, _) = pos[i], pos[i + 1]
        line(s.shapes, x + nw + 0.06, y + nh / 2, x2 - 0.06, y + nh / 2, color=LINE2, w=1.5, head=True,
             name=f"arr_{i + 1}")
    x4, x5 = pos[3][0] + nw / 2, pos[4][0] + nw / 2
    polyline(s.shapes, [(x4, 1.95 + nh + 0.04), (x4, 3.66), (x5, 3.66), (x5, 3.95 - 0.04)], color=LINE2, w=1.5,
             head=True, name="ret")
    x8 = pos[7][0] + nw / 2
    line(s.shapes, x8, 3.95 + nh + 0.04, x8, 5.82 - 0.04, color=LINE2, w=1.5, head=True, name="down")
    g = card(s.shapes, X0, 5.82, CW, 0.66, "outputs", fill_=SURF2)
    text(g.shapes, X0 + 0.3, 5.82, CW - 0.6, 0.66, [[
        ("OUTPUTS    ", {"font": MONO, "size": 12, "color": AMBER}),
        (f"result tables · {F.n_figures} figures · experiment report · IEEE report · interactive dashboard",
         {"size": 14, "color": TEXT}),
        ("      reporting.py · report/ · app/", {"font": MONO, "size": 11, "color": MUTED})]], anchor="m")
    mo.stagger(["node_1", "arr_1", "node_2", "arr_2", "node_3", "arr_3", "node_4"], "rise", gap=0.1, dur=0.45)
    mo.stagger(["ret", "node_5", "arr_5", "node_6", "arr_6", "node_7", "arr_7", "node_8"], "rise", start="click",
               gap=0.1, dur=0.45)
    mo.stagger(["down", "outputs"], "rise", start="click", gap=0.15, dur=0.45)
    D.note(s, (
        "This is the architecture; each box maps to modules in our src folder. The first row prepares the data: "
        "download and validate daily prices from Yahoo Finance, compute 42 point-in-time features, build the "
        "Down, Neutral, Up target, and split chronologically with a purge. [click] The second row is learning and "
        "verification: five model families produce probabilities, a threshold rule turns them into positions, the "
        "backtester simulates next-open fills with costs, and the evaluation module computes metrics, bootstrap "
        "intervals and calibration; the experiment runner also drives the walk-forward and robustness runs. [click] Everything ends in generated outputs: tables, figures, "
        "reports and the dashboard. Nothing in the results is typed by hand."))

    # 8 ---------------------------------------------------------------------------------- tech stack
    s = D.content("06 · TECH STACK", "Built on the open-source Python stack")
    env = F.env
    stack = [
        ("DATA", "Acquire & engineer", [("Python", env["python"]), ("pandas", env["pandas"]),
                                        ("NumPy", env["numpy"]), ("yfinance", "Yahoo Finance OHLCV")]),
        ("MODELLING", "Learn P(Up)", [("scikit-learn", env["scikit_learn"]), ("XGBoost", env["xgboost"]),
                                      ("PyTorch", env["torch"].replace("+cpu", " · CPU"))]),
        ("EVALUATION", "Measure honestly", [("SciPy", "statistics"), ("matplotlib", f"{F.n_figures} figures"),
                                            ("pytest", f"{F.n_tests} tests")]),
        ("DELIVERY", "Show & share", [("Streamlit", "interactive dashboard"), ("Plotly", "dashboard charts"),
                                      ("Jupyter", f"{F.n_notebooks} executed notebooks"),
                                      ("python-docx", "IEEE report builder")]),
    ]
    names = []
    for i, (label, purpose, items) in enumerate(stack):
        x = X0 + i * (2.78 + 0.271)
        g = card(s.shapes, x, 1.95, 2.78, 4.15, f"stack_{i + 1}")
        text(g.shapes, x + 0.3, 2.25, 2.2, 0.28, label, size=12, color=AMBER, font=MONO, spc=2)
        text(g.shapes, x + 0.3, 2.58, 2.2, 0.36, purpose, size=17, color=TEXT, font=HEAD)
        line(g.shapes, x + 0.3, 3.15, x + 2.48, 3.15, color=LINE, w=0.75)
        for j, (nm, ver) in enumerate(items):
            yy = 3.35 + j * 0.66
            text(g.shapes, x + 0.3, yy, 2.2, 0.3, nm, size=15, color=TEXT)
            text(g.shapes, x + 0.3, yy + 0.3, 2.2, 0.25, ver, size=11, color=MUTED, font=MONO)
        names.append(f"stack_{i + 1}")
    text(s.shapes, X0, 6.32, CW, 0.3, "Versions as recorded for the reported run (results/metrics/run_summary.json). "
         f"One configuration file, config.yaml, with a fixed seed ({F.seed}).", size=12, color=MUTED, name="caption")
    mo.stagger(names, "rise", gap=0.12)
    mo.add("caption", "fade", "after", dur=0.4)
    D.note(s, (
        "The stack is entirely open source. pandas and NumPy for data, scikit-learn for Logistic Regression and "
        "Random Forest, XGBoost for gradient boosting and PyTorch for the LSTM. Evaluation uses SciPy and "
        f"matplotlib, and the code is covered by {F.n_tests} pytest tests. For delivery we built a Streamlit "
        "dashboard with Plotly charts, executed Jupyter notebooks, and a script that generates the IEEE report. "
        "The versions shown are exactly those recorded for the reported run."))

    # 9 ---------------------------------------------------------------------------------- split timeline
    ax0, ax1, y0 = 1.65, 12.55, 2011.0

    def X(date: str | float) -> float:
        if isinstance(date, str):
            ts = pd.Timestamp(date)
            date = ts.year + (ts.dayofyear - 1) / (366 if ts.is_leap_year else 365)
        return ax0 + (date - y0) * (ax1 - ax0) / 15.0

    def timeline_axis(slide) -> None:
        for yr in range(2011, 2027):
            text(slide.shapes, X(yr) - 0.3, 2.0, 0.6, 0.24, str(yr), size=10, color=MUTED, font=MONO, align="c",
                 name=f"!!yr{yr}")
        line(slide.shapes, ax0, 2.36, ax1, 2.36, color=LINE2, w=1.0, name="!!axis")

    seg_style = {"train": dict(fill=CYAN, alpha=0.32, line=CYAN, line_alpha=0.85),
                 "val": dict(fill=AMBER, alpha=0.9, line=AMBER),
                 "test": dict(fill=TEXT, alpha=0.92, line=TEXT)}

    def seg(slide, kind, start, end, y, h, name):
        x1_, x2_ = X(start) + 0.015, X(end) - 0.015
        return rect(slide.shapes, x1_, y, x2_ - x1_, h, r=min(0.06, h / 2), name=name, **seg_style[kind])

    s = D.content("07 · METHODOLOGY", "Time only flows forward: chronological splits")
    timeline_axis(s)
    sp = F.split
    seg(s, "train", sp["train"]["start"], sp["train"]["end"], 2.74, 0.62, "!!seg_train")
    seg(s, "val", sp["val"]["start"], sp["val"]["end"], 2.74, 0.62, "!!seg_val")
    seg(s, "test", sp["test"]["start"], sp["test"]["end"], 2.74, 0.62, "!!seg_test")
    text(s.shapes, X(sp["train"]["start"]) + 0.2, 2.74, 4.0, 0.62,
         [[("TRAIN  ", {"font": MONO, "size": 12, "color": CYAN}), (f"{sp['train']['rows']:,} days", {})]],
         size=13, color=TEXT, font=HEAD, anchor="m", name="lbl_train")
    text(s.shapes, X(sp["val"]["start"]) + 0.12, 2.74, 1.3, 0.62, f"VAL · {sp['val']['rows']}", size=12,
         color=INK, font=HEAD, anchor="m", name="lbl_val")
    text(s.shapes, X(sp["test"]["start"]) + 0.2, 2.74, 2.6, 0.62,
         [[("TEST  ", {"font": MONO, "size": 12}), (f"{sp['test']['rows']:,} days", {})]], size=13, color=INK,
         font=HEAD, anchor="m", name="lbl_test")
    for i, b_ in enumerate((sp["val"]["start"], sp["test"]["start"])):
        text(s.shapes, X(b_) - 0.6, 2.45, 1.2, 0.24, "2-day purge", size=10, color=AMBER, font=MONO, align="c",
             name=f"purge_{i + 1}")
    roles = [
        ("TRAIN", CYAN, f"{sp['train']['start'][:4]}–{sp['train']['end'][:4]}",
         "Fit every model. Tabular models are tuned with purged time-series CV (4 folds, gap 2) on log loss; the "
         "LSTM on a purged hold-out slice."),
        ("VALIDATION", AMBER, f"{sp['val']['start'][:4]}–{sp['val']['end'][:4]}",
         f"Choose ensemble members (log loss) and the threshold τ from {F.grid_n} values (Sharpe)."),
        ("TEST", TEXT, f"{sp['test']['start'][:4]}–{sp['test']['end'][:4]}",
         "Never used for fitting or for choosing τ. Main results come from here; robustness checks re-use it and "
         "are labelled."),
    ]
    names = []
    for i, (lab, col, yrs, body) in enumerate(roles):
        x = X0 + i * (3.8 + 0.2665)
        g = card(s.shapes, x, 3.9, 3.8, 2.05, f"role_{lab.lower()}")
        text(g.shapes, x + 0.3, 4.15, 3.2, 0.28, [[(lab, {"color": col}), (f"  ·  {yrs}", {"color": MUTED})]],
             size=12, font=MONO, spc=1)
        text(g.shapes, x + 0.3, 4.55, 3.2, 1.28, body, size=14, color=TEXT2, line=1.1)
        names.append(f"role_{lab.lower()}")
    text(s.shapes, X0, 6.15, CW, 0.5, "Rows are never shuffled. Each label looks two opens ahead, so the last two "
         "rows before every boundary are purged from the earlier segment.", size=13, color=MUTED, name="note")
    mo.add("!!axis", "wipe", "after", dur=0.6, direction="left")
    mo.stagger([f"!!yr{y}" for y in range(2011, 2027)], "fade", start="with", gap=0.03, dur=0.3)
    mo.add("!!seg_train", "wipe", "after", dur=0.8, direction="left")
    mo.add("lbl_train", "fade", "with", delay=0.4, dur=0.4)
    mo.add("purge_1", "fade", "after", dur=0.3)
    mo.add("!!seg_val", "wipe", "with", delay=0.1, dur=0.35, direction="left")
    mo.add("lbl_val", "fade", "with", delay=0.25, dur=0.3)
    mo.add("purge_2", "fade", "after", dur=0.3)
    mo.add("!!seg_test", "wipe", "with", delay=0.1, dur=0.5, direction="left")
    mo.add("lbl_test", "fade", "with", delay=0.3, dur=0.3)
    mo.stagger(names, "rise", gap=0.12)
    mo.add("note", "fade", "after", dur=0.4)
    D.note(s, (
        f"Time only flows forward, so our splits do too. We train on {sp['train']['start'][:4]} to "
        f"{sp['train']['end'][:4]}, which is {sp['train']['rows']:,} trading days, validate on 2020 and 2021, and keep "
        f"{sp['test']['rows']:,} test days from 2022 to 2025 that are never used for fitting or for choosing anything. "
        "Hyperparameters are tuned inside the training period only: purged time-series cross-validation for the "
        "tabular models, and a purged hold-out slice for the LSTM. The validation period "
        "chooses the trading threshold and the ensemble members. Because each label looks two opens ahead, we purge "
        "the last two rows before each boundary so no label overlaps the next segment."))

    # 10 --------------------------------------------------------------------------------- walk-forward (morph)
    s = D.content("07 · METHODOLOGY", "Walk-forward: retrain every year, trade the next", transition="morph",
                  duration=1.5)
    timeline_axis(s)
    spy_folds = {f["test_year"]: f for f in F.folds if f["ticker"] == "SPY"}
    years = F.wf_years
    for i, yr in enumerate(years):
        f = spy_folds[yr]
        y, h = 2.62 + i * 0.37, 0.27
        last = yr == years[-1]
        seg(s, "train", f["train_start"], f["train_end"], y, h, "!!seg_train" if last else f"wf_{yr}_train")
        seg(s, "val", f["val_start"], f["val_end"], y, h, "!!seg_val" if last else f"wf_{yr}_val")
        seg(s, "test", f"{yr}-01-02", f"{yr}-12-31", y, h, "!!seg_test" if last else f"wf_{yr}_test")
        text(s.shapes, X0, y, 0.72, h, str(yr), size=11, color=TEXT2, font=MONO, align="r", anchor="m",
             name=f"wf_lbl_{yr}")
    lg = s.shapes.add_group_shape()
    lg.name = "legend"
    xx = X0
    for kind, lab in (("train", "Train, expanding"), ("val", "Validation: τ and ensemble"),
                      ("test", "Test year, traded out-of-sample")):
        rect(lg.shapes, xx, 6.49, 0.3, 0.16, r=0.04, **seg_style[kind])
        wl = 0.092 * len(lab) + 0.15
        text(lg.shapes, xx + 0.4, 6.42, wl, 0.3, lab, size=12, color=TEXT2)
        xx += 0.4 + wl + 0.35
    text(s.shapes, 9.6, 6.42, XR - 9.6, 0.3, f"{len(years)} folds × 3 tickers", size=12, color=MUTED, align="r",
         name="legend_r")
    mo.add("legend", "fade", "after", delay=0.1, dur=0.5)
    mo.add("legend_r", "fade", "with", delay=0.2, dur=0.5)
    D.note(s, (
        "A single split could just be lucky, so we also ran a walk-forward test. The bars you just saw move to the "
        f"bottom row and become one of {len(years)} folds. For every test year from {years[0]} to {years[-1]}, we "
        "retrain on all data up to two years before it, choose the threshold and the ensemble on the year just before "
        "it, and then trade the test year out-of-sample. The tabular models are re-tuned in every fold; the LSTM keeps "
        "its architecture and re-tunes only its class weighting. The training window grows each year, like a real "
        "trader who keeps learning. This gives ten years of out-of-sample trading for each ticker."))

    # 11 --------------------------------------------------------------------------------- features & target
    s = D.content("08 · IMPLEMENTATION", f"{F.n_features} point-in-time features, one executable target")
    text(s.shapes, X0, 1.95, 5.65, 0.26, f"{F.n_features} FEATURES · {len(F.fgroups)} GROUPS", size=11, color=AMBER,
         font=MONO, spc=2, name="feat_label")
    groups = [("returns", "Returns", "lags 1, 2, 3 and 5 days, momentum 5–60 days, overnight and intraday moves, range"),
              ("technical", "Technical", "close/SMA and close/EMA (5–50), RSI(14), MACD, Bollinger %B and width"),
              ("volatility", "Volatility", "ATR(14), realised volatility 5/20/60 days, vol ratio, normalised return"),
              ("volume", "Volume", "relative volume (5 and 20 days), volume z-score"),
              ("regime", "Regime", "volatility percentile, close/SMA200, golden cross, 252-day drawdown")]
    ramp = ["4CC9F0", "36A3C8", "2A82A3", "21657F", "1A4B5F"]
    gb = s.shapes.add_group_shape()
    gb.name = "featbar"
    xx = X0
    for (key, label, _), col in zip(groups, ramp):
        wseg = 5.65 * F.fgroups[key] / F.n_features
        r_ = rect(gb.shapes, xx, 2.32, wseg - 0.03, 0.5, fill=col, r=0.05)
        shape_text(r_, str(F.fgroups[key]), size=13, color=INK if col in ramp[:2] else TEXT, font=HEAD, align="c")
        xx += wseg
    names = []
    for i, ((key, label, ex), col) in enumerate(zip(groups, ramp)):
        y = 3.05 + i * 0.56
        g = s.shapes.add_group_shape()
        g.name = f"fg_{i + 1}"
        rect(g.shapes, X0, y + 0.07, 0.16, 0.16, fill=col, r=0.03)
        text(g.shapes, X0 + 0.3, y, 1.5, 0.3, f"{label} · {F.fgroups[key]}", size=14, color=TEXT, font=HEAD)
        text(g.shapes, 2.55, y + 0.01, 3.8, 0.52, ex, size=12.5, color=TEXT2, line=1.0)
        names.append(f"fg_{i + 1}")
    text(s.shapes, X0, 5.98, 5.65, 0.65, "Only ratios, never price levels, so back-adjusted prices cannot leak the "
         "future. A test recomputes features on truncated data and requires identical values.", size=12,
         color=MUTED, line=1.05, name="feat_note")
    text(s.shapes, 6.95, 1.95, 5.68, 0.26, "TARGET · WHAT THE MODEL PREDICTS", size=11, color=AMBER, font=MONO,
         spc=2, name="tgt_label")
    tl = s.shapes.add_group_shape()
    tl.name = "timeline"
    nx = [7.3, 9.55, 11.8]
    line(tl.shapes, 7.1, 2.95, 12.3, 2.95, color=LINE2, w=1.5)
    line(tl.shapes, nx[1], 2.95, nx[2], 2.95, color=AMBER, w=3.0)
    for j, (xn, top, bot) in enumerate(zip(nx, ["close t", "open t+1", "open t+2"],
                                           ["features known, predict", "enter, or stay in cash",
                                            "exit, or re-decide"])):
        rect(tl.shapes, xn - 0.1, 2.85, 0.2, 0.2, fill=INK, line=AMBER if j else TEXT, line_w=1.75, kind=MSO_SHAPE.OVAL)
        text(tl.shapes, xn - 0.8, 2.42, 1.6, 0.28, top, size=12, color=TEXT, font=MONO, align="c")
        text(tl.shapes, xn - 0.85, 3.18, 1.7, 0.5, bot, size=12, color=TEXT2, align="c", line=1.0)
    g = card(s.shapes, 6.95, 3.95, 5.68, 1.0, "eq_target", accent=True)
    equation(g.shapes, 6.95, 3.95, 5.68, 1.0, [[("r", "v"), ("t", "vs"), ("  =  ", "o"), ("O", "v"), ("t+2", "vs"),
                                               (" / ", "o"), ("O", "v"), ("t+1", "vs"), ("  −  1", "o")]], size=26)
    nt = F.neutral * 100
    chips = [("DOWN", CORAL, f"r < −{nt:.1f}%"), ("NEUTRAL", SLATE, f"|r| ≤ {nt:.1f}%"), ("UP", TEAL, f"r > +{nt:.1f}%")]
    cg = s.shapes.add_group_shape()
    cg.name = "chips_cls"
    for j, (lab, col, rule) in enumerate(chips):
        c_ = rect(cg.shapes, 6.95 + j * 1.94, 5.15, 1.8, 0.58, fill=col, alpha=0.14, line=col, line_alpha=0.8, r=0.12)
        shape_text(c_, [[(f"{lab}  ", {"font": MONO, "size": 11, "color": col}), (rule, {"font": SERIF})]], size=14,
                   color=TEXT, align="c")
    sh = F.spy_test_share
    text(s.shapes, 6.95, 5.92, 5.68, 0.3, f"SPY test days: Up {pct(sh['Up'])} · Down {pct(sh['Down'])} · Neutral "
         f"{pct(sh['Neutral'])}", size=12, color=MUTED, name="cls_share")
    mo.add("feat_label", "fade", "after", dur=0.3)
    mo.add("featbar", "wipe", "with", delay=0.1, dur=0.8, direction="left")
    mo.stagger(names, "rise", start="with", gap=0.1, first_delay=0.4, dur=0.45)
    mo.add("feat_note", "fade", "after", dur=0.4)
    mo.add("tgt_label", "fade", "after", dur=0.3)
    mo.add("timeline", "wipe", "with", delay=0.1, dur=0.8, direction="left")
    mo.add("eq_target", "rise", "after", dur=0.5)
    mo.add("chips_cls", "rise", "after", dur=0.45)
    mo.add("cls_share", "fade", "with", delay=0.2, dur=0.4)
    D.note(s, (
        f"On the left are the {F.n_features} features in five groups: returns, technical indicators such as RSI, "
        "MACD and Bollinger bands, volatility, volume and market regime. All of them are ratios rather than price "
        "levels, so adjusted prices cannot leak future information, and a test checks this. On the right is the "
        "target. After the close on day t we compute features and predict; we can only trade at the next open, "
        "t plus one, and the position is held until the following open. So the return we predict is open t+2 over "
        f"open t+1, split into Down, Neutral and Up with a plus or minus {nt:.1f} percent band."))

    # 12 --------------------------------------------------------------------------------- models
    s = D.content("08 · IMPLEMENTATION", "Five model families, tuned without peeking")
    P = F.params

    def per_ticker(model: str, key: str, fmt=str) -> str:
        vals = [fmt(P[tk][model][key]) for tk in TICKERS]
        return vals[0] if len(set(vals)) == 1 else "/".join(vals)

    def members(tk: str) -> str:
        inv = {v: SHORT[k] for k, v in KEYS.items()}
        return " + ".join(inv[m] for m in P[tk]["ensemble"]["members"])

    models = [
        ("LR", "Logistic Regression", "Linear, L2-regularised",
         [f"C = {per_ticker('logistic_regression', 'C', lambda v: f'{v:g}')}", "all three tickers"]),
        ("RF", "Random Forest", f"{P['SPY']['random_forest']['n_estimators']} bagged trees",
         [f"depth {per_ticker('random_forest', 'max_depth')}",
          f"min leaf {per_ticker('random_forest', 'min_samples_leaf')}"]),
        ("XGB", "XGBoost", "Gradient-boosted trees",
         [f"depth {per_ticker('xgboost', 'max_depth')}", f"η {per_ticker('xgboost', 'learning_rate', lambda v: f'{v:g}')}",
          f"{per_ticker('xgboost', 'n_estimators')} trees"]),
        ("LSTM", "LSTM network", f"PyTorch, {P['SPY']['lstm']['num_layers']} layer",
         [f"hidden {per_ticker('lstm', 'hidden_size')}", f"window {per_ticker('lstm', 'seq_len')} days"]),
        ("ENS", "Soft-vote ensemble", "Two members, chosen on validation",
         [f"{tk}: {members(tk)}" for tk in TICKERS]),
    ]
    names = []
    for i, (ab, nm, what, sel) in enumerate(models):
        x = X0 + i * (2.17 + 0.2708)
        g = card(s.shapes, x, 1.95, 2.17, 3.2, f"model_{i + 1}", accent=(ab == "XGB"))
        text(g.shapes, x + 0.25, 2.2, 1.6, 0.28, ab, size=13, color=AMBER, font=MONO, spc=1)
        text(g.shapes, x + 0.25, 2.52, 1.75, 0.65, nm, size=17, color=TEXT, font=HEAD, line=0.95)
        text(g.shapes, x + 0.25, 3.2, 1.75, 0.5, what, size=13, color=TEXT2, line=1.0)
        line(g.shapes, x + 0.25, 3.78, x + 1.92, 3.78, color=LINE, w=0.75)
        text(g.shapes, x + 0.25, 3.9, 1.75, 0.24, "SELECTED", size=10, color=MUTED, font=MONO, spc=2)
        text(g.shapes, x + 0.25, 4.17, 1.75, 0.9, [{"text": t_, "after": 2} for t_ in sel], size=12, color=TEXT,
             font=MONO)
        names.append(f"model_{i + 1}")
    text(s.shapes, X0, 5.24, CW, 0.45, "Where values differ they are shown as SPY / AAPL / MSFT. Tabular models are "
         "tuned on log loss with purged time-series CV; the LSTM is tuned and early-stopped on a purged slice at the "
         "end of the training period.",
         size=12, color=MUTED, line=1.05, name="legend")
    g = card(s.shapes, X0, 5.8, 7.6, 0.85, "insight", fill_=SURF2)
    text(g.shapes, X0 + 0.3, 5.8, 7.0, 0.85, [[("INSIGHT   ", {"font": MONO, "size": 11, "color": AMBER}),
          ("Tuning picked the most regularised option in almost every grid: there is little stable signal to fit.",
           {})]], size=14, color=TEXT, anchor="m", line=1.05)
    g = card(s.shapes, 8.55, 5.8, XR - 8.55, 0.85, "baselines", fill_=SURF2)
    text(g.shapes, 8.8, 5.8, XR - 9.05, 0.85, [[("BASELINES   ", {"font": MONO, "size": 11, "color": AMBER}),
          ("majority class · naive persistence · MA crossover (10/50) · buy-and-hold", {})]], size=12.5,
         color=TEXT2, anchor="m", line=1.05)
    mo.stagger(names, "rise", gap=0.1)
    mo.add("legend", "fade", "after", dur=0.4)
    mo.stagger(["insight", "baselines"], "rise", gap=0.15, dur=0.45)
    lr_c = per_ticker("logistic_regression", "C", lambda v: f"{v:g}")
    D.note(s, (
        "We trained five model families: Logistic Regression, Random Forest, XGBoost, a small LSTM in PyTorch, and a "
        "soft-vote ensemble whose two members are chosen on the validation set. All tuning used log loss inside the "
        "training period: purged time-series cross-validation for the tabular models, and a purged hold-out slice "
        "for the LSTM. The selected values are telling: Logistic "
        f"Regression chose C = {lr_c}, the strongest regularisation in its grid, the trees chose depth 2 or 3, and the "
        "LSTM chose a short window. When tuning keeps choosing the simplest model, it usually means there is little "
        "stable signal to learn, which already hints at the results. We also built four baselines."))

    # 13 --------------------------------------------------------------------------------- signals & backtest
    s = D.content("08 · IMPLEMENTATION", "From probability to position, net of costs")
    text(s.shapes, X0, 1.95, 5.65, 0.26, "DECISION RULE", size=11, color=AMBER, font=MONO, spc=2, name="rule_label")
    bx0, bx1 = 0.9, 6.15
    tau = F.tau_spy_xgb
    xt = bx0 + tau * (bx1 - bx0)
    pb = s.shapes.add_group_shape()
    pb.name = "probbar"
    c_ = rect(pb.shapes, bx0, 2.9, xt - bx0 - 0.02, 0.42, fill=SLATE, alpha=0.45, r=0.06)
    shape_text(c_, "CASH", size=11, color=TEXT, font=MONO, align="c", spc=2)
    l_ = rect(pb.shapes, xt + 0.02, 2.9, bx1 - xt - 0.02, 0.42, fill=AMBER, r=0.06)
    shape_text(l_, "LONG", size=11, color=INK, font=MONO, align="c", spc=2)
    for v_, lab in ((0.0, "0"), (0.5, "0.5"), (1.0, "1")):
        text(pb.shapes, bx0 + v_ * (bx1 - bx0) - 0.3, 3.42, 0.6, 0.24, lab, size=10, color=MUTED, font=MONO, align="c")
    text(pb.shapes, bx1 + 0.08, 2.9, 0.6, 0.42, "P(Up)", size=11, color=MUTED, font=MONO, anchor="m")
    tm = s.shapes.add_group_shape()
    tm.name = "tau_marker"
    tri = rect(tm.shapes, xt - 0.1, 2.66, 0.2, 0.17, fill=TEXT, kind=MSO_SHAPE.ISOSCELES_TRIANGLE)
    tri.rotation = 180
    text(tm.shapes, xt - 1.3, 2.33, 2.6, 0.27, f"τ = {tau:g}  (SPY · XGBoost)", size=11, color=TEXT, font=MONO,
         align="c")
    text(s.shapes, X0, 3.95, 5.65, 2.6, [
        {"text": "Long at the next open when P(Up) ≥ τ, otherwise cash (long-only).", "bullet": AMBER, "after": 8},
        {"text": f"τ comes from {F.grid_n} values ({F.grid_min:.2f}–{F.grid_max:.2f}) and is chosen by validation "
                 f"Sharpe, never on test.", "bullet": AMBER, "after": 8},
        {"text": f"Chosen τ ranged {F.thr_min:.2f}–{F.thr_max:.2f} across the 15 model–ticker pairs.",
         "bullet": AMBER, "after": 8},
        {"text": "Also backtested: an argmax rule (long when Up is the most likely class).", "bullet": AMBER},
    ], size=15, color=TEXT2, line=1.05, name="rule_text")
    text(s.shapes, 6.95, 1.95, 5.68, 0.26, "BACKTEST · EACH DAY t", size=11, color=AMBER, font=MONO, spc=2,
         name="bt_label")
    g = card(s.shapes, 6.95, 2.33, 5.68, 1.6, "eq_backtest", accent=True)
    equation(g.shapes, 6.95, 2.33, 5.68, 1.6, [
        [("cost", "v"), ("t", "vs"), ("  =  ", "o"), ("c", "v"), (" · |", "o"), ("p", "v"), ("t", "vs"), (" − ", "o"),
         ("p", "v"), ("t−1", "vs"), ("| · ", "o"), ("E", "v"), ("t−1", "vs")],
        [("E", "v"), ("t", "vs"), ("  =  (", "o"), ("E", "v"), ("t−1", "vs"), (" − ", "o"), ("cost", "v"),
         ("t", "vs"), (") · (1 + ", "o"), ("p", "v"), ("t", "vs"), (" ", "o"), ("r", "v"), ("t", "vs"), (")", "o")],
    ], size=21)
    text(s.shapes, 6.95, 4.03, 5.68, 0.62, f"p = position (0 or 1) · r = next-open return · E = equity · c = "
         f"{F.cost_total}\u00a0bps per unit turnover ({F.fee_bps}\u00a0fee + {F.slip_bps}\u00a0slippage)",
         size=12.5, color=TEXT2,
         line=1.05, name="bt_legend")
    text(s.shapes, 6.95, 4.88, 5.68, 0.24, "REPORTED FOR EVERY STRATEGY", size=10, color=MUTED, font=MONO, spc=2,
         name="metrics_label")
    mg = s.shapes.add_group_shape()
    mg.name = "metric_chips"
    xx, yy = 6.95, 5.2
    for lab in ["cumulative return", "annual return", "volatility", "Sharpe", "Sortino", "max drawdown", "Calmar",
                "win rate", "turnover", "trades", "costs paid"]:
        wch = 0.085 * len(lab) + 0.36
        if xx + wch > XR + 0.01:
            xx, yy = 6.95, yy + 0.46
        c_ = rect(mg.shapes, xx, yy, wch, 0.36, fill=SURF2, line=LINE, r=0.18)
        shape_text(c_, lab, size=12, color=TEXT2, align="c")
        xx += wch + 0.1
    mo.add("rule_label", "fade", "after", dur=0.3)
    mo.add("probbar", "wipe", "with", delay=0.1, dur=0.8, direction="left")
    mo.add("tau_marker", "rise", "after", dur=0.4)
    mo.add("rule_text", "fade", "after", dur=0.5)
    mo.add("bt_label", "fade", "after", dur=0.3)
    mo.add("eq_backtest", "rise", "with", delay=0.1, dur=0.5)
    mo.add("bt_legend", "fade", "after", dur=0.4)
    mo.add("metrics_label", "fade", "after", dur=0.3)
    mo.add("metric_chips", "rise", "with", delay=0.1, dur=0.45)
    D.note(s, (
        "Here is how a probability becomes a trade. If the model's probability of an up move is at least tau, we go "
        f"long at the next open; otherwise we hold cash. Tau is picked from {F.grid_n} values using validation Sharpe "
        "only; the test set never influences it. The backtester on the right charges costs on every change in "
        f"position, {F.cost_total} basis points per unit of turnover, before applying the day's return, and it "
        "compounds equity day by day. Every strategy, baselines included, goes through exactly this code, and we "
        "report the full set of metrics for each."))

    # 14 --------------------------------------------------------------------------------- key features
    s = D.content("09 · KEY FEATURES", "Engineered like research software")
    tiles = [
        (f"{F.n_tests}", "automated tests", "pytest, including look-ahead and truncation-invariance checks", False),
        ("1", "command for all results", "python -m scripts.run_experiment", True),
        (f"{F.minutes} min", "full experiment on a CPU", f"seeded and deterministic (seed {F.seed})", False),
        (f"{F.n_figures}", "figures, generated", "plus a Markdown experiment report; the IEEE report has its own "
                                                 "builder", False),
        ("6", "dashboard tabs", "Streamlit app with live re-training on fresh data", False),
        (f"{F.n_boot:,}", "bootstrap resamples per CI", f"circular block bootstrap, block length {F.block}", False),
    ]
    names = []
    for i, (big, lab, det, mono) in enumerate(tiles):
        r_, c_ = divmod(i, 3)
        x, y = X0 + c_ * (3.8 + 0.2665), 1.95 + r_ * 2.25
        g = card(s.shapes, x, y, 3.8, 1.95, f"tile_{i + 1}")
        text(g.shapes, x + 0.35, y + 0.2, 3.1, 0.85, big, size=46, color=AMBER, font=LIGHT)
        text(g.shapes, x + 0.35, y + 1.03, 3.1, 0.34, lab, size=15, color=TEXT, font=HEAD)
        text(g.shapes, x + 0.35, y + 1.39, 3.1, 0.48, det, size=11.5 if mono else 12.5, color=TEXT2,
             font=MONO if mono else BODY, line=1.0)
        names.append(f"tile_{i + 1}")
    text(s.shapes, X0, 6.4, CW, 0.3, "All counts are measured from the repository: tests collected by pytest, figures "
         "in results/figures, runtime from results/metrics/run_summary.json.", size=12, color=MUTED, name="caption")
    mo.stagger(names, "rise", gap=0.1)
    mo.add("caption", "fade", "after", dur=0.4)
    D.note(s, (
        f"We built this like research software rather than a one-off notebook. There are {F.n_tests} automated tests, "
        "including tests that check that features never use future data. One command reproduces every result, "
        "figure and the experiment report, and the IEEE report has its own build script. "
        f"The full run takes about {F.minutes} minutes on a CPU and is seeded, so it is repeatable. "
        f"It generates {F.n_figures} figures and the reports, there is a six-tab dashboard, and every confidence "
        f"interval uses {F.n_boot:,} block-bootstrap resamples."))

    # 15 --------------------------------------------------------------------------------- dashboard
    s = D.content("10 · LIVE SYSTEM", "The working system: an interactive dashboard")
    hero_box = (X0, 2.3, 7.6, 4.275)
    cx, cw_ = 8.55, XR - 8.55
    # crops are (left, top, right, bottom) fractions of the 1600 x 900 screenshots
    wf_crop, live_crop = (0.23125, 0.4278, 0.04375, 0.1111), (0.23125, 0.0778, 0.04375, 0.6533)
    live_full = (0.0, 0.0722, 0.0, 0.0156)  # drops a stray tab underline at the top and a sliver at the bottom

    def shot(slide, path, box, crop, name, alt):
        pic = picture(slide.shapes, path, *box, name=name, alt=alt)
        pic.crop_left, pic.crop_top, pic.crop_right, pic.crop_bottom = crop
        return pic

    def tab(slide, x, y, w, label, hint, name):
        text(slide.shapes, x, y, w, 0.26, [[(label, {"color": TEXT}), (f"   {hint}", {"color": MUTED})]], size=10.5,
             font=MONO, spc=1, name=name)

    def box_for(crop, x, y, w):
        frac_w, frac_h = 1 - crop[0] - crop[2], 1 - crop[1] - crop[3]
        return (x, y, w, w * (900 * frac_h) / (1600 * frac_w))

    wf_box = box_for(wf_crop, cx, 2.3, cw_)
    live_box = box_for(live_crop, cx, 4.9, cw_)
    tab(s, X0, 1.95, 7.6, "INTERACTIVE BACKTEST", "model · τ · costs · sizing, recomputed live", "tab_backtest")
    shot(s, SHOTS / "02_backtest.png", hero_box, (0, 0, 0, 0), "!!shot_backtest",
         "Screenshot of the dashboard's Interactive backtest tab for SPY with XGBoost")
    tab(s, cx, 1.95, cw_, "WALK-FORWARD", "2016–2025", "tab_wf")
    shot(s, SHOTS / "04_walkforward.png", wf_box, wf_crop, "!!shot_wf",
         "Crop of the dashboard's Walk-forward tab: SPY out-of-sample portfolio value 2016-2025")
    text(s.shapes, cx, wf_box[1] + wf_box[3] + 0.09, cw_, 0.5, "Dashed: buy-and-hold, finishing above all five ML "
         "strategies (SPY, walk-forward).", size=12.5, color=TEXT2, line=1.05, name="wf_caption")
    tab(s, cx, 4.55, cw_, "TRAIN A MODEL LIVE", "fresh data", "!!tab_live")
    shot(s, SHOTS / "05_train_live.png", live_box, live_crop, "!!shot_live",
         "Crop of the dashboard's Train a model live tab: the metric cards after a live XGBoost run")
    text(s.shapes, cx, live_box[1] + live_box[3] + 0.09, cw_, 0.5, f"Fresh download → retrain → backtest in "
         f"{LIVE['seconds']} s; opened on the next slide.", size=12.5, color=TEXT2, line=1.05, name="live_caption")
    text(s.shapes, 5.6, 0.5, XR - 5.6, 0.26, "REAL SCREENSHOTS · LOCALHOST:8501 · CAPTURED 24 OCT 2026", size=10,
         color=MUTED, font=MONO, spc=1, align="r", name="shots_note")
    mo.add("tab_backtest", "fade", "after", dur=0.3)
    mo.add("!!shot_backtest", "rise", "with", delay=0.1, dur=0.6)
    mo.add("tab_wf", "fade", "with", delay=0.3, dur=0.3)
    mo.add("!!shot_wf", "rise", "with", delay=0.35, dur=0.55)
    mo.add("wf_caption", "fade", "with", delay=0.55, dur=0.4)
    mo.add("!!tab_live", "fade", "with", delay=0.6, dur=0.3)
    mo.add("!!shot_live", "rise", "with", delay=0.65, dur=0.55)
    mo.add("live_caption", "fade", "with", delay=0.85, dur=0.4)
    mo.add("shots_note", "fade", "after", dur=0.4)
    D.note(s, (
        "This is the working system: a Streamlit dashboard, and these are real screenshots of it running. The large "
        "view is the interactive backtest. You pick a model, move the threshold, change costs or turn on volatility "
        "sizing, and our own backtester recomputes every number from the saved out-of-sample probabilities. On the "
        "right are the walk-forward tab, where buy-and-hold finishes above all five ML strategies, and the "
        "live-training tab, which the next slide opens. There are six tabs in "
        "total, and if time permits I can switch to the running app now."))

    # 16 --------------------------------------------------------------------------------- live run (morph)
    s = D.content("10 · LIVE SYSTEM", "Retrain live, with ten more months of data", transition="morph",
                  duration=1.3)
    tab(s, X0, 1.95, 7.6, "TRAIN A MODEL LIVE", "SPY · XGBoost · the study's tuned settings", "!!tab_live")
    shot(s, SHOTS / "05_train_live.png", box_for(live_full, X0, 2.3, 7.6), live_full, "!!shot_live",
         "Screenshot of the dashboard's Train a model live tab after a live XGBoost run on SPY")
    new_days = LIVE["test_days"] - F.split["test"]["rows"]
    facts = [
        ("live_1", f"{LIVE['seconds']} s", "fresh download → features → split → XGBoost with the study's settings → "
                                           "τ → backtest (no re-tuning)"),
        ("live_2", f"{LIVE['test_days']:,}", f"test days to {LIVE['test_end_label']}, {new_days} more than the "
                                             "study's test set"),
        ("live_3", f"{LIVE['sharpe']:.2f} vs {LIVE['bh_sharpe']:.2f}", "Sharpe over the whole live window: XGBoost "
                                                                       "vs buy-and-hold"),
    ]
    text(s.shapes, cx, 1.95, cw_, 0.28, "LIVE RUN · 24 OCT 2026", size=11, color=AMBER, font=MONO, spc=2,
         name="live_label")
    for i, (nm, big, lab) in enumerate(facts):
        y = 2.38 + i * 1.24
        g = s.shapes.add_group_shape()
        g.name = nm
        text(g.shapes, cx, y, cw_, 0.6, big, size=34, color=TEXT, font=LIGHT)
        text(g.shapes, cx, y + 0.6, cw_, 0.55, lab, size=12.5, color=TEXT2, line=1.0)
    text(s.shapes, cx, 6.15, cw_, 0.36, "Consistent with the results that follow.", size=15, color=AMBER, font=HEAD,
         name="live_verdict")
    text(s.shapes, 5.6, 0.5, XR - 5.6, 0.26, "REAL SCREENSHOT · ONE RUN, NOT PART OF THE REPORTED RESULTS", size=10,
         color=MUTED, font=MONO, spc=1, align="r", name="shots_note")
    mo.add("live_label", "fade", "after", delay=0.1, dur=0.4)
    mo.stagger(["live_1", "live_2", "live_3"], "rise", start="with", gap=0.15, first_delay=0.15, dur=0.5)
    mo.add("live_verdict", "fade", "after", dur=0.5)
    mo.add("shots_note", "fade", "with", delay=0.2, dur=0.4)
    D.note(s, (
        "This tab runs the core pipeline live for one model. It downloads fresh data, builds the features, splits "
        "the data, trains XGBoost with the hyperparameters we tuned in the study, picks tau on validation and "
        f"backtests, all in about {LIVE['seconds']:.0f} seconds because nothing is re-tuned. Because the data are "
        f"current, the test window now runs to {LIVE['test_end_label']}, which adds {new_days} test days beyond "
        f"our study's test set. Over the whole window XGBoost reached a Sharpe ratio of {LIVE['sharpe']:.2f} "
        f"against {LIVE['bh_sharpe']:.2f} for buy-and-hold, and its accuracy was {LIVE['acc_gap_pp']} points below "
        "the majority class, consistent with the results we show next. This is one live run, kept separate from "
        "the reported results."))

    # 17 --------------------------------------------------------------------------------- classification
    s = D.content("11 · RESULTS", "Accuracy: none reliably beat 'always predict Up'")
    text(s.shapes, X0, 1.95, 7.75, 0.3, "Test accuracy minus 'always predict Up', percentage points (95% bootstrap CI)",
         size=13, color=TEXT2, name="chart_head")
    cd = CategoryChartData(number_format="0.0")
    cd.categories = [SHORT[m] for m in ML]
    da = F.dacc
    for tk in TICKERS:
        sub_ = da[da.ticker == tk].set_index("strategy").loc[ML]
        cd.add_series(tk, (sub_.estimate * 100).round(2).tolist())
    gf = s.shapes.add_chart(XL_CHART_TYPE.COLUMN_CLUSTERED, emu(X0), emu(2.35), emu(7.75), emu(4.1), cd)
    gf.name = "chart_acc"
    ch = gf.chart
    _chart_frame(ch, legend=True)
    _axes(ch, vmin=-5, vmax=4, major=1, fmt='+0;-0;0')
    plot = ch.plots[0]
    plot.gap_width, plot.overlap = 70, -8
    for ser, tk in zip(plot.series, TICKERS):
        _series_fill(ser, TICKER_COLORS[tk])
        sub_ = da[da.ticker == tk].set_index("strategy").loc[ML]
        _err_bars(ser, ((sub_.high - sub_.estimate) * 100).tolist(), ((sub_.estimate - sub_.low) * 100).tolist())
    rel = F.acc_reliable.iloc[0]
    stats = [
        ("acc_1", f"{F.acc_min * 100:.1f}–{pct(F.acc_max)}", TEXT, "test accuracy, 5 models × 3 tickers"),
        ("acc_2", f"{F.maj_min * 100:.1f}–{pct(F.maj_max)}", MUTED, "accuracy of 'always predict Up', the "
                                                                   "majority class"),
        ("acc_3", f"{len(F.acc_reliable)} / 15", AMBER, f"differs reliably from that baseline: {rel.ticker} "
                                                         f"{rel.strategy}, {abs(rel.estimate) * 100:.1f} pp worse"),
    ]
    for i, (nm, big, col, lab) in enumerate(stats):
        y = 1.95 + i * 1.5
        g = s.shapes.add_group_shape()
        g.name = nm
        text(g.shapes, 8.85, y, XR - 8.85, 0.75, big, size=38, color=col, font=LIGHT)
        text(g.shapes, 8.85, y + 0.75, XR - 8.85, 0.6, lab, size=13, color=TEXT2, line=1.05)
    mo.add("chart_head", "fade", "after", dur=0.4)
    mo.add("chart_acc", "wipe", "with", delay=0.1, dur=0.9)
    mo.stagger(["acc_1", "acc_2", "acc_3"], "rise", gap=0.2, dur=0.5)
    D.note(s, (
        "Now the results, starting with prediction. Each bar is a model's test accuracy minus the accuracy of always "
        "predicting up, which is the most common class, with 95 percent bootstrap intervals. The models scored between "
        f"{F.acc_min * 100:.1f} and {pct(F.acc_max)}, while the trivial majority-class predictor scored "
        f"{F.maj_min * 100:.1f} to {pct(F.maj_max)}. Almost every interval crosses zero. The only model that is "
        f"reliably different is {rel.ticker} {rel.strategy}, and it is worse. At a daily horizon, none of our models "
        "predicted direction reliably better than the base rate."))

    # 18 --------------------------------------------------------------------------------- trading forest
    s = D.content("11 · RESULTS", "Trading: none of the 15 reliably beat buy-and-hold")
    text(s.shapes, X0, 1.95, 7.75, 0.3, "τ-rule strategies: Sharpe minus buy-and-hold, test 2022–25 (95% bootstrap CI)",
         size=13, color=TEXT2, name="chart_head")
    dsr = F.dsharpe.set_index(["ticker", "model"])
    est_, lo_, hi_, labels_ = [], [], [], []
    cd = CategoryChartData(number_format="0.00")
    for tk in TICKERS:
        cat = cd.add_category(tk)
        for m in ML:
            cat.add_sub_category(SHORT[m])
            r = dsr.loc[(tk, m)]
            est_.append(r.estimate)
            lo_.append(r.low)
            hi_.append(r.high)
            labels_.append((tk, m))
    cd.add_series("Sharpe − buy-and-hold", [round(v, 4) for v in est_])
    gf = s.shapes.add_chart(XL_CHART_TYPE.BAR_CLUSTERED, emu(X0), emu(2.3), emu(7.75), emu(3.95), cd)
    gf.name = "chart_sharpe"
    ch = gf.chart
    _chart_frame(ch, legend=False)
    _axes(ch, vmin=-0.45, vmax=0.35, major=0.15, fmt='+0.00;-0.00;0', cat_size=11)
    ch.category_axis.reverse_order = True
    ch.value_axis._element.find(qn("c:crosses")).set("val", "max")  # value axis at the bottom of the reversed axis
    plot = ch.plots[0]
    plot.gap_width = 45
    ser = plot.series[0]
    _series_fill(ser, AMBER)
    rel_set = set(zip(F.sharpe_reliable.ticker, F.sharpe_reliable.model))
    for i, key in enumerate(labels_):
        if key in rel_set:
            pt = ser.points[i]
            pt.format.fill.solid()
            pt.format.fill.fore_color.rgb = rgb(CORAL)
            dpt = next(d for d in ser._element.findall(qn("c:dPt")) if d.find(qn("c:idx")).get("val") == str(i))
            if dpt.find(qn("c:invertIfNegative")) is None:  # a point does not inherit the series' setting
                dpt.find(qn("c:idx")).addnext(etree.Element(qn("c:invertIfNegative"), val="0"))
        if abs(est_[i]) < 1e-9 and abs(hi_[i] - lo_[i]) < 1e-9:
            dl = ser.points[i].data_label
            dl.has_text_frame = True
            dl.text_frame.text = "= buy-and-hold"
            dl.position = XL_LABEL_POSITION.OUTSIDE_END
            for run in dl.text_frame.paragraphs[0].runs:
                run.font.size, run.font.color.rgb, run.font.name = Pt(10), rgb(MUTED), BODY
    _err_bars(ser, [h - e for h, e in zip(hi_, est_)], [e - lo for e, lo in zip(est_, lo_)])
    am_txt = ", ".join(f"{tk} {SHORT[st.replace(' (argmax)', '')]} {sh_:.2f} vs {b_:.2f}"
                       for tk, st, sh_, b_ in F.argmax_higher)
    text(s.shapes, X0, 6.36, CW, 0.5, [
        {"runs": [("■ ", {"color": CORAL}), ("coral: 95% interval excludes zero (reliably worse)", {})], "after": 2},
        {"runs": [("Argmax rule (slide 13): ", {"color": TEXT2}),
                  (f"{len(F.argmax_higher)} of {F.n_argmax} above buy-and-hold ({am_txt}); no intervals computed.",
                   {})]}], size=11.5, color=MUTED, name="chart_key")
    stats = [
        ("tr_1", f"{F.n_higher} / {F.n_ml}", AMBER, "above buy-and-hold on Sharpe; all of their 95% intervals "
                                                    "include zero"),
        ("tr_2", f"{F.n_identical} / {F.n_ml}", TEXT, "identical to buy-and-hold: the validation-chosen τ kept them "
                                                      "invested on every test day"),
        ("tr_3", f"{signed(F.pers_spy_cum * 100, 0)}%", CORAL, f"naive persistence on SPY, after "
                                                               f"{F.pers_spy_trades} trades and costs"),
    ]
    for i, (nm, big, col, lab) in enumerate(stats):
        y = 1.95 + i * 1.5
        g = s.shapes.add_group_shape()
        g.name = nm
        text(g.shapes, 8.85, y, XR - 8.85, 0.75, big, size=38, color=col, font=LIGHT)
        text(g.shapes, 8.85, y + 0.75, XR - 8.85, 0.72, lab, size=13, color=TEXT2, line=1.05)
    mo.add("chart_head", "fade", "after", dur=0.4)
    mo.add("chart_sharpe", "fade", "with", delay=0.1, dur=0.7)
    mo.add("chart_key", "fade", "with", delay=0.5, dur=0.5)
    mo.stagger(["tr_1", "tr_2", "tr_3"], "rise", gap=0.2, dur=0.5)
    worst = F.sharpe_reliable.iloc[0]
    D.note(s, (
        "Trading results. Each bar is how much a strategy's Sharpe ratio beat or trailed buy-and-hold over the test "
        f"period, with bootstrap intervals. {F.n_higher} of the {F.n_ml} ML strategies had a slightly higher Sharpe, "
        f"but every one of those intervals includes zero. {F.n_identical} strategies were literally identical to "
        "buy-and-hold, because the threshold chosen on the bullish 2020-21 validation period kept them invested every "
        f"day. The only reliable difference is {worst.ticker} {worst.model}, in red, which was worse. For contrast, the "
        f"naive persistence baseline lost {abs(F.pers_spy_cum) * 100:.0f} percent on SPY once costs were included. "
        f"One caveat, shown under the chart: with the simpler argmax rule, {len(F.argmax_higher)} of {F.n_argmax} "
        "strategies were above buy-and-hold, both Logistic Regression, but we did not compute intervals for that "
        "rule, so we make no claim either way."))

    # 19 --------------------------------------------------------------------------------- walk-forward & robustness
    s = D.content("11 · RESULTS", "Walk-forward and robustness checks agree")
    text(s.shapes, X0, 1.95, 6.45, 0.3, f"Walk-forward {F.wf_years[0]}–{F.wf_years[-1]}: Sharpe ratio, retrained "
         "every year, out-of-sample", size=13, color=TEXT2, name="chart_head")
    cd = CategoryChartData(number_format="0.00")
    cd.categories = TICKERS
    cd.add_series("Buy-and-hold", [float(F.wf_table[t_][0]) for t_ in TICKERS])
    cd.add_series("MA crossover (10/50)", [float(F.wf_table[t_][1]) for t_ in TICKERS])
    cd.add_series("Best ML model, picked in hindsight", [float(F.wf_table[t_][2]) for t_ in TICKERS])
    gf = s.shapes.add_chart(XL_CHART_TYPE.COLUMN_CLUSTERED, emu(X0), emu(2.3), emu(6.45), emu(4.3), cd)
    gf.name = "chart_wf"
    ch = gf.chart
    _chart_frame(ch, legend=True)
    _axes(ch, vmin=0, vmax=1.2, major=0.2, fmt="0.0", cat_size=13)
    ch.value_axis.has_major_gridlines = False  # the data labels carry the values; gridlines crossed them
    plot = ch.plots[0]
    plot.gap_width, plot.overlap = 60, -6
    plot.has_data_labels = True
    dls = plot.data_labels
    dls.number_format, dls.number_format_is_linked = "0.00", False
    dls.position = XL_LABEL_POSITION.OUTSIDE_END
    dls.font.size, dls.font.color.rgb = Pt(11), rgb(TEXT)
    for ser_, col in zip(plot.series, [TEXT, SLATE, AMBER]):
        _series_fill(ser_, col)
    text(s.shapes, 7.5, 1.95, XR - 7.5, 0.95, f"{F.wf_beat} / {F.wf_n}", size=54, color=AMBER, font=LIGHT,
         name="wf_big")
    text(s.shapes, 7.5, 2.92, XR - 7.5, 0.6, "ML strategies beat buy-and-hold over the ten walk-forward years",
         size=14, color=TEXT2, name="wf_big_label")
    lo_r, hi_r = F.vol_red
    rob = [
        ("rob_cost", "COSTS", f"{F.cost_beat_lo[0]} / 15 beat buy-and-hold at {F.cost_bps[0]} bps, "
                              f"{F.cost_beat_hi[0]} / 15 at {F.cost_bps[-1]} bps"),
        ("rob_vol", "SIZING", f"{F.vol_label} volatility targeting cut max drawdown by {lo_r * 100:.0f}–"
                              f"{hi_r * 100:.0f}% at a similar Sharpe, by holding less"),
        ("rob_var", "ASSUMPTIONS", f"Binary target or close fills (no LSTM, reduced grids): {F.var_lo}–{F.var_hi} of "
                                   f"{F.var_n} beat buy-and-hold, and the winners change"),
    ]
    text(s.shapes, 7.5, 3.56, XR - 7.5, 0.24, "ROBUSTNESS CHECKS · HOLD-OUT TEST 2022–2025", size=10, color=MUTED,
         font=MONO, spc=2, name="rob_head")
    for i, (nm, lab, txt) in enumerate(rob):
        y = 3.88 + i * 0.92
        g = card(s.shapes, 7.5, y, XR - 7.5, 0.8, nm)
        text(g.shapes, 7.78, y, XR - 8.06, 0.8, [[(f"{lab}   ", {"font": MONO, "size": 11, "color": AMBER}),
                                                  (txt, {})]], size=13, color=TEXT, anchor="m", line=1.05)
    mo.add("chart_head", "fade", "after", dur=0.4)
    mo.add("chart_wf", "wipe", "with", delay=0.1, dur=0.9)
    mo.add("wf_big", "zoom", "after", dur=0.45)
    mo.add("wf_big_label", "fade", "with", delay=0.15, dur=0.4)
    mo.add("rob_head", "fade", "after", dur=0.3)
    mo.stagger(["rob_cost", "rob_vol", "rob_var"], "rise", start="with", gap=0.15, first_delay=0.1, dur=0.45)
    D.note(s, (
        "The walk-forward test tells the same story over ten years. Buy-and-hold is the white bar, the moving-average "
        "crossover is grey, and amber is the best ML model for each ticker, picked with hindsight. Even with that "
        f"cherry-picking, ML never beat buy-and-hold: {F.wf_beat} of {F.wf_n}. The robustness checks on the right use the "
        f"2022 to 2025 hold-out test, and they agree. With zero costs, {F.cost_beat_lo[0]} strategies edge ahead, but at "
        f"{F.cost_bps[-1]} basis points none do. Volatility targeting cut drawdowns by {lo_r * 100:.0f} to "
        f"{hi_r * 100:.0f} percent at a similar Sharpe ratio, simply by holding less when markets are volatile; that "
        "is risk control, not prediction. And changing the target or the fill price changes which strategies 'win', "
        "a classic sign of noise."))

    # 20 --------------------------------------------------------------------------------- challenges
    s = D.content("12 · CHALLENGES", "Challenges we handled, limitations we report")
    cols = [
        ("CHALLENGES WE HANDLED", AMBER, "challenges", [
            ("Leakage at every step", "Purged splits, point-in-time features and tests that fail on any look-ahead."),
            ("Thresholds collapsing to buy-and-hold", "A bullish 2020–21 validation often picks 'always invested'; "
                                                      "we report it rather than hide it."),
            ("An unstable LSTM on ~2,200 training days", "A small one-layer network, early stopping on a purged slice and "
                                                   "fixed seeds."),
            ("Unfinished live bars", "Live data can include today's partial session; a guard drops it before "
                                     "16:15 New York time."),
            ("A 24-minute experiment", "Reduced grids for walk-forward, and a separate results folder for every "
                                       "flag combination."),
        ]),
        ("LIMITATIONS WE REPORT", TEXT2, "limits", [
            ("Weak daily signal", "Liquid large caps have near-zero daily autocorrelation; ~50% accuracy is "
                                  "expected."),
            ("Survivorship bias", "SPY, AAPL and MSFT were chosen in hindsight; weaker or delisted stocks could "
                                  "differ."),
            ("Multiple testing", "Many models × thresholds × assets, and no deflated-Sharpe correction."),
            ("Simple cost model", f"A flat {F.cost_total} bps; no market impact, spreads, borrow costs or taxes."),
            ("Price and volume only", "No news, macro, fundamental or order-book data."),
        ]),
    ]
    assert F.minutes == 24  # the "24-minute" heading above
    for ci, (head, col, nm, items) in enumerate(cols):
        x = X0 + ci * (5.84 + 0.253)
        g = card(s.shapes, x, 1.95, 5.84, 4.62, nm, fill_=SURF if ci == 0 else SURF2)
        text(g.shapes, x + 0.35, 2.2, 5.1, 0.28, head, size=12, color=col, font=MONO, spc=2)
        for j, (t_, d_) in enumerate(items):
            y = 2.65 + j * 0.77
            text(g.shapes, x + 0.35, y, 0.5, 0.3, f"0{j + 1}", size=11, color=col, font=MONO)
            text(g.shapes, x + 0.9, y - 0.02, 4.6, 0.74, [{"text": t_, "after": 1,
                                                          "runs": [(t_, {"font": HEAD, "size": 14.5, "color": TEXT})]},
                                                         {"text": d_}], size=12.5, color=TEXT2, line=1.0)
    mo.stagger(["challenges", "limits"], "rise", gap=0.3, dur=0.55)
    D.note(s, (
        "On the left, the main engineering challenges we handled. Leakage can creep in at every step, so we built tests that "
        "fail on any look-ahead. Thresholds chosen on a bull-market validation period often collapse to "
        "buy-and-hold, and we report that honestly. The LSTM is unstable with only about 2,200 training days, so we "
        "kept it small. Live data can include today's unfinished bar, which we now drop. On the right are the "
        "limitations we state openly: a weak daily signal, survivorship bias from choosing successful stocks, many "
        "comparisons without a multiple-testing correction, a simple cost model, and price data only."))

    # 21 --------------------------------------------------------------------------------- future scope
    s = D.content("13 · FUTURE SCOPE", "Where this goes next")
    text(s.shapes, X0, 1.95, CW, 0.26, "THE STRETCH GOAL, DESIGNED LEAKAGE-SAFE · NEWS SENTIMENT VIA RETRIEVAL (RAG)",
         size=11,
         color=AMBER, font=MONO, spc=2, name="rag_label")
    steps = [("Time-stamped news", "GDELT, SEC 8-K filings, licensed archives"),
             ("Retrieve before close t", "vector index by ticker and publication time"),
             ("Score sentiment", "a financial model such as FinBERT"),
             ("Daily features", "mean, count, spread; lagged one bar"),
             ("Same protocol", "same splits, τ rule and backtester")]
    names = []
    for i, (nm, det) in enumerate(steps):
        x = X0 + i * (2.17 + 0.2708)
        g = card(s.shapes, x, 2.35, 2.17, 1.7, f"rag_{i + 1}", accent=(i == 1))
        text(g.shapes, x + 0.25, 2.55, 1.7, 0.26, f"0{i + 1}", size=11, color=AMBER, font=MONO)
        text(g.shapes, x + 0.25, 2.85, 1.75, 0.55, nm, size=15, color=TEXT, font=HEAD, line=0.95)
        text(g.shapes, x + 0.25, 3.4, 1.75, 0.58, det, size=12, color=TEXT2, line=1.0)
        names.append(f"rag_{i + 1}")
        if i < 4:
            line(s.shapes, x + 2.17 + 0.03, 3.2, x + 2.17 + 0.2408, 3.2, color=AMBER, w=1.25, head=True,
                 name=f"rag_arr_{i + 1}")
            names.append(f"rag_arr_{i + 1}")
    nexts = [("DATA", "A broader, fairer universe", "Include delisted stocks to remove survivorship bias, and test "
                                                   "weekly and intraday horizons."),
             ("STATISTICS", "Stronger evidence", "Deflated Sharpe ratio and multiple-testing control; calibrate "
                                                 "probabilities before thresholding."),
             ("PRACTICE", "From backtest to live", "Host the dashboard publicly and paper-trade the signals on live "
                                                   "data with risk-based sizing.")]
    names2 = []
    for i, (lab, ttl_, body) in enumerate(nexts):
        x = X0 + i * (3.8 + 0.2665)
        g = card(s.shapes, x, 4.45, 3.8, 1.95, f"next_{i + 1}")
        text(g.shapes, x + 0.3, 4.68, 3.2, 0.26, lab, size=11, color=AMBER, font=MONO, spc=2)
        text(g.shapes, x + 0.3, 5.0, 3.2, 0.36, ttl_, size=16, color=TEXT, font=HEAD)
        text(g.shapes, x + 0.3, 5.42, 3.2, 0.9, body, size=13, color=TEXT2, line=1.05)
        names2.append(f"next_{i + 1}")
    mo.add("rag_label", "fade", "after", dur=0.3)
    mo.stagger(names, "rise", start="with", gap=0.1, first_delay=0.1, dur=0.4)
    mo.stagger(names2, "rise", gap=0.12, dur=0.5)
    D.note(s, (
        "Future work starts with the stretch goal. A correct news-sentiment layer needs time-stamped articles, "
        "retrieval that only sees documents published before the decision time, a financial sentiment model such as "
        "FinBERT, daily features lagged by one bar, and then exactly the same evaluation protocol. Beyond that: a "
        "broader universe including delisted stocks, stronger statistics such as the deflated Sharpe ratio, and "
        "moving from backtest to paper trading on live data."))

    # 22 --------------------------------------------------------------------------------- conclusion
    s = D.content("14 · CONCLUSION", "What we can claim")
    claims = [
        ("ML did not reliably beat simple baselines at a daily horizon.",
         f"No model reliably beat 'always Up' on accuracy (one was reliably worse); τ-rule: {F.wf_beat} of "
         f"{F.wf_n} beat buy-and-hold in walk-forward."),
        ("The contribution is the framework, not a trading edge.",
         f"Leakage-safe, cost-aware, covered by {F.n_tests} tests and reproducible with one command."),
        ("Position sizing changed risk more than any forecast changed returns.",
         f"Volatility targeting cut max drawdown by {lo_r * 100:.0f}–{hi_r * 100:.0f}% at a similar Sharpe ratio, "
         "by holding less, with no forecast at all."),
    ]
    names = []
    for i, (claim, support) in enumerate(claims):
        y = 1.95 + i * 1.25
        g = s.shapes.add_group_shape()
        g.name = f"claim_{i + 1}"
        text(g.shapes, X0, y - 0.05, 0.8, 0.7, str(i + 1), size=30, color=AMBER, font=LIGHT)
        text(g.shapes, 1.6, y, 11.0, 0.45, claim, size=21, color=TEXT, font=HEAD)
        text(g.shapes, 1.6, y + 0.5, 11.0, 0.62, support, size=14, color=TEXT2, line=1.05)
        names.append(f"claim_{i + 1}")
    draw_trace(s, X0, XR, 5.98, 6.72, alpha=0.55, w=1.75)
    mo.stagger(names, "rise", gap=0.3, dur=0.55)
    D.note(s, (
        "To conclude, three claims we can defend. First, at a daily horizon, machine learning did not reliably beat "
        "simple baselines, either on accuracy or on risk-adjusted returns, which is consistent with how efficient "
        "liquid large caps are. Second, our real contribution is the framework: leakage-safe, cost-aware, tested and "
        "reproducible, so a negative result here is trustworthy. Third, position sizing mattered more than "
        "prediction: volatility targeting cut drawdowns by a fifth to almost a half at a similar Sharpe ratio, simply "
        "by holding less when markets are volatile."))

    # 23 --------------------------------------------------------------------------------- thank you (morph)
    s = D.hero(transition="morph", duration=1.5)
    ttl = s.shapes.title
    ttl.name = "title"
    ttl.text_frame.clear()
    ttl.text_frame.margin_left = ttl.text_frame.margin_top = 0
    ttl.text_frame.vertical_anchor = MSO_ANCHOR.TOP
    Deck._place(ttl, X0, 1.45, 10.0, 1.2)
    fill(ttl.text_frame, "Thank you", size=66, color=TEXT, font=HEAD)
    sub = s.placeholders[1]
    sub.name = "subtitle"
    Deck._place(sub, X0, 2.72, 9.5, 0.6)
    sub.text_frame.clear()
    sub.text_frame.margin_left = sub.text_frame.margin_top = 0
    fill(sub.text_frame, "Questions and discussion", size=24, color=TEXT2, font=LIGHT)
    text(s.shapes, X0, 3.75, 8.0, 1.0, [
        {"runs": [("CODE   ", {"font": MONO, "size": 11, "color": AMBER}),
                  ("github.com/KennethMartin06/ml-based-algorithmic-trading-signals", {"font": MONO, "size": 13,
                                                                                      "color": TEXT})], "after": 5},
        {"runs": [("DEMO   ", {"font": MONO, "size": 11, "color": AMBER}),
                  ("streamlit run app/streamlit_app.py", {"font": MONO, "size": 13, "color": TEXT})], "after": 5},
        {"runs": [("TEAM   ", {"font": MONO, "size": 11, "color": AMBER}),
                  ("Ishan Singh · Kenneth Martin · MIT Manipal", {"size": 13, "color": TEXT2})]},
    ], name="info")
    draw_trace(s, X0, XR, 4.75, 6.85)
    mo.add("info", "rise", "after", delay=0.2, dur=0.6)
    D.note(s, (
        "Thank you. The code, results and reports are in our repository, and the dashboard runs with a single "
        "Streamlit command. We are happy to take questions, and we can open the live dashboard to explore any "
        "result in more detail."))

    assert len(D.prs.slides) == len(D.motion.slides) == len(D.notes)
    leftover = re.compile(r"\{[A-Za-z_][^{}]*\}")  # an f-string prefix forgotten somewhere
    for i, slide in enumerate(D.prs.slides, 1):
        texts = [sh.text_frame.text for sh in _all_shapes(slide.shapes) if sh.has_text_frame] + [D.notes[i - 1]]
        assert not any(leftover.search(t) for t in texts), f"slide {i}: unformatted placeholder"
    return D


def _all_shapes(shapes):
    for sh in shapes:
        yield sh
        if sh.shape_type == 6:  # group
            yield from _all_shapes(sh.shapes)


def outline(D: Deck, path: Path) -> None:
    """A plain-text companion: slide titles and speaker notes for rehearsal."""
    lines = ["# Slide outline and speaker notes", "",
             "Generated by `presentation/scripts/build_deck.py`; edit the script, not this file.", ""]
    for i, (slide, note, mo) in enumerate(zip(D.prs.slides, D.notes, D.motion.slides), 1):
        title = slide.shapes.title.text_frame.text.replace("\n", " ").replace("\v", " ")
        clicks = sum(1 for st in mo["steps"] if st["trigger"] == "click")
        lines += [f"## {i}. {title}", "",
                  f"*Transition: {mo['transition']}* · *clicks to advance builds: {clicks}*", "", note, ""]
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    F = Facts()
    D = build(F)
    out = BUILD / "deck_static.pptx"
    D.save(out)
    (BUILD / "motion_plan.json").write_text(json.dumps({"slides": D.motion.slides}, indent=1), encoding="utf-8")
    (PRES / "source").mkdir(exist_ok=True)
    outline(D, PRES / "source" / "slide_outline.md")
    print(f"wrote {out} ({len(D.prs.slides)} slides) and the motion plan; tests counted: {F.n_tests}")


if __name__ == "__main__":
    main()
