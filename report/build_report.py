"""Build the IEEE-format mini-project report (.docx) from the saved experiment results.

    python report/report_figures.py                 # column-width figures -> report/figures/
    python report/build_report.py [--pdf]           # report -> report/output/

The document is created from the official IEEE conference template (A4), so it uses the template's
own styles: paper title, Author, Abstract, Keywords, auto-numbered Heading 1/2 (I., A.), "Fig. n."
figure captions, "TABLE n." table heads and auto-numbered [n] references. Every number in the text and
tables is computed here from ``results/`` - nothing is typed in by hand.
"""

from __future__ import annotations

import argparse
import copy
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from docx import Document
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_BREAK
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Inches, Pt

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))

from src.utils import load_config, load_json  # noqa: E402

MET = ROOT / "results" / "metrics"
FIGS = HERE / "figures"
TEMPLATE = HERE / "ieee_conference_template_a4.docx"
OUT_DIR = HERE / "output"
OUT_NAME = "ML_Trading_Signals_IEEE_Report.docx"

AUTHORS = [
    ("Ishan Singh", "240962292"),
    ("Kenneth Martin", "240962047"),
]
TITLE = "Machine Learning Based Algorithmic Trading Signals: A Comparative Study Against Technical Baseline Strategies"

# figure / table order (numbers must match the auto-numbering of the template's caption styles)
FIG_ORDER = ["pipeline", "prices", "accuracy", "confusion", "equity", "sharpe", "walkforward", "importance"]
TAB_ORDER = ["features", "hyper", "classification", "trading", "walkforward", "robust"]
ROMAN = ["I", "II", "III", "IV", "V", "VI", "VII", "VIII", "IX", "X"]
FIG = {k: i + 1 for i, k in enumerate(FIG_ORDER)}
TAB = {k: ROMAN[i] for i, k in enumerate(TAB_ORDER)}

ML_KEYS = ["logistic_regression", "random_forest", "xgboost", "lstm", "ensemble"]
NAME = {"logistic_regression": "Logistic Regression", "random_forest": "Random Forest", "xgboost": "XGBoost", "lstm": "LSTM",
        "ensemble": "Ensemble", "majority_class": "Majority class", "persistence": "Naive persistence"}
SHORT = {"logistic_regression": "LR", "random_forest": "RF", "xgboost": "XGB", "lstm": "LSTM", "ensemble": "Ens."}

# ---------------------------------------------------------------------------------------------
# number formatting
# ---------------------------------------------------------------------------------------------


def pct(x, d=1):
    return "–" if x is None or not np.isfinite(x) else f"{x * 100:.{d}f}%".replace("-", "−")


def f2(x, d=2):
    return "–" if x is None or not np.isfinite(x) else f"{x:.{d}f}".replace("-", "−")


def pnum(x, d=1):
    """Percentage as a bare number (for table columns whose header carries the unit)."""
    return "–" if x is None or not np.isfinite(x) else f"{x * 100:.{d}f}".replace("-", "−")


WORDS = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten"]


def nw(n: int) -> str:
    """Small counts as words (IEEE/editorial style); larger numbers as digits."""
    return WORDS[n] if 0 <= n < len(WORDS) else str(n)


def Nw(n: int) -> str:
    return nw(n).capitalize()


AMERICAN = {
    "normalisation": "normalization", "normalised": "normalized", "regularised": "regularized", "regularisation": "regularization",
    "standardised": "standardized", "summarises": "summarizes", "summarise": "summarize", "optimisation": "optimization",
    "emphasises": "emphasizes", "modelling": "modeling", "behaviour": "behavior", "favourable": "favorable",
    "annualised": "annualized", "maximise": "maximize", "maximising": "maximizing", "minimise": "minimize",
    "recognise": "recognize", "analyse": "analyze", "characterise": "characterize", "colour": "color",
}
AMERICAN_RE = re.compile(r"\b(" + "|".join(AMERICAN) + r")\b", re.IGNORECASE)


def americanize(text: str) -> str:
    def sub(m):
        w = m.group(0)
        rep = AMERICAN[w.lower()]
        return rep.capitalize() if w[0].isupper() else rep
    return AMERICAN_RE.sub(sub, text)


def signed(x, d=2):
    return "–" if x is None or not np.isfinite(x) else (f"+{x:.{d}f}" if x >= 0 else f"−{abs(x):.{d}f}")


def rng(values, fmt=pct):
    v = [x for x in values if np.isfinite(x)]
    lo, hi = fmt(min(v)), fmt(max(v))
    return lo if lo == hi else f"{lo}–{hi}"


# ---------------------------------------------------------------------------------------------
# template handling
# ---------------------------------------------------------------------------------------------


def make_clean_template(src: Path, dst: Path) -> None:
    """Strip all body content from a (transitional) copy of the IEEE template, keeping its styles,
    numbering definitions and the four section definitions (title / authors / body / end)."""
    doc = Document(str(src))
    body = doc.element.body
    sect_paras = [p for p in body.iter(qn("w:p")) if p.find(qn("w:pPr") + "/" + qn("w:sectPr")) is not None]
    if len(sect_paras) < 3:
        raise ValueError("unexpected template structure (need title, author and body sections)")
    keep = []
    for p in sect_paras[:3]:
        new = OxmlElement("w:p")
        ppr = OxmlElement("w:pPr")
        ppr.append(copy.deepcopy(p.find(qn("w:pPr") + "/" + qn("w:sectPr"))))
        new.append(ppr)
        keep.append(new)
    final = body.find(qn("w:sectPr"))
    for el in list(body):
        if el is not final:
            body.remove(el)
    for p in keep:
        final.addprevious(p)
    for section in doc.sections:  # remove the "XXX-X-XXXX-XXXX-X/XX/$XX.00 ©20XX IEEE" placeholder footer
        for part in (section.footer, section.first_page_footer, section.even_page_footer):
            for par in part.paragraphs:
                for r in list(par.runs):
                    r._r.getparent().remove(r._r)
    cp = doc.core_properties  # drop any personal metadata carried by the source copy of the template
    cp.title = cp.author = cp.comments = cp.subject = cp.keywords = cp.category = cp.last_modified_by = ""
    doc.save(str(dst))


class Builder:
    TOKEN = re.compile(r"(\*\*.+?\*\*|\*.+?\*|_\{.+?\}|\^\{.+?\})")
    CITE_GROUP = re.compile(r"\[@[^\]]+\](?:, \[@[^\]]+\])*")

    def __init__(self, template: Path):
        self.doc = Document(str(template))
        body = self.doc.element.body
        self.breaks = [p for p in body.iter(qn("w:p")) if p.find(qn("w:pPr") + "/" + qn("w:sectPr")) is not None]
        self.cursor = self.breaks[0]  # insert before the end of the title section
        self.fig_n = 0
        self.tab_n = 0
        self.eq_n = 0
        self.cited: dict[str, int] = {}  # IEEE: references numbered in order of first citation

    def cite(self, text: str) -> str:
        def resolve(m):
            nums = []
            for key in re.findall(r"@([\w-]+)", m.group(0)):
                if key not in REFERENCES:
                    raise KeyError(f"unknown reference key {key}")
                nums.append(self.cited.setdefault(key, len(self.cited) + 1))
            nums = sorted(set(nums))
            parts, i = [], 0
            while i < len(nums):  # IEEE: three or more consecutive numbers are written as a range, e.g. [1]-[4]
                j = i
                while j + 1 < len(nums) and nums[j + 1] == nums[j] + 1:
                    j += 1
                parts.append(f"[{nums[i]}]–[{nums[j]}]" if j - i >= 2 else ", ".join(f"[{n}]" for n in nums[i:j + 1]))
                i = j + 1
            return ", ".join(parts)
        return self.CITE_GROUP.sub(resolve, text)

    # -- low level -------------------------------------------------------------------------
    def _new_par(self, style: str):
        p = self.doc.add_paragraph(style=style)  # appended at the end of the body ...
        self.cursor.addprevious(p._p)  # ... then moved before the current section break
        return p

    def runs(self, par, text: str, italic: bool = False, bold: bool = False, raw: bool = False):
        text = self.cite(text if raw else americanize(text))
        for part in self.TOKEN.split(text):
            if not part:
                continue
            if part.startswith("**"):
                r = par.add_run(part[2:-2]); r.bold = True
            elif part.startswith("*"):
                r = par.add_run(part[1:-1]); r.italic = True
            elif part.startswith("_{"):
                r = par.add_run(part[2:-1]); r.font.subscript = True
            elif part.startswith("^{"):
                r = par.add_run(part[2:-1]); r.font.superscript = True
            else:
                r = par.add_run(part)
            if italic:
                r.italic = True
            if bold:
                r.bold = True
        return par

    # -- sections ----------------------------------------------------------------------------
    def enter(self, section: int):
        """0 = title, 1 = authors, 2 = two-column body."""
        self.cursor = self.breaks[section]

    def set_author_columns(self, n: int):
        cols = self.breaks[1].find(qn("w:pPr") + "/" + qn("w:sectPr") + "/" + qn("w:cols"))
        cols.set(qn("w:num"), str(n))
        if n == 1:
            cols.attrib.pop(qn("w:space"), None)

    # -- content -----------------------------------------------------------------------------
    def title(self, text: str):
        self.runs(self._new_par("paper title"), text)

    def authors(self, blocks: list[list[tuple[str, bool]]]):
        """Author blocks side by side in a borderless one-row table (keeps every block top-aligned)."""
        t = self.doc.add_table(rows=1, cols=len(blocks))
        self.cursor.addprevious(t._tbl)
        t.alignment = WD_TABLE_ALIGNMENT.CENTER
        borders = OxmlElement("w:tblBorders")
        for side in ("top", "left", "bottom", "right", "insideH", "insideV"):
            e = OxmlElement(f"w:{side}"); e.set(qn("w:val"), "nil"); borders.append(e)
        t._tbl.tblPr.append(borders)
        width = 7.0 / len(blocks)
        for cell, lines in zip(t.rows[0].cells, blocks):
            cell.width = Inches(width)
            p = cell.paragraphs[0]
            p.style = self.doc.styles["Author"]
            p.paragraph_format.space_before = Pt(0)
            for i, (line, italic) in enumerate(lines):
                r = p.add_run(line)
                r.italic = italic
                if i < len(lines) - 1:
                    r.add_break()

    def abstract(self, text: str):
        p = self._new_par("Abstract")
        p.paragraph_format.space_before = Pt(12)
        r = p.add_run("Abstract")
        r.italic = True
        self.runs(p, "—" + text)

    def keywords(self, text: str):
        p = self._new_par("Keywords")
        p.add_run("Keywords—").italic = True
        p.add_run(text)

    def h1(self, text: str):
        self._new_par("Heading 1").add_run(text)

    def h2(self, text: str):
        self._new_par("Heading 2").add_run(text)

    def h5(self, text: str):
        self._new_par("Heading 5").add_run(text)

    def para(self, text: str, cont: bool = False):
        """Body paragraph; ``cont=True`` continues a sentence after an equation (no first-line indent)."""
        p = self._new_par("Body Text")
        if cont:
            p.paragraph_format.first_line_indent = Pt(0)
        return self.runs(p, text)

    def bullets(self, items: list[str]):
        for it in items:
            self.runs(self._new_par("bullet list"), it)

    def equation(self, expr: str):
        self.eq_n += 1
        p = self._new_par("equation")
        p.add_run("\t")
        self.runs(p, expr)
        p.add_run(f"\t({self.eq_n})")
        # the template's equation style uses the Symbol font (r -> rho, no minus sign): force Times New Roman
        for run in p.runs:
            run.font.name = "Times New Roman"
            rfonts = run._r.get_or_add_rPr().get_or_add_rFonts()
            for attr in ("w:ascii", "w:hAnsi", "w:cs", "w:eastAsia"):
                rfonts.set(qn(attr), "Times New Roman")
        return self.eq_n

    def figure(self, key: str, path: Path, caption: str, width_in: float = 3.3):
        self.fig_n += 1
        assert self.fig_n == FIG[key], f"figure order mismatch for {key}"
        p = self._new_par("Normal")
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p.paragraph_format.keep_with_next = True
        p.paragraph_format.space_before = Pt(6)
        p.add_run().add_picture(str(path), width=Inches(width_in))
        self.runs(self._new_par("figure caption"), caption)

    def table(self, key: str, caption: str, header: list[str], rows: list[list[str]], widths_in: list[float],
              align: list[str] | None = None, note: str | None = None, group_rows: set[int] | None = None,
              top_header: list[tuple[str, int]] | None = None):
        """IEEE table: caption above (auto 'TABLE n.'), thin borders, 8 pt cells, optional grouped first header row
        (``top_header`` = [(label, span), ...]; a span-1 label is merged vertically with the second header row)."""
        self.tab_n += 1
        assert self.tab_n == TAB_ORDER.index(key) + 1, f"table order mismatch for {key}"
        cap = self._new_par("table head")
        cap.add_run(caption)
        cap.paragraph_format.keep_with_next = True
        n_head = 2 if top_header else 1
        t = self.doc.add_table(rows=len(rows) + n_head, cols=len(header))
        self.cursor.addprevious(t._tbl)
        t.alignment = WD_TABLE_ALIGNMENT.CENTER
        tblPr = t._tbl.tblPr
        borders = OxmlElement("w:tblBorders")
        for side in ("top", "left", "bottom", "right", "insideH", "insideV"):
            b = OxmlElement(f"w:{side}")
            b.set(qn("w:val"), "single"); b.set(qn("w:sz"), "2"); b.set(qn("w:space"), "0"); b.set(qn("w:color"), "auto")
            borders.append(b)
        tblPr.append(borders)
        layout = OxmlElement("w:tblLayout"); layout.set(qn("w:type"), "fixed"); tblPr.append(layout)
        mar = OxmlElement("w:tblCellMar")  # compact padding so narrow numeric columns do not wrap
        for side, w in (("top", 0), ("left", 43), ("bottom", 0), ("right", 43)):
            e = OxmlElement(f"w:{side}"); e.set(qn("w:w"), str(w)); e.set(qn("w:type"), "dxa"); mar.append(e)
        tblPr.append(mar)
        align = align or (["left"] + ["center"] * (len(header) - 1))
        amap = {"left": WD_ALIGN_PARAGRAPH.LEFT, "center": WD_ALIGN_PARAGRAPH.CENTER, "right": WD_ALIGN_PARAGRAPH.RIGHT}
        for j, w in enumerate(widths_in):
            t.columns[j].width = Inches(w)
        all_rows = ([[lbl for lbl, span in top_header for _ in range(span)]] if top_header else []) + [header] + rows
        for i, row in enumerate(all_rows):
            tr = t.rows[i]
            if i < n_head:
                tr._tr.get_or_add_trPr().append(OxmlElement("w:tblHeader"))
            for j, txt in enumerate(row):
                cell = tr.cells[j]
                cell.width = Inches(widths_in[j])
                par = cell.paragraphs[0]
                par.style = self.doc.styles["table col head" if i < n_head else "table copy"]
                par.alignment = WD_ALIGN_PARAGRAPH.CENTER if i < n_head else amap[align[j]]
                if top_header and i == 0:
                    continue  # filled after merging
                bold = bool(group_rows and (i - n_head) in group_rows)
                self.runs(par, txt, bold=bold)
        if top_header:
            col = 0
            for label, span in top_header:
                if span == 1:
                    merged = t.cell(0, col).merge(t.cell(1, col))
                    for extra in merged.paragraphs[1:]:  # merging concatenates paragraphs; keep only the label
                        extra._p.getparent().remove(extra._p)
                    merged.paragraphs[0].text = ""
                    self.runs(merged.paragraphs[0], label)
                else:
                    merged = t.cell(0, col).merge(t.cell(0, col + span - 1))
                    for extra in merged.paragraphs[1:]:
                        extra._p.getparent().remove(extra._p)
                    self.runs(merged.paragraphs[0], label)
                merged.paragraphs[0].alignment = WD_ALIGN_PARAGRAPH.CENTER
                col += span
        # keep each table in one block (no orphaned rows or captions at the bottom of a column)
        for i, tr in enumerate(t.rows):
            tr._tr.get_or_add_trPr().append(OxmlElement("w:cantSplit"))
            if i < len(t.rows) - 1:
                for cell in tr.cells:
                    for par in cell.paragraphs:
                        par.paragraph_format.keep_with_next = True
        if note:
            self.runs(self._new_par("table footnote"), note)

    def reference(self, text: str):
        self.runs(self._new_par("references"), text, raw=True)  # cited titles keep their original spelling

    def save(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.doc.core_properties.title = TITLE
        self.doc.core_properties.author = ", ".join(a for a, _ in AUTHORS)
        self.doc.save(str(path))


# ---------------------------------------------------------------------------------------------
# results
# ---------------------------------------------------------------------------------------------


class R:
    """All result tables used by the report."""

    def __init__(self):
        self.cfg = load_config()
        self.summary = load_json(MET / "run_summary.json")
        if self.summary.get("synthetic_data"):
            raise SystemExit("results/ contains SYNTHETIC data - refusing to build the report")
        self.tickers = self.summary["tickers"]
        self.manifest = {m["ticker"]: m for m in load_json(MET / "data_manifest.json")}
        self.ds = load_json(MET / "dataset_summary.json")
        self.best = load_json(MET / "best_params.json")
        self.thr = load_json(MET / "selected_thresholds.json")
        self.cls = pd.read_csv(MET / "classification_metrics.csv")
        self.test_cls = self.cls[self.cls["split"] == "test"]
        self.trading = pd.read_csv(MET / "trading_metrics_test.csv")
        self.boot = pd.read_csv(MET / "bootstrap_confidence_intervals.csv")
        self.wf = pd.read_csv(MET / "walk_forward_trading.csv")
        self.wfc = pd.read_csv(MET / "walk_forward_classification.csv")
        self.folds = load_json(MET / "walk_forward_folds.json")
        self.dist = pd.read_csv(MET / "class_distribution.csv")
        self.cost = pd.read_csv(MET / "robustness_cost_sensitivity.csv")
        self.vt = pd.read_csv(MET / "robustness_vol_target.csv")
        self.var = pd.read_csv(MET / "robustness_variants.csv")
        self.abl = pd.read_csv(MET / "feature_ablation.csv")
        self.cal = pd.read_csv(MET / "calibration.csv")
        self.fdict = pd.read_csv(MET / "feature_dictionary.csv")

    def c(self, ticker, key, col):
        r = self.test_cls[(self.test_cls["ticker"] == ticker) & (self.test_cls["model_key"] == key)]
        return float(r[col].iloc[0])

    def t(self, ticker, strategy, col):
        r = self.trading[(self.trading["ticker"] == ticker) & (self.trading["strategy"] == strategy)]
        return float(r[col].iloc[0])

    def conf(self):
        return self.trading[self.trading["kind"] == "ml_confidence"]


def strategy_name(key: str) -> str:
    return {"ensemble": "Ensemble (soft vote) (conf.)"}.get(key, f"{NAME[key]} (conf.)")


# ---------------------------------------------------------------------------------------------
# document content
# ---------------------------------------------------------------------------------------------


def count_tests() -> int | None:
    try:
        out = subprocess.run([sys.executable, "-m", "pytest", "--collect-only", "-q", "-p", "no:cacheprovider"], cwd=ROOT,
                             capture_output=True, text=True, timeout=300).stdout
        m = re.search(r"(\d+) tests? collected", out)
        return int(m.group(1)) if m else None
    except Exception:
        return None


def build(out: Path) -> Path:
    from feature_names import readable

    r = R()
    cfg = r.cfg
    T = r.tickers
    ds0 = r.ds[T[0]]
    sp = ds0["split"]
    n_feat = ds0["n_features"]
    tau_grid = cfg["signals"]["threshold_grid"]
    cost_bps = cfg["backtest"]["fee_bps"] + cfg["backtest"]["slippage_bps"]
    boot_cfg = cfg["robustness"]["bootstrap"]
    wcfg = cfg["walk_forward"]
    wf_years = (wcfg["first_test_year"], wcfg["last_test_year"])
    lcfg = cfg["models"]["lstm"]
    ma_name = f"MA Crossover ({cfg['baselines']['ma_short']}/{cfg['baselines']['ma_long']})"
    gap_days = 1 + cfg["target"]["horizon"]  # next-open execution: label of t needs the open of t + 1 + h

    # --- derived statistics (everything below is read from results/) -------------------------
    maj_acc = {t: r.c(t, "majority_class", "accuracy") for t in T}
    ml_acc = {t: {k: r.c(t, k, "accuracy") for k in ML_KEYS} for t in T}
    best_acc = {t: max(ml_acc[t].items(), key=lambda kv: kv[1]) for t in T}
    acc_b = r.boot[r.boot["statistic"] == "accuracy_minus_majority_class"]
    acc_above_rows, acc_below_rows = acc_b[acc_b["low"] > 0], acc_b[acc_b["high"] < 0]
    acc_above, acc_below = len(acc_above_rows), len(acc_below_rows)
    acc_below_names = [f"{s.replace(' (soft vote)', '')} on {t}" for t, s in zip(acc_below_rows["ticker"], acc_below_rows["strategy"])]
    conf = r.conf()
    bh = r.trading[r.trading["strategy"] == "Buy & Hold"].set_index("ticker")
    tol = 1e-9
    n_conf = len(conf)
    n_higher = sum(row["sharpe"] - bh.loc[row["ticker"], "sharpe"] > tol for _, row in conf.iterrows())
    clones = conf[(conf["exposure"] >= 1.0) & (conf["n_trades"] <= 1)]
    clone_val_exp = []
    for _, row in clones.iterrows():
        tt = pd.read_csv(MET / "tuning" / f"{row['ticker']}_{row['model_key']}_threshold_selection_val.csv")
        clone_val_exp.append(float(tt.loc[(tt["threshold"] - row["threshold"]).abs() < 1e-9, "exposure"].iloc[0]))
    sb = r.boot[r.boot["statistic"] == "sharpe_minus_buy_and_hold"].merge(r.trading[["ticker", "strategy", "kind"]], on=["ticker", "strategy"])
    sb_ml = sb[sb["kind"] == "ml_confidence"]
    sb_ml_above, sb_ml_below = int((sb_ml["low"] > 0).sum()), int((sb_ml["high"] < 0).sum())
    below_names = [f"{s.replace(' (conf.)', '').replace(' (soft vote)', '')} on {t}" for t, s in zip(sb_ml[sb_ml["high"] < 0]["ticker"], sb_ml[sb_ml["high"] < 0]["strategy"])]
    argmax = r.trading[r.trading["kind"] == "ml_argmax"]
    arg_higher = argmax[[row["sharpe"] - bh.loc[row["ticker"], "sharpe"] > tol for _, row in argmax.iterrows()]]
    ma = r.trading[r.trading["model_key"] == "ma_crossover"].set_index("ticker")
    pers = r.trading[r.trading["model_key"] == "persistence"].set_index("ticker")
    after_up, after_other = {}, {}
    for t in T:
        pr = pd.read_csv(ROOT / "results" / "predictions" / f"{t}_holdout_predictions.csv", index_col=0, parse_dates=True)
        po = pd.read_csv(ROOT / "results" / "predictions" / f"{t}_holdout_positions.csv", index_col=0, parse_dates=True)["Naive persistence"]
        test = pr[pr["split"] == "test"].join(po.rename("pos"), how="inner")
        after_up[t] = test.loc[test["pos"] == 1, "fwd_ret"].mean() * 1e4
        after_other[t] = test.loc[test["pos"] == 0, "fwd_ret"].mean() * 1e4
    wf_bh = r.wf[r.wf["strategy"] == "Buy & Hold"].set_index("ticker")
    wf_ml = r.wf[r.wf["strategy"].str.endswith("(conf.)")]
    wf_higher = sum(row["sharpe"] - wf_bh.loc[row["ticker"], "sharpe"] > tol for _, row in wf_ml.iterrows())
    wf_thr = [v for f in r.folds for v in f["thresholds"].values() if v is not None]
    wf_trades = wf_ml["n_trades"]
    yearly = pd.read_csv(MET / "walk_forward_yearly.csv")
    low_tau_exp = []
    for f in r.folds:
        for key, tau in f["thresholds"].items():
            if tau is not None and abs(tau - min(tau_grid)) < 1e-9:
                y = yearly[(yearly.ticker == f["ticker"]) & (yearly.year == f["test_year"]) & (yearly.strategy == strategy_name(key))]
                if len(y) and np.isfinite(y["invested_days"].iloc[0]):
                    low_tau_exp.append(float(y["invested_days"].iloc[0]) / f["test_rows"])
    n_up = {t: r.dist[(r.dist["ticker"] == t) & (r.dist["split"] == "test") & (r.dist["class"] == "Up")]["frequency"].iloc[0] for t in T}
    n_neu = {t: r.dist[(r.dist["ticker"] == t) & (r.dist["split"] == "test") & (r.dist["class"] == "Neutral")]["frequency"].iloc[0] for t in T}
    neu_all = r.dist[r.dist["class"] == "Neutral"]["frequency"]
    ml_cls = r.test_cls[r.test_cls["model_key"].isin(ML_KEYS)]
    pred_up, pred_neu = ml_cls["pred_share_up"], ml_cls["pred_share_neutral"]
    f1_worse = [(t, k) for t in T for k in ML_KEYS if r.c(t, k, "f1_macro") <= r.c(t, "majority_class", "f1_macro") + tol]
    ll_better = sum(r.c(t, k, "log_loss") < r.c(t, "majority_class", "log_loss") for t in T for k in ML_KEYS)
    lstm_up = ml_cls[ml_cls["model_key"] == "lstm"]["pred_share_up"]
    lstm_ll_worse = [t for t in T if r.c(t, "lstm", "log_loss") > r.c(t, "majority_class", "log_loss")]
    ece = r.cal["ece_p_up"]
    cw_all_none = all(r.best[t][k].get("class_weight") in (None, "None") for t in T for k in ["logistic_regression", "random_forest", "xgboost", "lstm"])
    lr_C = sorted({r.best[t]["logistic_regression"]["C"] for t in T})
    rf_leaf = {t: r.best[t]["random_forest"]["min_samples_leaf"] for t in T}
    rf_depth = sorted({r.best[t]["random_forest"]["max_depth"] for t in T})
    xgb_cfg = sorted({(r.best[t]["xgboost"]["max_depth"], r.best[t]["xgboost"]["learning_rate"], r.best[t]["xgboost"]["n_estimators"]) for t in T})
    xs = pd.read_csv(MET / "tuning" / f"{T[0]}_xgboost_search.csv")
    xgb_ll_best, xgb_ll_worst = xs["cv_log_loss_mean"].min(), xs["cv_log_loss_mean"].max()
    lstm_meta = load_json(MET / "lstm" / f"{T[0]}_lstm_history.json")
    vt = r.vt.pivot_table(index=["ticker", "strategy"], columns="sizing", values="max_drawdown")
    vt_red = (1 - vt.iloc[:, 1] / vt.iloc[:, 0])
    cost_piv = r.cost.pivot_table(index=["ticker", "strategy"], columns="cost_bps", values="sharpe")
    cmax = max(cost_piv.columns)
    ml_drop = {(t, s): cost_piv.loc[(t, s), 0] - cost_piv.loc[(t, s), cmax] for t in T for s in [strategy_name(k) for k in ML_KEYS]}
    worst_drop_key = max(ml_drop, key=ml_drop.get)
    ma_drops = [cost_piv.loc[(t, ma_name), 0] - cost_piv.loc[(t, ma_name), cmax] for t in T]
    pers_drops = [cost_piv.loc[(t, "Naive persistence"), 0] - cost_piv.loc[(t, "Naive persistence"), cmax] for t in T]
    abl = r.abl.set_index(["ticker", "feature_set"])
    var = r.var
    v_def, v_moc, v_bin = "default: next_open, three_class", "market-on-close, three_class", "next_open, binary target"

    def vsh(t, s, label):
        x = var[(var.ticker == t) & (var.strategy == s) & (var.variant == label)]["sharpe"]
        return float(x.iloc[0]) if len(x) else np.nan

    var_ml = [s for s in var["strategy"].unique() if s.endswith("(conf.)")]
    moc_diff = {(t, s): abs(vsh(t, s, v_moc) - vsh(t, s, v_def)) for t in T for s in var_ml}
    moc_max_key = max(moc_diff, key=moc_diff.get)
    bin_rows = [(t, s, vsh(t, s, v_bin) - vsh(t, "Buy & Hold", v_bin)) for t in T for s in var_ml]
    bin_higher = [x for x in bin_rows if x[2] > tol]
    bin_best = max(bin_rows, key=lambda x: x[2])
    n_tests = count_tests()
    rows_before = r.ds[T[0]]["dataset"]["rows_before_warmup_drop"]
    warmup_rows = rows_before - r.ds[T[0]]["dataset"]["rows"]
    first_feature_date = r.ds[T[0]]["dataset"]["first_complete_feature_date"]
    rows_per = {t: r.manifest[t]["rows"] for t in T}
    rows_text = f"{rows_per[T[0]]:,}" if len(set(rows_per.values())) == 1 else rng(list(rows_per.values()), lambda v: f"{int(v):,}")
    grid_txt = ", ".join(f"{v:.3f}" for v in tau_grid[:2]) + ", …, " + ", ".join(f"{v:.3f}" for v in tau_grid[-2:])
    red = wcfg["reduced_grids"]

    b = Builder(TEMPLATE)
    b.set_author_columns(1)  # authors are laid out in a borderless table instead

    # ================================ title + authors =======================================
    b.enter(0)
    b.title(TITLE)
    b.enter(1)
    b.authors([[(name, False), ("B.Tech. CSE (AI & ML), Section B", True), ("Manipal Institute of Technology, MAHE", True),
                ("Manipal, Karnataka, India", False), (f"Reg. No. {reg}", False)] for name, reg in AUTHORS])
    b.enter(2)

    # ================================ abstract ===============================================
    acc_low_txt = (f", and {nw(acc_below)} ({', '.join(acc_below_names)}) {'was' if acc_below == 1 else 'were'} reliably lower"
                   if acc_below else "")
    b.abstract(
        "Rule-based technical strategies such as moving-average crossovers are simple and interpretable, but they are "
        "static and may fail to beat a passive buy-and-hold benchmark once trading costs are counted. This report asks whether "
        "supervised machine learning can turn the same market information into better trading signals. Daily price and volume "
        f"data for SPY, AAPL and MSFT ({r.manifest[T[0]]['first_date'][:4]}–{r.manifest[T[0]]['last_date'][:4]}) are converted "
        f"into {n_feat} point-in-time technical features, and five learners (Logistic Regression, Random Forest, XGBoost, a "
        "Long Short-Term Memory network and a soft-voting ensemble) predict the direction of the next tradable return as Down, "
        "Neutral or Up. Hyperparameters are tuned inside the training period with purged validation, probabilities are converted "
        "into long/cash positions with a confidence threshold chosen on a separate validation period, and all strategies, "
        "including buy-and-hold, a 10/50 moving-average crossover and naive persistence, are backtested with next-open execution "
        f"and {cost_bps} basis points of transaction cost per trade side. On the untouched "
        f"{sp['test']['start'][:4]}–{sp['test']['end'][:4]} test period, no model's accuracy was reliably higher than always "
        f"predicting the majority class ({rng([v for d in ml_acc.values() for v in d.values()])} vs. {rng(list(maj_acc.values()))})"
        f"{acc_low_txt}. {Nw(n_higher)} of the {n_conf} ML strategies achieved a higher Sharpe ratio than buy-and-hold, none by "
        f"a statistically reliable margin, and in a {wf_years[1] - wf_years[0] + 1}-year walk-forward evaluation none outperformed "
        "buy-and-hold. Under a leakage-controlled protocol with realistic costs, the ML signals showed no detectable exploitable "
        "edge at a daily horizon for these three instruments. The reproducible, tested evaluation framework is the main "
        "contribution.")
    b.keywords("algorithmic trading, machine learning, XGBoost, LSTM, technical indicators, backtesting, walk-forward validation, "
               "time-series classification")

    # ================================ I. introduction =======================================
    b.h1("Introduction")
    b.para("Algorithmic trading systems have long relied on technical indicators such as moving averages, the Relative "
           "Strength Index (RSI) and Moving Average Convergence Divergence (MACD) to generate buy and sell decisions. Such "
           "rules are cheap to compute and easy to interpret, but they are fixed in advance and cannot adapt when market "
           "conditions change. Once transaction costs are counted, they may fail to beat a passive buy-and-hold investment, "
           "as also observed for the moving-average crossover in Section VII-C.")
    b.para("Machine learning (ML) offers an alternative: instead of hand-written rules, a model learns the mapping from "
           "indicators to future price movements from historical data. Neural networks, LSTM sequence models and ML models "
           "combined with technical indicators have been widely applied to stock prediction [@chauhan2023], [@sukma2024], "
           "[@tang2024], [@yang2022]. However, financial ML is unusually prone to evaluation errors. Features that silently "
           "use future information (look-ahead bias), random train/test splits of time-ordered data, hyperparameters tuned on "
           "the test period, and backtests without trading costs can all make a model look profitable when it is not "
           "[@lopezdeprado2018].")
    b.para("Following the approved project synopsis, this mini-project builds a reproducible framework that compares ML-based "
           "trading signals with conventional technical baselines under strictly controlled conditions. The guiding question "
           "is: *does supervised learning on technical features produce better risk-adjusted trading signals than simple "
           "rules, after costs and without access to future information?* The main contributions are:")
    b.bullets([
        "A leakage-controlled pipeline: point-in-time features verified by truncation-invariance tests, targets aligned "
        "with a realistic next-open execution, purged chronological splits, and training-only normalization.",
        "Five ML models (Logistic Regression, Random Forest, XGBoost, a Long Short-Term Memory (LSTM) network and a "
        "soft-voting ensemble). The tabular models are tuned with purged time-series cross-validation, the LSTM on a purged "
        "slice of the training period and the ensemble members on validation log loss. All are compared with three "
        "baselines through one cost-aware backtester.",
        "A probability-threshold decision rule whose threshold is selected on a validation period only.",
        f"An evaluation that goes beyond a single holdout: a {wf_years[1] - wf_years[0] + 1}-year expanding-window walk-forward, "
        "block-bootstrap confidence intervals, and sensitivity to costs, execution timing, target definition and features.",
        "An open, tested implementation" + (f" ({n_tests} automated tests)" if n_tests else "") +
        " with an interactive dashboard for demonstration.",
    ])
    b.para("Section II reviews related work. Section III formulates the problem, Section IV describes the data and features, "
           "Section V the models and training protocol, and Section VI the signal generation, backtesting and implementation. "
           "Results are discussed in Section VII, limitations in Section VIII, and Section IX concludes.")

    # ================================ II. related work ======================================
    b.h1("Related Work")
    b.h2("Reinforcement Learning for Trading")
    b.para("Several recent studies treat trading as a sequential decision problem. Huang *et al.* combine deep "
           "reinforcement learning with BiLSTM-attention networks [@huang2024drl] and study how human alignment and imitation "
           "learning can improve the consistency of algorithmic trading [@huang2024align]. Ye and Schuller propose imitative "
           "multi-loss reinforcement learning for human-aligned trading [@ye2023]. Jin proposes a mean-VaR based deep "
           "reinforcement learning framework for practical algorithmic trading [@jin2023], and Park *et al.* develop deep "
           "reinforcement learning trading robots that take stock market conditions and U.S. interest rates into account "
           "[@park2024]. These approaches learn trading policies directly, whereas this project keeps a supervised formulation "
           "so that predictive quality and trading quality can be evaluated separately.")
    b.h2("Machine Learning for Price Forecasting and Indicator-Based Trading")
    b.para("A second line of work applies ML models to price forecasting and indicator-based trading. Sevastjanov *et al.* "
           "develop a multi-model approach to algorithmic trading systems for the Forex market [@sevastjanov2024], Maji and Sen "
           "build a sectoral recommendation system from technical indicators [@maji2024], and Sukma and Namahoot merge machine "
           "learning with multi-indicator strategies [@sukma2024]. Chauhan *et al.* forecast stock prices with neural networks "
           "hypertuned by particle swarm optimization and combined by ensembling [@chauhan2023]. Other studies co-evolve neural "
           "architectures and features for stock market forecasting [@coevolution2024] or integrate principal component analysis "
           "with deep learning models [@pca2024]. Like [@sukma2024] and [@maji2024], the present work uses technical indicators, "
           "but it emphasizes how the models are validated.")
    b.h2("Sequence Models and Sentiment")
    b.para("LSTM networks [@hochreiter1997] are a common choice for price sequences. Tang *et al.* propose a multi-input LSTM "
           "approach to stock movement prediction [@tang2024], self-supervised learning has been applied to predicting stock "
           "market trends [@ssl2024], and Goel and Som apply neural networks trained with the scaled conjugate gradient "
           "algorithm to stock market prediction during the COVID-19 pandemic [@goel2023]. Sentiment-augmented models combine "
           "market data with text: LSTM networks with tweet sentiment for North American and European banks [@bacco2024], "
           "technical data with sentiment scores [@kurniawan2024], and a LASSO-LSTM model combining technical indicators and "
           "financial sentiment [@yang2022]. Sentiment features were a stretch goal of this project and are discussed as future "
           "work in Section VIII.")
    b.h2("Interpretability and Evaluation Methodology")
    b.para("Deng *et al.* use causal inference to make stock-trend prediction interpretable [@deng2024]. On methodology, López "
           "de Prado [@lopezdeprado2018] describes purging of overlapping labels and walk-forward evaluation for financial ML, "
           "and Bailey and López de Prado show that selecting the best of many backtests inflates Sharpe ratios [@bailey2014]. "
           "This project adopts these practices and reports every evaluated strategy, not only the best one.")

    # ================================ III. problem formulation ==============================
    b.h1("Problem Formulation")
    b.para("For each trading day *t*, a decision is made after the close using only information available at that time. "
           "Let *O*_{t} and *C*_{t} denote the split- and dividend-adjusted open and close. Because a signal computed from "
           "the close of day *t* cannot be traded at that same close, the order is assumed to fill at the next open and the "
           "position is held until the following open. The tradable forward return of the decision on day *t* is therefore")
    b.equation("*r*_{t} = *O*_{t+2} / *O*_{t+1} − 1.")
    b.para(f"The classification label uses a neutral band θ = {cfg['target']['neutral_threshold'] * 100:.1f}%: "
           "*y*_{t} = Down if *r*_{t} < −θ, Neutral if |*r*_{t}| ≤ θ, and Up if *r*_{t} > θ. The band is a modeling "
           "assumption that separates negligible moves from directional ones. A binary Up/Down target and the "
           "market-on-close return *C*_{t+1}/*C*_{t} − 1 are evaluated as sensitivity variants. The label of day *t* depends "
           "on prices up to day *t*+2, so its *label end date* is used to purge observations whose label crosses a split "
           "boundary. The learning task is to estimate the class probabilities *P*(*y*_{t} | **x**_{≤t}) from the feature "
           "vectors **x** observed up to day *t*. A decision rule then maps these probabilities to a position "
           f"*p*_{{t}} ∈ {{0, 1}} (cash or long). Fig. {FIG['pipeline']} summarizes the complete pipeline.")
    b.figure("pipeline", FIGS / "fig_pipeline.png",
             "System architecture of the experimental pipeline (OHLCV = open, high, low, close and volume).", width_in=3.0)

    # ================================ IV. data and features =================================
    b.h1("Data and Feature Engineering")
    b.h2("Data Acquisition and Preprocessing")
    b.para(f"Daily open, high, low, close, adjusted close and volume data were downloaded from Yahoo Finance with the "
           f"yfinance package [@yfinance] for the S&P 500 exchange-traded fund (SPY), Apple (AAPL) and Microsoft (MSFT), from "
           f"{r.manifest[T[0]]['first_date']} to {r.manifest[T[0]]['last_date']} ({rows_text} trading days per instrument). "
           "The data were sorted chronologically and checked for duplicate dates, missing or non-positive prices, and "
           "inconsistent high/low values. No values were imputed, and every correction was logged. To account for splits "
           "and dividends, the factor *f*_{t} = AdjClose_{t}/Close_{t} was applied to the open, high and low. Downloads are "
           "cached and fingerprinted with SHA-256 so that the experiment can be reproduced on identical inputs. "
           f"Fig. {FIG['prices']} shows the adjusted price paths and the chronological data split.")
    b.figure("prices", FIGS / "fig_prices.png",
             "Adjusted price growth of the three instruments (log scale) with the training, validation and test periods.")
    b.h2("Feature Engineering")
    groups = r.fdict.groupby("group", sort=False)["feature"].apply(list)
    b.para(f"The {n_feat} features, in five groups (Table {TAB['features']}), are computed from the adjusted data with pandas "
           "and NumPy. Every indicator uses trailing windows or causal recursions, so a feature at day *t* depends only on data "
           "up to *t*. Unit tests verify this property: they recompute all features on a price history truncated at ten fixed "
           "cut points, and after randomly perturbing all later prices, and check that earlier values are unchanged. All "
           "features are scale-invariant (returns, ratios and normalized oscillators). Back-adjusted price levels are "
           f"excluded because they implicitly encode future dividends. Indicator warm-up ({warmup_rows} trading days, set by "
           "the 252-day percentile of the 20-day volatility) is discarded rather than imputed, so the first usable decision day "
           f"is {first_feature_date}. RSI and the Average True Range (ATR) follow Wilder's smoothing [@wilder1978].")
    feat_rows = []
    labels = {"returns": "Returns", "technical": "Technical", "volatility": "Volatility", "volume": "Volume", "regime": "Regime"}
    desc = {"returns": "1-day, log and lagged returns; 5–60-day momentum; overnight and intraday returns; high–low range; close location",
            "technical": "Close/SMA and close/EMA (5, 10, 20, 50); SMA10/SMA50; RSI(14); MACD(12, 26, 9) line, signal and histogram; Bollinger %B and width",
            "volatility": "ATR(14)/close; 5-, 20- and 60-day volatility and their ratio; volatility-normalized return",
            "volume": "Relative volume (5, 20 days); 20-day volume z-score",
            "regime": "Volatility percentile (252 days); close/SMA200; SMA50 > SMA200; drawdown from 252-day high"}
    for g, feats in groups.items():
        feat_rows.append([labels.get(g, g), str(len(feats)), desc.get(g, ", ".join(feats))])
    b.table("features", "Feature Groups", ["Group", "No.", "Features"], feat_rows, [0.62, 0.3, 2.4], ["left", "center", "left"],
            note="SMA/EMA = simple/exponential moving average.")
    b.h2("Class Distribution")
    b.para(f"In the test period, Up days account for {rng(list(n_up.values()))} of observations and Neutral days for only "
           f"{rng(list(n_neu.values()))} (between {pct(neu_all.min())} and {pct(neu_all.max())} across all splits). A "
           f"classifier that always predicts Up therefore reaches {rng(list(maj_acc.values()))} accuracy. This majority-class "
           "rule is the minimum bar a useful model must clear, and it is the reason macro-averaged metrics are reported "
           "alongside accuracy.")

    # ================================ V. models ============================================
    b.h1("Models and Training Protocol")
    b.h2("Chronological Splits and Leakage Control")
    b.para(f"The data are split by date into training ({sp['train']['start']} to {sp['train']['end']}, {sp['train']['rows']:,} "
           f"days), validation ({sp['val']['start']} to {sp['val']['end']}, {sp['val']['rows']} days) and test "
           f"({sp['test']['start']} to {sp['test']['end']}, {sp['test']['rows']:,} days) periods (Fig. {FIG['pipeline']}). "
           "Observations are never shuffled across time. Rows whose label end date falls after their split's boundary are "
           "purged [@lopezdeprado2018]. Feature scaling is fitted on training data only: inside a scikit-learn pipeline "
           "[@pedregosa2011] for Logistic Regression, and on training rows for the LSTM. In the holdout protocol the test "
           "period is used only once, after hyperparameters, architectures, ensemble members and decision thresholds have "
           "been fixed. The walk-forward analysis (Section VII-D) later re-uses 2022–2025 as out-of-sample years.")
    b.h2("Learning Algorithms")
    xf = cfg["models"]["xgboost"]["fixed"]
    b.para("**Logistic Regression** (multinomial, L2-regularized, standardized inputs) serves as a linear, interpretable ML "
           f"baseline. **Random Forest** [@breiman2001] averages {cfg['models']['random_forest']['fixed']['n_estimators']} "
           "bootstrapped decision trees (square-root feature sampling) and captures non-linear interactions. **XGBoost** "
           "[@chen2016], the primary classifier proposed in the synopsis, fits gradient-boosted trees with a softmax objective "
           f"(row and column subsampling {xf['subsample']}, minimum child weight {xf['min_child_weight']}, L2 penalty "
           f"{xf['reg_lambda']}); balanced class weighting is implemented through sample weights. The **LSTM** "
           "[@hochreiter1997] receives windows of the last *L* standardized feature vectors (rows *t*−*L*+1 to *t*) and predicts "
           "the label of day *t*. A single LSTM layer feeds a dropout layer "
           f"(rate {lcfg['dropout']}) and a linear soft-max output. The network is implemented in PyTorch [@paszke2019] and "
           f"trained with Adam (learning rate {lcfg['learning_rate']}, weight decay {lcfg['weight_decay']}, batch size "
           f"{lcfg['batch_size']}, at most {lcfg['max_epochs']} epochs) on cross-entropy loss. The **ensemble** averages the "
           "class probabilities of the subset of models (two or more) with the lowest validation log loss. Two naive "
           "classifiers provide context: the training-set majority class, and *naive persistence*, which predicts *y*_{t} as "
           f"the class of day *t*'s close-to-close return *C*_{{t}}/*C*_{{t−1}} − 1 (same ±{cfg['target']['neutral_threshold'] * 100:.1f}% band).")
    b.h2("Hyperparameter Tuning")
    b.para(f"The tabular models are tuned by exhaustive search over small grids, using {cfg['models']['tuning']['cv_splits']}-fold "
           f"time-series cross-validation inside the training period. A purge gap of {nw(gap_days)} trading days (the label's "
           "look-ahead to *O*_{t+2}) separates each training fold from its validation fold. The folds are scored by multiclass "
           "log loss, a proper scoring rule suited to probability-based decisions. Class weighting (none or balanced) is treated "
           "as a hyperparameter. The LSTM's window length, hidden size and class weighting are chosen, and early stopping "
           f"(patience {lcfg['patience']}, best weights restored) is performed, on a purged slice formed by the last "
           f"{lcfg['early_stopping_fraction']:.0%} of the training period. The validation period therefore remains "
           "out-of-sample for every model. Unlike the tabular models, the selected LSTM is not refitted on the full training "
           f"period: it is the network fitted on the first {lstm_meta['fit_rows']:,} training days. Table {TAB['hyper']} lists "
           "the search spaces and the selected values.")
    hyper_rows = []
    sel = lambda key, fmt: " / ".join(fmt(r.best[t][key]) for t in T)  # noqa: E731
    hyper_rows.append(["LR", "C ∈ {0.001, 0.01, 0.1, 1}", sel("logistic_regression", lambda p: f"{p['C']}")])
    hyper_rows.append(["RF", "depth ∈ {3, 6, 10}; min leaf ∈ {20, 80}",
                       sel("random_forest", lambda p: f"{p['max_depth']}, {p['min_samples_leaf']}")])
    hyper_rows.append(["XGB", "depth ∈ {2, 3, 5}; η ∈ {0.03, 0.1}; trees ∈ {100, 300}",
                       sel("xgboost", lambda p: f"{p['max_depth']}, {p['learning_rate']}, {p['n_estimators']}")])
    hyper_rows.append(["LSTM", "hidden ∈ {16, 32}; window ∈ {10, 20}",
                       sel("lstm", lambda p: f"{p['hidden_size']}, {p['seq_len']}")])
    hyper_rows.append(["Ensemble", "member subset", " / ".join("+".join(SHORT[m] for m in r.best[t]["ensemble"]["members"]) for t in T)])
    b.table("hyper", "Hyperparameter Search and Selected Values", ["Model", "Search space", f"Selected ({'/'.join(T)})"], hyper_rows,
            [0.6, 1.48, 1.24], ["left", "left", "left"],
            note=("LR = Logistic Regression, RF = Random Forest, XGB = XGBoost; C = inverse regularization strength, η = learning "
                  "rate. The tabular models and the LSTM also tuned class weighting (none / balanced)"
                  + ("; the unweighted option was selected in every case." if cw_all_none else ".")))

    # ================================ VI. signals and backtest ==============================
    b.h1("Signal Generation and Backtesting")
    b.h2("Decision Rule and Baselines")
    b.para(f"An ML strategy goes long for the next session when *P*(Up) ≥ τ and stays in cash otherwise. The threshold τ is "
           f"chosen from {{{grid_txt}}} ({len(tau_grid)} values) to maximize the Sharpe ratio on the validation period (ties go "
           f"to the smallest τ), subject to being invested on at least {cfg['signals']['min_exposure']:.0%} of validation days. "
           "A second rule that goes long whenever Up is the most probable class (argmax, no threshold) was also backtested; "
           "its results are summarized in Section VII-C. The baselines use the same timing: **buy-and-hold** is invested from "
           f"the first test day; the **moving-average crossover** is long when the {cfg['baselines']['ma_short']}-day simple "
           f"moving average exceeds the {cfg['baselines']['ma_long']}-day average; and **naive persistence** is long after an "
           "up day.")
    b.h2("Backtesting Engine")
    b.para("Positions *p*_{t} and the tradable returns *R*_{t} ≡ *r*_{t} from (1) are indexed by decision day. With a one-way "
           f"cost *c* = {cost_bps} basis points (bps; {cfg['backtest']['fee_bps']} bps fee plus {cfg['backtest']['slippage_bps']} "
           "bps slippage) per unit of turnover, the equity evolves as")
    b.equation("*E*_{t} = (*E*_{t−1} − *c*·|*p*_{t} − *p*_{t−1}|·*E*_{t−1})(1 + *p*_{t}*R*_{t}),")
    b.para(f"starting from *E*_{{−1}} = {cfg['backtest']['initial_capital']:,} in cash (*p*_{{−1}} = 0). Costs are charged "
           "only when the position changes, a strategy that never trades pays nothing, and a missing prediction is treated as "
           "cash so that it can never create a position.", cont=True)
    b.h2("Evaluation Metrics")
    b.para("Classification quality is measured on the test period by accuracy, balanced accuracy, macro-averaged "
           "precision, recall and F1-score (each class weighted equally), multiclass log loss and the Brier score")
    b.equation("BS = (1/*n*) Σ_{i} Σ_{k} (*p*_{ik} − **1**[*y*_{i} = *k*])^{2},")
    b.para("where *n* is the number of test days, *p*_{ik} the predicted probability of class *k* on day *i* and **1**[·] the "
           "indicator function. Trading quality is computed from the net daily strategy returns *g*_{t} = *E*_{t}/*E*_{t−1} − 1: "
           "cumulative and annualized return, annualized volatility, the Sharpe ratio [@sharpe1994] and the Sortino ratio "
           "[@sortino1994] (risk-free rate 0, 252 days per year),", cont=True)
    b.equation("SR = √252 · mean(*g*_{t}) / std(*g*_{t}),")
    b.equation("SoR = √252 · mean(*g*_{t}) / √mean(min(*g*_{t}, 0)^{2}),")
    b.para("as well as the maximum drawdown MDD = min_{t} (*E*_{t} / max_{s≤t} *E*_{s} − 1), the win rate (share of invested "
           "days with a positive net return), annual turnover and the number of trades (days on which the position changes, "
           "so an entry and its exit count as two trades).", cont=True)
    b.h2("Robustness Protocol")
    b.para(f"Beyond the single holdout, an expanding-window walk-forward evaluation retrains every model once per year. For "
           f"each test year *Y* = {wf_years[0]}, …, {wf_years[1]}, the models are re-tuned and refitted on all data up to *Y*−2, "
           "thresholds and ensemble members are selected on *Y*−1, and year *Y* is traded out-of-sample. To keep the "
           f"computation small, the folds use {wcfg['cv_splits']}-fold purged cross-validation over reduced grids: LR "
           f"C ∈ {{{', '.join(str(v) for v in red['logistic_regression']['C'])}}}; RF depth ∈ "
           f"{{{', '.join(str(v) for v in red['random_forest']['max_depth'])}}} with minimum leaf "
           f"{red['random_forest']['min_samples_leaf'][0]}; XGBoost depth ∈ {{{', '.join(str(v) for v in red['xgboost']['max_depth'])}}} "
           f"with η = {red['xgboost']['learning_rate'][0]} and {red['xgboost']['n_estimators'][0]} trees; class weighting for "
           f"all. The LSTM keeps a fixed architecture (hidden size {lcfg['hidden_size']}, window {lcfg['seq_len']}) and tunes "
           "only class weighting. Uncertainty is quantified with a circular block bootstrap [@politis1992], [@kunsch1989] "
           f"(blocks of {boot_cfg['block_len']} days, {boot_cfg['n_boot']:,} resamples) of daily returns and prediction outcomes. "
           "This gives 95% intervals for each Sharpe ratio, for the paired difference from buy-and-hold, and for the accuracy "
           "difference from the majority class. A difference is called *statistically reliable* when its 95% interval excludes "
           "zero; the intervals are not adjusted for multiple comparisons. Further analyses vary transaction costs, position "
           "sizing and the feature set. The execution-timing and target-definition variants re-run the holdout with the "
           "reduced grids and without the LSTM.")
    env = r.summary["environment"]
    b.h2("Implementation and Reproducibility")
    b.para(f"The framework is written in Python {env['python']} with pandas {env['pandas']}, NumPy {env['numpy']}, "
           f"scikit-learn {env['scikit_learn']}, XGBoost {env['xgboost']} and PyTorch {env['torch'].split('+')[0]} (CPU only). "
           "One command runs the complete experiment (data, tuning, training, backtests, walk-forward, robustness analyses, "
           f"figures and report tables) in about {round(r.summary['total_seconds'] / 60)} minutes on a laptop CPU without a "
           f"GPU. Seeds are fixed (seed {cfg['experiment']['seed']}), library versions and input-file checksums are recorded, "
           "and every table and figure in this report is generated by code from the saved results. "
           + (f"An automated test suite of {n_tests} tests checks " if n_tests else "An automated test suite checks ")
           + "indicator values against hand calculations, point-in-time feature construction, label alignment, purging, "
           "backtest accounting and an end-to-end run. A Streamlit web dashboard recomputes the backtests live from the saved "
           "out-of-sample predictions, lets the user change the threshold, costs and position sizing, and can retrain the "
           "tabular models on freshly downloaded data for demonstration. The source code is available at "
           "https://github.com/KennethMartin06/ml-based-algorithmic-trading-signals.")

    # ================================ VII. results ==========================================
    b.h1("Results and Discussion")
    b.h2("Model Selection")
    b.para(f"Cross-validation selected strongly regularized configurations: C = {', '.join(map(str, lr_C))} for Logistic "
           f"Regression, depth-{', '.join(map(str, rf_depth))} forests (minimum leaf "
           + ", ".join(f"{v} for {t}" for t, v in rf_leaf.items()) +
           f") and the smallest XGBoost model ({', '.join(f'depth {d}, η = {lr}, {n} trees' for d, lr, n in xgb_cfg)}). "
           "These values lie on the edge of their grids, so even stronger regularization might have scored better. "
           + ("Unweighted training was preferred over balanced class weights for every model and instrument. " if cw_all_none else "")
           + f"More flexible configurations had a higher out-of-fold log loss (e.g., {T[0]} XGBoost {f2(xgb_ll_best, 3)} for the "
           f"selected configuration vs. {f2(xgb_ll_worst, 3)} for the most flexible one), which suggests that the features "
           "carry little stable, exploitable signal. This was the first indication of the main result.")

    b.h2("Classification Performance")
    cls_rows, groups_idx = [], set()
    for t in T:
        groups_idx.add(len(cls_rows))
        cls_rows.append([t, "", "", "", ""])
        for k in ["majority_class"] + ML_KEYS + ["persistence"]:
            ll = r.c(t, k, "log_loss")
            cls_rows.append([f"  {NAME[k]}", f2(r.c(t, k, "accuracy"), 3), f2(r.c(t, k, "balanced_accuracy"), 3),
                             f2(r.c(t, k, "f1_macro"), 3), f2(ll, 4)])
    b.para(f"Table {TAB['classification']} reports the test-period classification metrics. The best ML accuracy per "
           f"instrument ({', '.join(f'{NAME[k]} {pct(v)} on {t}' for t, (k, v) in best_acc.items())}) is within one "
           f"percentage point of the majority-class baseline. Fig. {FIG['accuracy']} shows this accuracy difference for all "
           f"{len(acc_b)} model–instrument pairs: {len(acc_b) - acc_above - acc_below} intervals include zero, "
           f"{'none' if acc_above == 0 else nw(acc_above)} {'lies' if acc_above == 1 else 'lie'} entirely above zero, and "
           f"{'none' if acc_below == 0 else nw(acc_below)} {'lies' if acc_below == 1 else 'lie'} entirely below it.")
    b.table("classification", "Test-Period Classification Metrics", ["Model", "Acc.", "Bal. acc.", "Macro F1", "Log loss"], cls_rows,
            [1.2, 0.48, 0.54, 0.54, 0.56], ["left", "center", "center", "center", "center"], group_rows=groups_idx,
            note="Macro F1 weights Down, Neutral and Up equally. The majority-class log loss uses the training-set class "
                 "frequencies (the prior). Naive persistence outputs hard labels only (no log loss).")
    b.figure("accuracy", FIGS / "fig_accuracy_ci.png",
             f"Test accuracy minus majority-class accuracy in percentage points (pp) with {int(boot_cfg['ci'] * 100)}% "
             "block-bootstrap confidence intervals.")
    f1_exc = "; ".join(f"{NAME[k]} on {t}" for t, k in f1_worse)
    b.para(f"Fig. {FIG['confusion']} shows the confusion matrices of XGBoost. The model predicts Up on most days and never "
           f"predicts Neutral. Across all ML models and instruments, Up is predicted on {rng(pred_up.tolist())} of test days "
           f"and Neutral on {rng(pred_neu.tolist())}. This is the expected behavior of a well-regularized model in a "
           "near-random environment: it stays close to the class prior. Macro F1 exceeds the majority baseline's for "
           f"{len(ML_KEYS) * len(T) - len(f1_worse)} of {len(ML_KEYS) * len(T)} model–instrument pairs"
           + (f" (the exception being {f1_exc})" if f1_worse else "") +
           ", and only because Down is occasionally predicted. The probabilities themselves are reasonable: "
           f"{nw(ll_better)} of {len(T) * len(ML_KEYS)} model–instrument pairs achieve a lower test log loss than the prior, and "
           f"the expected calibration error of *P*(Up) is between {f2(ece.min(), 3)} and {f2(ece.max(), 3)}.")
    b.figure("confusion", FIGS / "fig_confusion_xgb.png",
             "Test-period confusion matrices of XGBoost (D = Down, N = Neutral, U = Up; counts of days).")

    b.h2("Trading Performance")
    trade_rows, gidx = [], set()
    for t in T:
        gidx.add(len(trade_rows))
        trade_rows.append([t, "", "", "", "", "", ""])
        for s, label in [("Buy & Hold", "Buy & Hold"), (ma_name, "MA crossover"), ("Naive persistence", "Persistence")] + \
                [(strategy_name(k), NAME[k].replace("Logistic Regression", "Logistic Reg.")) for k in ML_KEYS]:
            th = r.t(t, s, "threshold")
            trade_rows.append([f"  {label}", f2(th, 3) if np.isfinite(th) else "–", pnum(r.t(t, s, "cumulative_return")),
                               f2(r.t(t, s, "sharpe")), f2(r.t(t, s, "sortino")), pnum(r.t(t, s, "max_drawdown")),
                               str(int(r.t(t, s, "n_trades")))])
    clone_txt = ", ".join(f"{NAME[row['model_key']]} on {row['ticker']}" for _, row in clones.iterrows())
    b.para(f"Table {TAB['trading']} summarizes the trading results on the test period, net of costs. "
           f"{Nw(len(clones))} of the {n_conf} ML strategies ({clone_txt}) were invested on every test day and are therefore "
           "identical to buy-and-hold: on the 2020–2021 validation period, a strong bull market apart from the brief COVID-19 "
           f"crash, their Sharpe-maximizing threshold kept them invested on at least {pct(min(clone_val_exp))} of validation "
           f"days. Of the remaining strategies, {nw(n_higher)} achieved a higher Sharpe ratio than buy-and-hold. "
           f"Fig. {FIG['equity']} illustrates the SPY test period, in which the ML equity curves follow buy-and-hold closely. "
           f"The paired bootstrap intervals (Fig. {FIG['sharpe']}) show that "
           f"{'none' if sb_ml_above == 0 else nw(sb_ml_above)} of these differences {'is' if sb_ml_above <= 1 else 'are'} "
           "reliably positive"
           + (f". The interval for {', '.join(below_names)} lies just below zero, but with {len(sb_ml)} unadjusted intervals "
              "about one such exclusion is expected by chance." if sb_ml_below else "."))
    b.table("trading", "Test-Period Trading Results (Net of Costs)", ["Strategy", "τ", "Ret. (%)", "Sharpe", "Sortino", "MDD (%)", "Trades"],
            trade_rows, [0.92, 0.34, 0.42, 0.4, 0.44, 0.42, 0.4], group_rows=gidx,
            note=f"Ret. = cumulative return; MDD = maximum drawdown. ML strategies use the confidence rule with τ selected on "
                 f"validation; one-way cost {cost_bps} bps per unit turnover.")
    b.figure("equity", FIGS / "fig_equity_spy.png",
             "SPY test-period portfolio value net of costs. Logistic Regression, Random Forest and the ensemble coincide with buy-and-hold.")
    b.figure("sharpe", FIGS / "fig_sharpe_ci.png",
             f"Difference in test Sharpe ratio from buy-and-hold with {int(boot_cfg['ci'] * 100)}% paired block-bootstrap intervals. "
             "Points at zero without an interval are strategies identical to buy-and-hold.")
    arg_names = ", ".join(f"{NAME[k]} on {t} ({f2(sh)} vs. {f2(bh.loc[t, 'sharpe'])})"
                          for t, k, sh in zip(arg_higher["ticker"], arg_higher["model_key"], arg_higher["sharpe"]))
    b.para(f"With the argmax rule, the ML strategies were invested on {rng(argmax['exposure'].tolist(), lambda v: pct(v, 0))} of "
           f"test days, and their Sharpe ratios ranged from {f2(argmax['sharpe'].min())} to {f2(argmax['sharpe'].max())}. "
           f"{Nw(len(arg_higher))} of {len(argmax)} exceeded buy-and-hold" + (f" ({arg_names})" if len(arg_higher) else "") +
           ". No bootstrap intervals were computed for these strategies, so, given the number of comparisons, they are not "
           "evidence of an edge.")
    b.para(f"Among the strategies in Table {TAB['trading']}, the moving-average crossover had the smallest maximum drawdown on "
           f"every instrument ({', '.join(pct(ma.loc[t, 'max_drawdown']) for t in T)}) because it sits out sustained declines. "
           "However, it re-entered late after the sharp recoveries, and its Sharpe ratio "
           f"({', '.join(f2(ma.loc[t, 'sharpe']) for t in T)}) was below buy-and-hold's on every instrument. Naive persistence "
           f"lost {rng([-pers.loc[t, 'cumulative_return'] for t in T])} of capital through roughly "
           f"{int(round(pers['n_trades'].mean(), -1))} trades. Before costs its Sharpe ratio "
           f"({rng([cost_piv.loc[(t, 'Naive persistence'), 0] for t in T], f2)}) was positive only because it holds rising "
           f"assets about half of the time, and it stayed below buy-and-hold's ({rng([cost_piv.loc[(t, 'Buy & Hold'), 0] for t in T], f2)}). "
           f"The mean next-session return after an up day ({rng(list(after_up.values()), lambda v: f'{v:.1f}')} bps) was lower "
           f"than after a down or flat day ({rng(list(after_other.values()), lambda v: f'{v:.1f}')} bps), so there is no "
           "evidence of exploitable daily momentum.")

    b.h2("Walk-Forward Evaluation")
    wf_names = ["Buy & Hold", ma_name] + [strategy_name(k) for k in ML_KEYS]
    wf_rows = []
    for s in wf_names:
        sub = r.wf[r.wf["strategy"] == s].set_index("ticker")
        label = s.replace(" (conf.)", "").replace(" (soft vote)", "").replace(" (10/50)", "").replace("Logistic Regression", "Logistic Reg.")
        wf_rows.append([label] + [f2(sub.loc[t, "sharpe"]) for t in T] + [pnum(sub.loc[t, "cumulative_return"], 0) for t in T])
    wf_acc_text = ", ".join(
        f"{t}: best {pct(r.wfc[(r.wfc.ticker == t) & (r.wfc.model_key != 'majority_class')]['accuracy'].max())} vs. "
        f"{pct(float(r.wfc[(r.wfc.ticker == t) & (r.wfc.model_key == 'majority_class')]['accuracy'].iloc[0]))}" for t in T)
    wf_lead = ("None of the " + str(len(wf_ml))) if wf_higher == 0 else (Nw(wf_higher) + " of the " + str(len(wf_ml)))
    b.para(f"Table {TAB['walkforward']} and Fig. {FIG['walkforward']} show the {wf_years[0]}–{wf_years[1]} walk-forward results, "
           "in which every model is re-tuned and refitted each year on the data up to two years before the test year, with the "
           f"intervening year used to select thresholds and ensemble members. {wf_lead} ML strategies achieved a higher "
           "out-of-sample Sharpe ratio than buy-and-hold. Re-selecting the threshold on a fresh validation year makes it swing "
           f"between {min(wf_thr):.2f} (almost always invested; {pct(np.mean(low_tau_exp), 0)} of test days on average) and "
           f"{max(wf_thr):.2f} (rarely invested) from one year to the next. This produced {int(wf_trades.min())}–{int(wf_trades.max())} "
           "trades per strategy and missed large parts of the bull market. In the walk-forward, out-of-sample accuracy was "
           f"below that of the majority class for every model ({wf_acc_text}).")
    b.table("walkforward", f"Walk-Forward Results ({wf_years[0]}–{wf_years[1]})", ["Strategy"] + T + T,
            wf_rows, [0.92] + [0.4] * len(T) + [0.4] * len(T), note="Out-of-sample results of the concatenated walk-forward years, net of costs.",
            top_header=[("Strategy", 1), ("Sharpe ratio", len(T)), ("Cum. return (%)", len(T))])
    b.figure("walkforward", FIGS / "fig_walk_forward.png",
             "Walk-forward out-of-sample portfolio value (log scale) with annual retraining; buy-and-hold dashed.")

    b.h2("Robustness Analyses")
    rob_rows = []
    for s in ["Buy & Hold", ma_name] + [strategy_name(k) for k in ML_KEYS]:
        sh0 = np.mean([cost_piv.loc[(t, s), 0] for t in T])
        sh40 = np.mean([cost_piv.loc[(t, s), cmax] for t in T])
        base = np.mean([r.t(t, s, "sharpe") for t in T])
        base_dd = np.mean([r.t(t, s, "max_drawdown") for t in T])
        if s == "Buy & Hold" or s.endswith("(conf.)"):
            vt_sh = np.mean([float(r.vt[(r.vt.ticker == t) & (r.vt.strategy == s) & (r.vt.sizing.str.startswith("vol"))]["sharpe"].iloc[0]) for t in T])
            vt_dd = np.mean([float(r.vt[(r.vt.ticker == t) & (r.vt.strategy == s) & (r.vt.sizing.str.startswith("vol"))]["max_drawdown"].iloc[0]) for t in T])
        else:  # volatility targeting was evaluated for buy-and-hold and the ML strategies only
            vt_sh = vt_dd = np.nan
        label = s.replace(" (conf.)", "").replace(" (soft vote)", "").replace(" (10/50)", "").replace("Logistic Regression", "Logistic Reg.")
        rob_rows.append([label, f2(sh0), f2(base), f2(sh40), f2(vt_sh), pnum(base_dd, 0), pnum(vt_dd, 0)])
    wt, ws = worst_drop_key
    b.para(f"Table {TAB['robust']} summarizes the sensitivity of the test results, averaged over the three instruments. "
           "Buy-and-hold and the always-invested ML strategies are almost insensitive to transaction costs. Between 0 and "
           f"{cmax} bps, the ML strategies that trade lose up to {f2(ml_drop[worst_drop_key])} Sharpe ratio on a single instrument "
           f"({ws.replace(' (conf.)', '')} on {wt}: {f2(cost_piv.loc[worst_drop_key, 0])} → {f2(cost_piv.loc[worst_drop_key, cmax])}), "
           f"the moving-average crossover {rng(ma_drops, f2)}, and naive persistence {rng(pers_drops, f2)}. Volatility-targeted "
           "sizing reduced the maximum drawdown of buy-and-hold and of every ML strategy by "
           f"{rng(vt_red.tolist(), lambda v: f'{v * 100:.0f}%')}. Because ML and buy-and-hold benefit alike, the improvement comes "
           "from risk management rather than prediction.")
    b.table("robust", "Robustness of Test Sharpe Ratio and Drawdown (Mean of Three Instruments)",
            ["Strategy", "0 bps", f"{cost_bps} bps", f"{cmax} bps", "Vol-tgt", "Fixed", "Vol-tgt"],
            rob_rows, [0.8, 0.4, 0.42, 0.42, 0.42, 0.4, 0.46],
            note=f"Costs are one-way per unit turnover ({cost_bps} bps is the default). Vol-tgt = exposure "
                 "min(1, 15% / (σ_{20}·√252)), where σ_{20} is the 20-day realized volatility known at the decision time "
                 "(no leverage). MDD = maximum drawdown.",
            top_header=[("Strategy", 1), ("Sharpe ratio", 4), ("MDD (%)", 2)])
    mt, ms = moc_max_key
    bt_, bs_, bgap = bin_best
    b.para("The execution-timing and target variants were re-run with the reduced grids and without the LSTM, so their "
           f"next-open values differ from Table {TAB['trading']} (e.g., AAPL Logistic Regression "
           f"{f2(vsh('AAPL', 'Logistic Regression (conf.)', v_def))} vs. {f2(r.t('AAPL', 'Logistic Regression (conf.)', 'sharpe'))}) "
           "only because the grids and thresholds differ. This itself shows how fragile individual results are. In that "
           "re-run, switching to the optimistic market-on-close execution changed the test Sharpe ratio of the AAPL Random "
           f"Forest strategy from {f2(vsh('AAPL', 'Random Forest (conf.)', v_def))} to {f2(vsh('AAPL', 'Random Forest (conf.)', v_moc))} "
           f"and that of AAPL Logistic Regression from {f2(vsh('AAPL', 'Logistic Regression (conf.)', v_def))} to "
           f"{f2(vsh('AAPL', 'Logistic Regression (conf.)', v_moc))}, with no consistent direction. With a binary Up/Down target, "
           f"{nw(len(bin_higher))} of {len(bin_rows)} ML strategies had a higher test Sharpe ratio than buy-and-hold (largest: "
           f"{bs_.replace(' (conf.)', '').replace(' (soft vote)', '')} on {bt_}, {f2(vsh(bt_, bs_, v_bin))} vs. "
           f"{f2(vsh(bt_, 'Buy & Hold', v_bin))}). No bootstrap intervals were computed for the variants, so they neither "
           "confirm nor overturn the main result. Feature-group ablation of XGBoost changes test log loss only in the third "
           f"decimal place: for SPY, the 14 return features alone ({f2(abl.loc[('SPY', 'returns only'), 'test_log_loss'], 3)}) "
           f"are as good as all {n_feat} features ({f2(abl.loc[('SPY', 'all features'), 'test_log_loss'], 3)}).")
    imp = pd.read_csv(MET / "feature_importance" / "SPY_xgboost_permutation.csv")
    top_by_t = {t: pd.read_csv(MET / "feature_importance" / f"{t}_xgboost_permutation.csv")["feature"].head(2).tolist() for t in T}
    b.para(f"Permutation importance on the validation period (Fig. {FIG['importance']}) is small in absolute terms (at most a "
           f"{imp['importance_mean'].max() * 1000:.1f}×10^{{−3}} increase in log loss). The most important features also differ "
           f"between instruments: {'; '.join(f'{t}: ' + ' and '.join(readable(f) for f in v) for t, v in top_by_t.items())}. "
           "Such unstable rankings are consistent with models fitting noise rather than a persistent relationship. Importance "
           "measures what a model relies on, not causality.")
    b.figure("importance", FIGS / "fig_importance_spy.png",
             "Top-10 permutation importances of XGBoost for SPY (validation period; error bars = standard deviation over 10 repeats).")

    b.h2("Discussion")
    b.para("Taken together, the results show that a carefully controlled ML pipeline found no reliable daily edge in these "
           "liquid instruments. This is consistent with weak-form market efficiency [@fama1970]: information in past prices "
           "and volumes is largely reflected in current prices. The sequence model did not change this picture. With early "
           "stopping on a purged slice of the training period, the LSTM also stayed close to the class prior (Up predicted on "
           f"{rng(lstm_up.tolist(), lambda v: pct(v, 0))} of test days)"
           + (f", and its test log loss was worse than the prior on {' and '.join(lstm_ll_worse)}" if lstm_ll_worse else "") +
           ". The value of the ML components therefore lies in what they reveal. Proper validation shows that an apparently "
           "good model is indistinguishable from buy-and-hold, and the same analysis exposes how strongly backtest results "
           "depend on thresholds, execution assumptions and costs. A negative result obtained under these controls is more "
           "informative than a positive result from a leaky evaluation.")

    # ================================ VIII. limitations =====================================
    b.h1("Limitations and Future Work")
    b.bullets([
        "Universe and period: three large, successful U.S. instruments chosen with hindsight (survivorship bias) and a single "
        "four-year test period. The conclusions may not hold for other markets or horizons.",
        f"Trading assumptions: a constant proportional cost ({cost_bps} bps per unit of turnover) regardless of volatility or "
        "liquidity, full fills at the opening price, long/cash positions only, and no market impact, borrowing costs or taxes.",
        "Multiple testing: many models, thresholds and rules are compared without adjustment, and no deflated Sharpe ratio "
        "[@bailey2014] is applied. A single favorable or unfavorable result, such as the one interval below zero, should "
        "therefore be read together with the number of comparisons.",
        "Sentiment extension: the retrieval-augmented news-sentiment layer proposed in the synopsis was not implemented. No "
        "free, point-in-time news archive covering 2010–2025 was available, and fabricated or look-ahead-contaminated "
        "sentiment would invalidate the comparison. A leakage-safe design retrieves only articles published before each "
        "decision, scores them with a financial sentiment model, and adds the aggregates as features under the same protocol.",
        "Future work: longer prediction horizons, cross-sectional targets across many stocks, probability calibration, and "
        "portfolio-level allocation.",
    ])

    # ================================ IX. conclusion ========================================
    b.h1("Conclusion")
    b.para("This mini-project implemented and evaluated a complete ML-based trading-signal framework. It covers leakage-controlled "
           "features and targets, five tuned learners including an LSTM and an ensemble, probability-threshold signals, a "
           "cost-aware backtester and a multi-layered robustness analysis. Compared with buy-and-hold and a moving-average "
           "crossover on SPY, AAPL and MSFT, none of the ML strategies delivered a statistically reliable improvement in "
           "predictive accuracy or risk-adjusted return, either on the 2022–2025 holdout or across ten walk-forward years. "
           "The results suggest that evaluation choices can matter as much as the choice of model: switching from next-open "
           f"to market-on-close execution alone changed an individual test Sharpe ratio by up to {f2(moc_diff[moc_max_key])}, "
           "while more flexible models did not improve out-of-fold log loss. The open implementation and interactive dashboard "
           "allow every result to be reproduced and inspected.")

    # ================================ acknowledgment + references ===========================
    b.h5("Acknowledgment")
    b.para("The authors thank Dr. Narendra V G, Professor, and Dr. Ashwath Rao B, Assistant Professor (Selection Grade), "
           "School of Computer Engineering, Manipal Institute of Technology, for their guidance and feedback.")
    b.h5("References")
    uncited = [k for k in REFERENCES if k not in b.cited]
    if uncited:
        raise ValueError(f"references never cited in the text: {uncited}")
    for key, _ in sorted(b.cited.items(), key=lambda kv: kv[1]):
        b.reference(REFERENCES[key])

    b.save(out)
    return out


REFERENCES = {
    "huang2024drl": "Y. Huang, X. Wan, L. Zhang, and X. Lu, “A novel deep reinforcement learning framework with BiLSTM-Attention networks for algorithmic trading,” *Expert Syst. Appl.*, vol. 240, art. no. 122581, 2024, doi: 10.1016/j.eswa.2023.122581.",
    "huang2024align": "Y. Huang, C. Zhou, K. Cui, and X. Lu, “Improving algorithmic trading consistency via human alignment and imitation learning,” *Expert Syst. Appl.*, vol. 253, art. no. 124350, 2024, doi: 10.1016/j.eswa.2024.124350.",
    "ye2023": "Z. J. Ye and B. W. Schuller, “Human-aligned trading by imitative multi-loss reinforcement learning,” *Expert Syst. Appl.*, vol. 234, art. no. 120939, 2023, doi: 10.1016/j.eswa.2023.120939.",
    "sevastjanov2024": "P. Sevastjanov, K. Kaczmarek, and L. Rutkowski, “A multi-model approach to the development of algorithmic trading systems for the Forex market,” *Expert Syst. Appl.*, vol. 236, art. no. 121310, 2024, doi: 10.1016/j.eswa.2023.121310.",
    "deng2024": "Y. Deng, Y. Liang, and S.-M. Yiu, “Towards interpretable stock trend prediction through causal inference,” *Expert Syst. Appl.*, vol. 238, part B, art. no. 121654, 2024, doi: 10.1016/j.eswa.2023.121654.",
    "maji2024": "G. Maji and S. Sen, “Sectoral recommendation system for medium term investment using technical indicators,” *Expert Syst. Appl.*, vol. 258, art. no. 125161, 2024, doi: 10.1016/j.eswa.2024.125161.",
    "chauhan2023": "A. Chauhan, S. J. Shivaprakash, H. Sabireen, A. Q. Md, and N. Venkataraman, “Stock price forecasting using PSO hypertuned neural nets and ensembling,” *Appl. Soft Comput.*, vol. 147, art. no. 110835, 2023, doi: 10.1016/j.asoc.2023.110835.",
    "ssl2024": "Z. Ying *et al.*, “Predicting stock market trends with self-supervised learning,” *Neurocomputing*, vol. 568, art. no. 127033, 2024, doi: 10.1016/j.neucom.2023.127033.",
    "coevolution2024": "F. Hafiz, J. Broekaert, D. La Torre, and A. Swain, “Co-evolution of neural architectures and features for stock market forecasting: A multi-objective decision perspective,” *Decis. Support Syst.*, vol. 174, art. no. 114015, 2023, doi: 10.1016/j.dss.2023.114015.",
    "jin2023": "B. Jin, “A mean-VaR based deep reinforcement learning framework for practical algorithmic trading,” *IEEE Access*, vol. 11, pp. 28920–28933, 2023, doi: 10.1109/ACCESS.2023.3259108.",
    "sukma2024": "N. Sukma and C. S. Namahoot, “An algorithmic trading approach merging machine learning with multi-indicator strategies for optimal performance,” *IEEE Access*, vol. 12, pp. 188154–188173, 2024, doi: 10.1109/ACCESS.2024.3516053.",
    "park2024": "J.-H. Park, J.-H. Kim, and J.-H. Huh, “Deep reinforcement learning robots for algorithmic trading: Considering stock market conditions and U.S. interest rates,” *IEEE Access*, vol. 12, pp. 20705–20725, 2024, doi: 10.1109/ACCESS.2024.3361035.",
    "bacco2024": "L. Bacco, L. Petrosino, D. Arganese, L. Vollero, M. Papi, and M. Merone, “Investigating stock prediction using LSTM networks and sentiment analysis of tweets under high uncertainty: A case study of North American and European banks,” *IEEE Access*, vol. 12, pp. 122239–122248, 2024, doi: 10.1109/ACCESS.2024.3450311.",
    "kurniawan2024": "A. Kurniawan, Indrabayu, and M. Yusuf, “Stock price prediction using technical data and sentiment score,” in *Proc. IEEE Int. Conf. Industry 4.0, Artif. Intell., Commun. Technol. (IAICT)*, Bali, Indonesia, 2024, pp. 360–366, doi: 10.1109/IAICT62357.2024.10617768.",
    "tang2024": "P. Tang, C. Tang, and K. Wang, “Stock movement prediction: A multi-input LSTM approach,” *J. Forecasting*, vol. 43, no. 5, pp. 1199–1211, 2024, doi: 10.1002/for.3071.",
    "pca2024": "T. Uçkan, “Integrating PCA with deep learning models for stock market forecasting: An analysis of Turkish stocks markets,” *J. King Saud Univ. – Comput. Inf. Sci.*, vol. 36, no. 8, art. no. 102162, 2024, doi: 10.1016/j.jksuci.2024.102162.",
    "goel2023": "H. Goel and B. K. Som, “Stock market prediction, COVID-19 pandemic and neural networks: An SCG algorithm application,” *EconomiA*, vol. 24, no. 1, pp. 134–146, 2023, doi: 10.1108/ECON-07-2022-0101.",
    "yang2022": "J. Yang, Y. Wang, and X. Li, “Prediction of stock price direction using the LASSO-LSTM model combines technical indicators and financial sentiment analysis,” *PeerJ Comput. Sci.*, vol. 8, art. no. e1148, 2022, doi: 10.7717/peerj-cs.1148.",
    "chen2016": "T. Chen and C. Guestrin, “XGBoost: A scalable tree boosting system,” in *Proc. 22nd ACM SIGKDD Int. Conf. Knowl. Discov. Data Min.*, 2016, pp. 785–794.",
    "hochreiter1997": "S. Hochreiter and J. Schmidhuber, “Long short-term memory,” *Neural Comput.*, vol. 9, no. 8, pp. 1735–1780, 1997.",
    "breiman2001": "L. Breiman, “Random forests,” *Mach. Learn.*, vol. 45, no. 1, pp. 5–32, 2001.",
    "pedregosa2011": "F. Pedregosa *et al.*, “Scikit-learn: Machine learning in Python,” *J. Mach. Learn. Res.*, vol. 12, pp. 2825–2830, 2011.",
    "paszke2019": "A. Paszke *et al.*, “PyTorch: An imperative style, high-performance deep learning library,” in *Adv. Neural Inf. Process. Syst.*, vol. 32, 2019, pp. 8024–8035.",
    "lopezdeprado2018": "M. López de Prado, *Advances in Financial Machine Learning*. Hoboken, NJ, USA: Wiley, 2018.",
    "politis1992": "D. N. Politis and J. P. Romano, “A circular block-resampling procedure for stationary data,” in *Exploring the Limits of Bootstrap*, R. LePage and L. Billard, Eds. New York, NY, USA: Wiley, 1992, pp. 263–270.",
    "kunsch1989": "H. R. Künsch, “The jackknife and the bootstrap for general stationary observations,” *Ann. Statist.*, vol. 17, no. 3, pp. 1217–1241, 1989.",
    "fama1970": "E. F. Fama, “Efficient capital markets: A review of theory and empirical work,” *J. Finance*, vol. 25, no. 2, pp. 383–417, 1970.",
    "sharpe1994": "W. F. Sharpe, “The Sharpe ratio,” *J. Portfolio Manage.*, vol. 21, no. 1, pp. 49–58, 1994.",
    "sortino1994": "F. A. Sortino and L. N. Price, “Performance measurement in a downside risk framework,” *J. Investing*, vol. 3, no. 3, pp. 59–64, 1994.",
    "wilder1978": "J. W. Wilder, *New Concepts in Technical Trading Systems*. Greensboro, NC, USA: Trend Research, 1978.",
    "yfinance": "R. Aroussi, “yfinance: Download market data from Yahoo! Finance’s API,” GitHub repository. Accessed: Oct. 12, 2026. [Online]. Available: https://github.com/ranaroussi/yfinance",
    "bailey2014": "D. H. Bailey and M. López de Prado, “The deflated Sharpe ratio: Correcting for selection bias, backtest overfitting, and non-normality,” *J. Portfolio Manage.*, vol. 40, no. 5, pp. 94–107, 2014.",
}


def export_pdf(docx_path: Path) -> Path | None:
    """Render the report to PDF with Microsoft Word (Windows only); returns None if unavailable."""
    try:
        import pythoncom
        import win32com.client
    except ImportError:
        return None
    pdf = docx_path.with_suffix(".pdf")
    pythoncom.CoInitialize()
    word = win32com.client.DispatchEx("Word.Application")
    word.Visible = False
    word.DisplayAlerts = 0
    try:
        doc = word.Documents.Open(str(docx_path.resolve()))
        doc.Fields.Update()
        doc.Save()
        doc.ExportAsFixedFormat(str(pdf.resolve()), 17)
        pages = doc.ComputeStatistics(2)
        doc.Close(False)
        print(f"pages: {pages}")
    finally:
        word.Quit()
    return pdf


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--template-source", help="IEEE conference template (A4) to clean and store as report/ieee_conference_template_a4.docx")
    ap.add_argument("--out", default=str(OUT_DIR / OUT_NAME))
    ap.add_argument("--pdf", action="store_true", help="also export a PDF with Microsoft Word (Windows)")
    args = ap.parse_args()
    if args.template_source:
        make_clean_template(Path(args.template_source), TEMPLATE)
        print("clean template written to", TEMPLATE.relative_to(ROOT))
    out = build(Path(args.out))
    print("report written to", out)
    if args.pdf:
        pdf = export_pdf(out)
        print("pdf:", pdf)


if __name__ == "__main__":
    main()
