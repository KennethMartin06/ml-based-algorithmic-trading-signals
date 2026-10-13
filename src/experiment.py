"""End-to-end experiment orchestration.

Stages
------
1. ``train_stage``     data -> features -> targets -> purged chronological split -> tuning and
                       fitting of every model -> validation/test probabilities written to
                       ``results/predictions``; tuning tables, feature analysis, LSTM history.
2. ``evaluate_stage``  reads the saved predictions -> classification metrics -> probability
                       thresholds chosen on VALIDATION -> backtests of ML and baseline strategies on
                       the TEST period -> robustness analyses -> tables and figures.
3. ``walk_forward_stage`` expanding-window re-training (one test year at a time).
4. ``sensitivity_stage`` repeats the holdout under the optimistic market-on-close convention
                       and with a binary target, to show how much these modelling choices matter.

The test period is used exactly once per model: after every modelling choice (hyperparameters,
LSTM architecture, ensemble members, decision thresholds) has been fixed on train/validation data.
"""

from __future__ import annotations

import itertools
import platform
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from . import __version__
from . import visualization as viz
from .backtester import BacktestConfig, positions_from_probabilities, run_backtest, sharpe_ratio, volatility_target_positions
from .baselines import MajorityClassBaseline, buy_and_hold_positions, ma_crossover_positions, persistence_labels, persistence_positions
from .data_loader import load_universe
from .evaluation import (
    accuracy_difference_stat,
    bootstrap_ci,
    classification_metrics,
    expected_calibration_error,
    mutual_information_frame,
    permutation_importance_frame,
    reliability_curve,
    sharpe_difference_stat,
    sharpe_stat,
)
from .features import compute_features, feature_groups, feature_table
from .lstm_model import TORCH_AVAILABLE, TORCH_VERSION, LSTMSequenceClassifier
from .models import (
    TABULAR_MODELS,
    XGBOOST_AVAILABLE,
    XGBOOST_VERSION,
    display_name,
    fit_estimator,
    predict_proba_full,
    resolve_model_name,
    safe_log_loss,
    time_series_grid_search,
    tree_importance,
)
from .preprocessing import assemble_dataset, chronological_split, walk_forward_folds
from .targets import build_target_frame, class_distribution, label_end_offset
from .utils import class_names_for, deep_update, ensure_dir, get_logger, load_json, resolve_path, save_json, set_global_seed

LOGGER = get_logger()

ML_ORDER = ["logistic_regression", "random_forest", "xgboost", "hist_gradient_boosting", "lstm", "ensemble"]
MIN_ACTIVE_DAYS = 20  # minimum invested days for a per-regime / per-year Sharpe ratio to be reported

QUICK_OVERRIDES = {
    "experiment": {"name": "quick", "results_dir": "results_quick"},
    "data": {"tickers": ["SPY"]},
    "models": {
        "random_forest": {"fixed": {"n_estimators": 150}},
        "lstm": {"tune": False, "max_epochs": 25, "patience": 5},
    },
    "walk_forward": {"first_test_year": 2023, "models": ["logistic_regression", "xgboost"]},
    "robustness": {"bootstrap": {"n_boot": 300}, "permutation_repeats": 3, "sensitivity_variants": False},
}
SYNTHETIC_OVERRIDES = {"experiment": {"name": "synthetic", "results_dir": "results_synthetic"}}


@dataclass
class OutputPaths:
    root: Path

    def __post_init__(self):
        self.root = resolve_path(self.root)
        self.metrics = ensure_dir(self.root / "metrics")
        self.figures = ensure_dir(self.root / "figures")
        self.predictions = ensure_dir(self.root / "predictions")
        self.reports = ensure_dir(self.root / "reports")
        self.models = ensure_dir(self.root / "models")
        self.tuning = ensure_dir(self.metrics / "tuning")
        self.importance = ensure_dir(self.metrics / "feature_importance")
        self.lstm = ensure_dir(self.metrics / "lstm")


def build_config(cfg: dict, quick: bool = False, synthetic: bool = False, skip_lstm: bool = False, tickers=None,
                 skip_walk_forward: bool = False, results_dir: str | None = None) -> dict:
    """Apply CLI options. Every non-default option writes to its own results directory so that a
    partial run can never overwrite or mix with the main experiment in ``results/``."""
    if quick:
        cfg = deep_update(cfg, QUICK_OVERRIDES)
    if synthetic:
        cfg = deep_update(cfg, SYNTHETIC_OVERRIDES)
        if quick:
            cfg["experiment"]["results_dir"] = "results_synthetic_quick"
    suffix = []
    if tickers and list(tickers) != list(cfg["data"]["tickers"]):
        cfg["data"]["tickers"] = list(tickers)
        suffix.append("-".join(t.replace("^", "").replace("/", "_") for t in tickers))
    if skip_lstm:
        cfg = deep_update(cfg, {"models": {"lstm": {"enabled": False}}})
        suffix.append("nolstm")
    if skip_walk_forward:
        cfg["walk_forward"]["enabled"] = False
        suffix.append("nowf")
    if suffix:
        cfg["experiment"]["results_dir"] = cfg["experiment"]["results_dir"] + "_" + "_".join(suffix)
    if results_dir:
        cfg["experiment"]["results_dir"] = results_dir
    cfg["_synthetic"] = bool(synthetic)
    cfg["_is_default_full_run"] = not (quick or synthetic or suffix or results_dir)
    return cfg


def enabled_models(cfg: dict, names) -> list[str]:
    """Model names whose ``models.<name>.enabled`` flag is true."""
    return [n for n in names if cfg["models"].get(n, {}).get("enabled", True)]


def vol_column(cfg: dict) -> str:
    """Realised-volatility feature used for regimes and volatility targeting (vol_20 by default),
    chosen exactly as in ``features.compute_features``."""
    windows = cfg["features"].get("volatility_windows", [5, 20, 60])
    return f"vol_{20 if 20 in windows else sorted(windows)[0]}"


def lstm_internal_split(train_pos: np.ndarray, fraction: float, gap: int) -> tuple[np.ndarray, np.ndarray]:
    """Split the (contiguous) training decision rows into a fitting part and a final early-stopping
    slice of ``fraction`` of the rows, separated by ``gap`` purged rows so that no fitting label
    overlaps the slice. The validation period is therefore never used to fit the LSTM."""
    train_pos = np.asarray(train_pos, dtype=int)
    n = len(train_pos)
    n_es = max(1, int(round(n * fraction)))
    return train_pos[: max(0, n - n_es - gap)], train_pos[n - n_es:]


# ---------------------------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------------------------


def proba_cols(key: str, n_classes: int) -> list[str]:
    return [f"{key}__p{k}" for k in range(n_classes)]


def models_in_predictions(pred: pd.DataFrame) -> list[str]:
    keys = [c[: -len("__p0")] for c in pred.columns if c.endswith("__p0")]
    return sorted(keys, key=lambda k: ML_ORDER.index(k) if k in ML_ORDER else (-1 if k == "majority_class" else 99))


def prepare_ticker_dataset(prices: pd.DataFrame, cfg: dict, convention: str | None = None):
    conv = convention or cfg["execution"]["convention"]
    feats = compute_features(prices, cfg["features"])
    targets = build_target_frame(prices, cfg["target"], conv)
    data, report = assemble_dataset(feats, targets)
    return data, list(feats.columns), report


def lstm_params(lcfg: dict, override: dict, seed: int) -> dict:
    keys = ["seq_len", "hidden_size", "num_layers", "dropout", "learning_rate", "weight_decay", "batch_size",
            "max_epochs", "patience", "class_weight", "shuffle_train_batches"]
    p = {k: lcfg[k] for k in keys if k in lcfg}
    p.update(override)
    p["seed"] = seed
    return p


def select_ensemble(probas: dict, y_val: np.ndarray, n_classes: int) -> tuple[list[str], pd.DataFrame]:
    """Choose the member subset (size >= 2) whose equal-weight average has the lowest VALIDATION
    log loss."""
    keys = [k for k in ML_ORDER if k in probas and k != "ensemble"]
    rows = []
    for r in range(2, len(keys) + 1):
        for subset in itertools.combinations(keys, r):
            p = np.mean([probas[k]["val"] for k in subset], axis=0)
            rows.append({"members": "+".join(subset), "val_log_loss": safe_log_loss(y_val, p, n_classes)})
    table = pd.DataFrame(rows).sort_values("val_log_loss").reset_index(drop=True)
    return table.loc[0, "members"].split("+"), table


def select_threshold(proba, index, period_ret, up_idx, down_idx, scfg, bt_cfg) -> tuple[float | None, pd.DataFrame]:
    """Pick the P(Up) threshold that maximises the VALIDATION metric (default Sharpe) subject to a
    minimum exposure. Ties -> smallest threshold. If no threshold qualifies, None (argmax rule)."""
    rows = []
    for tau in scfg["threshold_grid"]:
        pos = positions_from_probabilities(proba, index, up_idx, down_idx, tau, scfg["mode"])
        m = run_backtest(pos, period_ret, bt_cfg).metrics
        rows.append({"threshold": tau, "sharpe": m["sharpe"], "sortino": m["sortino"], "cumulative_return": m["cumulative_return"],
                     "exposure": m["exposure"], "n_trades": m["n_trades"]})
    table = pd.DataFrame(rows)
    metric = scfg.get("selection_metric", "sharpe")
    ok = table[(table["exposure"] >= scfg.get("min_exposure", 0.05)) & table[metric].notna()]
    if ok.empty:
        return None, table
    best = ok.sort_values([metric, "threshold"], ascending=[False, True]).iloc[0]
    return float(best["threshold"]), table


# ---------------------------------------------------------------------------------------------
# stage 1: training
# ---------------------------------------------------------------------------------------------


def train_holdout_ticker(ticker, prices, cfg, paths: OutputPaths | None = None, logger=LOGGER, convention=None,
                         with_lstm: bool = True, with_analysis: bool = True) -> dict:
    seed = cfg["experiment"]["seed"]
    set_global_seed(seed)
    scheme = cfg["target"]["scheme"]
    names = class_names_for(scheme)
    K = len(names)
    theta = cfg["target"]["neutral_threshold"]
    conv = convention or cfg["execution"]["convention"]
    data, fcols, ds_report = prepare_ticker_dataset(prices, cfg, conv)
    split = chronological_split(data, cfg["split"]["train_end"], cfg["split"]["val_end"], cfg["split"].get("purge", True))
    empty = [name for name, idx in split.as_dict().items() if len(idx) == 0]
    if empty:
        raise ValueError(f"{ticker}: no labelled rows in split(s) {empty} - the data ({data.index[0].date()} to "
                         f"{data.index[-1].date()}) does not cover the configured split dates")
    gap = label_end_offset(cfg["target"]["horizon"], conv)
    X = data[fcols].to_numpy(dtype=float)
    y = data["label"].to_numpy(dtype=float)
    P = {k: data.index.get_indexer(v) for k, v in split.as_dict().items()}
    Xtr, ytr = X[P["train"]], y[P["train"]].astype(int)
    Xva, yva = X[P["val"]], y[P["val"]].astype(int)
    Xte = X[P["test"]]
    logger.info("%s: %d features | train %d | val %d | test %d rows (convention=%s)", ticker, len(fcols),
                len(Xtr), len(Xva), len(Xte), conv)

    mcfg = cfg["models"]
    probas, tuning, fitted, timings = {}, {}, {}, {}
    for name in enabled_models(cfg, TABULAR_MODELS):
        actual = resolve_model_name(name)
        if actual != name:
            logger.warning("xgboost not importable -> using HistGradientBoostingClassifier fallback (reported as such)")
        t0 = time.perf_counter()
        best, table = time_series_grid_search(actual, Xtr, ytr, mcfg[name]["grid"], mcfg[name].get("fixed"), K,
                                              mcfg["tuning"]["cv_splits"], gap, seed)
        model = fit_estimator(actual, best, Xtr, ytr, K, seed)
        probas[actual] = {"val": predict_proba_full(model, Xva, K), "test": predict_proba_full(model, Xte, K)}
        tuning[actual] = {"best_params": best, "cv_table": table}
        fitted[actual] = model
        timings[actual] = time.perf_counter() - t0
        logger.info("  %-22s best=%s  cv_logloss=%.4f  (%.1fs)", display_name(actual),
                    {k: v for k, v in best.items() if k in mcfg[name]["grid"]}, table["cv_log_loss_mean"].iloc[0], timings[actual])

    lstm_info = None
    lcfg = mcfg.get("lstm", {})
    if with_lstm and lcfg.get("enabled", False):
        if not TORCH_AVAILABLE:
            logger.warning("PyTorch not installed -> LSTM skipped (pip install torch)")
        else:
            t0 = time.perf_counter()
            if lcfg.get("tune", False):
                g = lcfg["grid"]
                candidates = [dict(zip(g, combo)) for combo in itertools.product(*g.values())]
            else:
                candidates = [{}]
            # Early stopping and architecture selection use a purged slice at the END OF THE TRAINING
            # period; the validation period is reserved for ensemble/threshold selection (as for the
            # tabular models, whose hyperparameters come from CV inside the training period).
            fit_pos, es_pos = lstm_internal_split(P["train"], lcfg.get("early_stopping_fraction", 0.2), gap)
            scaler_rows = np.arange(0, P["train"].max() + 1)
            results = []
            for cand in candidates:
                m = LSTMSequenceClassifier(n_classes=K, **lstm_params(lcfg, cand, seed)).fit(X, y, fit_pos, es_pos, scaler_rows)
                es_loss = safe_log_loss(y[es_pos], m.predict_proba(X, es_pos), K)
                results.append((es_loss, cand, m))
                logger.info("  LSTM %s -> early-stopping-slice log loss %.4f (best epoch %s)", cand or "default", es_loss, m.best_epoch)
            es_loss, cand, lstm = min(results, key=lambda r: r[0])
            probas["lstm"] = {"val": lstm.predict_proba(X, P["val"]), "test": lstm.predict_proba(X, P["test"])}
            tuning["lstm"] = {
                "best_params": lstm.get_params(),
                "cv_table": pd.DataFrame([{**{k: ("None" if v is None else v) for k, v in c.items()}, "es_slice_log_loss": v,
                                           "best_epoch": mm.best_epoch, "epochs_run": len(mm.history)}
                                          for v, c, mm in results]).sort_values("es_slice_log_loss"),
            }
            timings["lstm"] = time.perf_counter() - t0
            lstm_info = {"params": lstm.get_params(), "best_epoch": lstm.best_epoch, "history": lstm.history,
                         "fit_rows": int(len(fit_pos)), "early_stopping_rows": int(len(es_pos)),
                         "early_stopping_start": data.index[es_pos[0]].strftime("%Y-%m-%d"),
                         "early_stopping_end": data.index[es_pos[-1]].strftime("%Y-%m-%d")}
            if paths is not None:
                lstm.save(paths.models, f"{ticker}_lstm")
                save_json(lstm_info, paths.lstm / f"{ticker}_lstm_history.json")

    ensemble_table = None
    if mcfg.get("ensemble", {}).get("enabled", False) and len(probas) >= 2:
        members, ensemble_table = select_ensemble(probas, yva, K)
        probas["ensemble"] = {s: np.mean([probas[m][s] for m in members], axis=0) for s in ("val", "test")}
        tuning["ensemble"] = {"best_params": {"members": members}, "cv_table": ensemble_table}
        logger.info("  Ensemble members (validation log loss): %s", members)

    majority = MajorityClassBaseline(K).fit(ytr)
    probas["majority_class"] = {"val": majority.predict_proba(len(yva)), "test": majority.predict_proba(len(Xte))}

    # ---- predictions frame (validation + test) ---------------------------------------------
    vcol = vol_column(cfg)
    vol_median_train = float(data.loc[split.train, vcol].median())
    pers = persistence_labels(prices["close"], scheme, theta).reindex(data.index)
    frames = []
    for s in ("val", "test"):
        idx = split.as_dict()[s]
        f = pd.DataFrame(index=idx)
        f["split"] = s
        f["label"] = data.loc[idx, "label"].astype(int)
        f["fwd_ret"] = data.loc[idx, "fwd_ret"]
        f["period_ret"] = data.loc[idx, "period_ret"]
        f["realized_vol"] = data.loc[idx, vcol]  # daily volatility known at decision time
        f["high_vol"] = (data.loc[idx, vcol] > vol_median_train).astype(int)
        f["uptrend"] = (data.loc[idx, "trend_regime"] > 0).astype(int)
        f["persistence__pred"] = pers.loc[idx].astype(int)
        for key, pr in probas.items():
            f[proba_cols(key, K)] = pr[s]
        frames.append(f)
    predictions = pd.concat(frames)
    predictions.index.name = "date"

    out = {
        "ticker": ticker,
        "convention": conv,
        "predictions": predictions,
        "tuning": tuning,
        "timings": timings,
        "lstm": lstm_info,
        "feature_cols": fcols,
        "dataset_report": ds_report,
        "split": split.describe(),
        "train_vol_median": vol_median_train,
    }

    if with_analysis:
        dist = []
        for s, idx in split.as_dict().items():
            d = class_distribution(data.loc[idx, "label"], scheme)
            d.insert(0, "split", s)
            d.insert(0, "ticker", ticker)
            dist.append(d)
        out["class_distribution"] = pd.concat(dist, ignore_index=True)
        out["mutual_information"] = mutual_information_frame(Xtr, ytr, fcols, seed)
        reps = cfg["robustness"].get("permutation_repeats", 10)
        out["permutation_importance"] = {}
        out["tree_importance"] = {}
        for key, model in fitted.items():
            out["permutation_importance"][key] = permutation_importance_frame(model, Xva, yva, fcols, reps, seed, K)
            ti = tree_importance(model, fcols)
            if ti is not None:
                out["tree_importance"][key] = ti.sort_values(ascending=False).rename("importance").reset_index().rename(columns={"index": "feature"})
        xgb_key = resolve_model_name("xgboost")
        if cfg["robustness"].get("feature_ablation", True) and xgb_key in tuning:
            out["feature_ablation"] = feature_ablation(ticker, data, fcols, split, tuning[xgb_key]["best_params"], xgb_key,
                                                       out["mutual_information"], cfg, K)
        if paths is not None:
            for key, model in fitted.items():
                try:
                    import joblib

                    joblib.dump(model, paths.models / f"{ticker}_{key}.joblib")
                except Exception as exc:  # pragma: no cover
                    logger.warning("could not save %s model: %s", key, exc)
    return out


def feature_ablation(ticker, data, fcols, split, best_params, model_key, mi, cfg, K) -> pd.DataFrame:
    """XGBoost with the tuned hyperparameters on nested feature subsets. Thresholds per subset are
    chosen on validation; the test period is only scored."""
    groups = feature_groups(fcols)
    top_k = mi["feature"].head(15).tolist()  # MI ranked on TRAINING data only
    subsets = {
        "returns only": groups["returns"],
        "returns + technical": groups["returns"] + groups["technical"],
        "all except regime": [c for c in fcols if c not in groups["regime"]],
        "all except volume": [c for c in fcols if c not in groups["volume"]],
        "top-15 by mutual information": top_k,
        "all features": fcols,
    }
    scheme = cfg["target"]["scheme"]
    up, down = K - 1, 0
    bt_cfg = BacktestConfig.from_config(cfg)
    y = data["label"]
    rows = []
    for name, cols in subsets.items():
        model = fit_estimator(model_key, best_params, data.loc[split.train, cols].values, y.loc[split.train].astype(int).values, K, cfg["experiment"]["seed"])
        pv = predict_proba_full(model, data.loc[split.val, cols].values, K)
        pt = predict_proba_full(model, data.loc[split.test, cols].values, K)
        tau, _ = select_threshold(pv, split.val, data.loc[split.val, "period_ret"], up, down, cfg["signals"], bt_cfg)
        pos = positions_from_probabilities(pt, split.test, up, down, tau, cfg["signals"]["mode"])
        bt = run_backtest(pos, data.loc[split.test, "period_ret"], bt_cfg).metrics
        cm = classification_metrics(y.loc[split.test].astype(int).values, pt.argmax(1), class_names_for(scheme), pt)
        rows.append({"ticker": ticker, "feature_set": name, "n_features": len(cols), "test_log_loss": cm["log_loss"],
                     "test_accuracy": cm["accuracy"], "test_f1_macro": cm["f1_macro"], "threshold": tau,
                     "test_sharpe": bt["sharpe"], "test_cumulative_return": bt["cumulative_return"]})
    return pd.DataFrame(rows)


def train_stage(cfg: dict, frames: dict, paths: OutputPaths, logger=LOGGER) -> dict:
    """Train all holdout models for every ticker and persist predictions and training artefacts."""
    results = {}
    best_params, dist, ablation, ds_summary, timings = {}, [], [], {}, {}
    for ticker, prices in frames.items():
        res = train_holdout_ticker(ticker, prices, cfg, paths, logger)
        results[ticker] = res
        res["predictions"].to_csv(paths.predictions / f"{ticker}_holdout_predictions.csv")
        best_params[ticker] = {k: v["best_params"] for k, v in res["tuning"].items()}
        for key, t in res["tuning"].items():
            t["cv_table"].to_csv(paths.tuning / f"{ticker}_{key}_search.csv", index=False)
        res["mutual_information"].to_csv(paths.importance / f"{ticker}_mutual_information.csv", index=False)
        for key, df in res["permutation_importance"].items():
            df.to_csv(paths.importance / f"{ticker}_{key}_permutation.csv", index=False)
        for key, df in res["tree_importance"].items():
            df.to_csv(paths.importance / f"{ticker}_{key}_model_importance.csv", index=False)
        dist.append(res["class_distribution"])
        if "feature_ablation" in res:
            ablation.append(res["feature_ablation"])
        ds_summary[ticker] = {"dataset": res["dataset_report"], "split": res["split"], "n_features": len(res["feature_cols"]),
                              "train_vol_median": res["train_vol_median"]}
        timings[ticker] = res["timings"]
    save_json(best_params, paths.metrics / "best_params.json")
    save_json(ds_summary, paths.metrics / "dataset_summary.json")
    save_json(timings, paths.metrics / "training_times_seconds.json")
    pd.concat(dist, ignore_index=True).to_csv(paths.metrics / "class_distribution.csv", index=False)
    if ablation:
        pd.concat(ablation, ignore_index=True).to_csv(paths.metrics / "feature_ablation.csv", index=False)
    feature_table().to_csv(paths.metrics / "feature_dictionary.csv", index=False)
    return results


# ---------------------------------------------------------------------------------------------
# stage 2: evaluation
# ---------------------------------------------------------------------------------------------


def build_test_strategies(ticker, prices, pred, cfg, K, thresholds: dict | None = None) -> tuple[dict, dict, dict]:
    """Return (strategies, thresholds, threshold_tables). Strategies map display name ->
    dict(kind, model_key, color_key, threshold, positions on the TEST index)."""
    scfg = cfg["signals"]
    bt_cfg = BacktestConfig.from_config(cfg)
    up, down = K - 1, 0
    val, test = pred[pred["split"] == "val"], pred[pred["split"] == "test"]
    allow_short = scfg["mode"] == "long_short"
    strategies = {
        "Buy & Hold": {"kind": "baseline", "model_key": "buy_and_hold", "color_key": "buy_and_hold", "threshold": None,
                       "positions": buy_and_hold_positions(test.index)},
        f"MA Crossover ({cfg['baselines']['ma_short']}/{cfg['baselines']['ma_long']})": {
            "kind": "baseline", "model_key": "ma_crossover", "color_key": "ma_crossover", "threshold": None,
            "positions": ma_crossover_positions(prices["close"], cfg["baselines"]["ma_short"], cfg["baselines"]["ma_long"], allow_short).reindex(test.index)},
        "Naive persistence": {"kind": "baseline", "model_key": "persistence", "color_key": "persistence", "threshold": None,
                              "positions": persistence_positions(prices["close"], allow_short).reindex(test.index)},
    }
    chosen, tables = dict(thresholds or {}), {}
    for key in models_in_predictions(pred):
        if key == "majority_class":
            continue
        cols = proba_cols(key, K)
        if key not in chosen:
            chosen[key], tables[key] = select_threshold(val[cols].values, val.index, val["period_ret"], up, down, scfg, bt_cfg)
        name = display_name(key)
        strategies[f"{name} (conf.)"] = {"kind": "ml_confidence", "model_key": key, "color_key": key, "threshold": chosen[key],
                                         "positions": positions_from_probabilities(test[cols].values, test.index, up, down, chosen[key], scfg["mode"])}
        strategies[f"{name} (argmax)"] = {"kind": "ml_argmax", "model_key": key, "color_key": key, "threshold": None,
                                          "positions": positions_from_probabilities(test[cols].values, test.index, up, down, None, scfg["mode"])}
    return strategies, chosen, tables


def evaluate_holdout_ticker(ticker, prices, pred, cfg, logger=LOGGER, with_analysis=True) -> dict:
    scheme = cfg["target"]["scheme"]
    names = class_names_for(scheme)
    K = len(names)
    seed = cfg["experiment"]["seed"]
    bt_cfg = BacktestConfig.from_config(cfg)
    val, test = pred[pred["split"] == "val"], pred[pred["split"] == "test"]

    # ---- classification -------------------------------------------------------------------
    cls_rows, cms = [], {}
    for split_name, part in (("val", val), ("test", test)):
        for key in models_in_predictions(pred):
            proba = part[proba_cols(key, K)].values
            m = classification_metrics(part["label"].values, proba.argmax(1), names, proba)
            cls_rows.append({"ticker": ticker, "split": split_name, "model_key": key, "model": display_name(key), **_flat_cls(m)})
            if split_name == "test":
                cms[display_name(key)] = np.array(m["confusion_matrix"])
        m = classification_metrics(part["label"].values, part["persistence__pred"].values, names, None)
        cls_rows.append({"ticker": ticker, "split": split_name, "model_key": "persistence", "model": display_name("persistence"), **_flat_cls(m)})
        if split_name == "test":
            cms[display_name("persistence")] = np.array(m["confusion_matrix"])
    cls = pd.DataFrame(cls_rows)

    # ---- trading ----------------------------------------------------------------------------
    strategies, thresholds, thr_tables = build_test_strategies(ticker, prices, pred, cfg, K)
    backtests = {name: run_backtest(s["positions"], test["period_ret"], bt_cfg) for name, s in strategies.items()}
    trade_rows = []
    for name, s in strategies.items():
        trade_rows.append({"ticker": ticker, "strategy": name, "kind": s["kind"], "model_key": s["model_key"],
                           "color_key": s["color_key"], "threshold": s["threshold"], **backtests[name].metrics})
    trading = pd.DataFrame(trade_rows)
    out = {"classification": cls, "trading": trading, "thresholds": thresholds, "threshold_tables": thr_tables,
           "backtests": backtests, "strategies": strategies, "confusion_matrices": cms}
    if not with_analysis:
        return out

    main = [n for n, s in strategies.items() if s["kind"] in ("baseline", "ml_confidence")]

    # ---- transaction-cost sensitivity (positions fixed) -------------------------------------
    rows = []
    for c in cfg["robustness"]["cost_bps_grid"]:
        bc = BacktestConfig.from_config(cfg, fee_bps=c, slippage_bps=0.0)
        for n in main:
            m = run_backtest(strategies[n]["positions"], test["period_ret"], bc).metrics
            rows.append({"ticker": ticker, "strategy": n, "color_key": strategies[n]["color_key"], "cost_bps": c,
                         "sharpe": m["sharpe"], "cumulative_return": m["cumulative_return"], "costs_paid": m["costs_paid"]})
    out["cost_sensitivity"] = pd.DataFrame(rows)

    # ---- threshold sensitivity on TEST (post-hoc; thresholds were chosen on validation) -------
    rows = []
    for key in models_in_predictions(pred):
        if key == "majority_class":
            continue
        for tau in cfg["signals"]["threshold_grid"]:
            pos = positions_from_probabilities(test[proba_cols(key, K)].values, test.index, K - 1, 0, tau, cfg["signals"]["mode"])
            m = run_backtest(pos, test["period_ret"], bt_cfg).metrics
            rows.append({"ticker": ticker, "model_key": key, "model": display_name(key), "threshold": tau, "sharpe": m["sharpe"],
                         "exposure": m["exposure"], "cumulative_return": m["cumulative_return"]})
    out["threshold_sensitivity"] = pd.DataFrame(rows)

    # ---- regimes and calendar years ---------------------------------------------------------
    rows = []
    ann = bt_cfg.annualization
    for n in main:
        fr = backtests[n].frame
        groups = {
            "high_vol": test["high_vol"] == 1, "low_vol": test["high_vol"] == 0,
            "uptrend": test["uptrend"] == 1, "downtrend": test["uptrend"] == 0,
        }
        for year in sorted(set(fr.index.year)):
            groups[str(year)] = pd.Series(fr.index.year == year, index=fr.index)
        for g, mask in groups.items():
            r = fr.loc[mask.reindex(fr.index).fillna(False).values, "net_return"]
            if len(r) < MIN_ACTIVE_DAYS:
                continue
            invested = int((fr.loc[r.index, "position"] != 0).sum())
            rows.append({"ticker": ticker, "strategy": n, "regime": g, "n_days": len(r), "invested_days": invested,
                         # a Sharpe ratio over a handful of invested days is a sign indicator, not a statistic
                         "sharpe": sharpe_ratio(r, bt_cfg.risk_free_rate, ann) if invested >= MIN_ACTIVE_DAYS else np.nan,
                         "compounded_return": float(np.prod(1 + r.values) - 1),
                         "annualized_mean_return": float(r.mean() * ann),
                         "exposure": invested / len(r)})
    out["regimes"] = pd.DataFrame(rows)

    # ---- block bootstrap --------------------------------------------------------------------
    bcfg = cfg["robustness"]["bootstrap"]
    rows = []
    bh = backtests["Buy & Hold"].frame["net_return"].values
    for i, n in enumerate(main):
        r = backtests[n].frame["net_return"].values
        rf = bt_cfg.risk_free_rate
        ci = bootstrap_ci(r, sharpe_stat(ann, rf), bcfg["block_len"], bcfg["n_boot"], bcfg["ci"], seed + i)
        diff = bootstrap_ci(np.column_stack([r, bh]), sharpe_difference_stat(ann, rf), bcfg["block_len"], bcfg["n_boot"], bcfg["ci"], seed + i)
        rows.append({"ticker": ticker, "strategy": n, "statistic": "sharpe", **ci})
        if n != "Buy & Hold":
            rows.append({"ticker": ticker, "strategy": n, "statistic": "sharpe_minus_buy_and_hold", **diff})
    maj_correct = (test[proba_cols("majority_class", K)].values.argmax(1) == test["label"].values).astype(float)
    for i, key in enumerate(models_in_predictions(pred)):
        if key == "majority_class":
            continue
        correct = (test[proba_cols(key, K)].values.argmax(1) == test["label"].values).astype(float)
        d = bootstrap_ci(np.column_stack([correct, maj_correct]), accuracy_difference_stat(), bcfg["block_len"], bcfg["n_boot"], bcfg["ci"], seed + 100 + i)
        rows.append({"ticker": ticker, "strategy": display_name(key), "statistic": "accuracy_minus_majority_class", **d})
    out["bootstrap"] = pd.DataFrame(rows)

    # ---- volatility-targeted sizing ---------------------------------------------------------
    vt = cfg["robustness"].get("vol_target", {})
    if vt.get("enabled", False):
        rows = []
        for n in main:
            if strategies[n]["kind"] not in ("ml_confidence",) and n != "Buy & Hold":
                continue
            base = strategies[n]["positions"]
            sized = volatility_target_positions(base, test["realized_vol"], vt["annual_target"], ann, vt.get("max_leverage", 1.0))
            for label, pos in (("fixed (0/1)", base), (f"vol target {vt['annual_target']:.0%}", sized)):
                m = run_backtest(pos, test["period_ret"], bt_cfg).metrics
                rows.append({"ticker": ticker, "strategy": n, "sizing": label, **{k: m[k] for k in (
                    "cumulative_return", "annualized_return", "annualized_volatility", "sharpe", "sortino", "max_drawdown", "turnover_annual")}})
        out["vol_target"] = pd.DataFrame(rows)

    # ---- calibration of P(Up) ---------------------------------------------------------------
    curves, rows = {}, []
    is_up = (test["label"].values == K - 1).astype(float)
    for key in models_in_predictions(pred):
        if key == "majority_class":
            continue
        p_up = test[f"{key}__p{K - 1}"].values
        curves[display_name(key)] = reliability_curve(p_up, is_up, 10)
        rows.append({"ticker": ticker, "model": display_name(key), "ece_p_up": expected_calibration_error(p_up, is_up, 10),
                     "mean_p_up": float(p_up.mean()), "observed_up_rate": float(is_up.mean())})
    out["calibration"] = pd.DataFrame(rows)
    out["reliability_curves"] = curves
    return out


def _flat_cls(m: dict) -> dict:
    keep = ["n", "accuracy", "balanced_accuracy", "precision_macro", "recall_macro", "f1_macro", "precision_weighted",
            "recall_weighted", "f1_weighted", "log_loss", "brier"]
    out = {k: m[k] for k in keep}
    for cls_name, d in m["per_class"].items():
        for k in ("precision", "recall", "f1", "support"):
            out[f"{cls_name.lower()}_{k}"] = d[k]
    for cls_name, share in m["predicted_class_share"].items():
        out[f"pred_share_{cls_name.lower()}"] = share
    out["confusion_matrix"] = str(m["confusion_matrix"])
    return out


def evaluate_stage(cfg: dict, frames: dict, paths: OutputPaths, logger=LOGGER, train_results: dict | None = None) -> dict:
    """Evaluate saved holdout predictions for every ticker and write tables and figures."""
    names = class_names_for(cfg["target"]["scheme"])
    results = {}
    for ticker, prices in frames.items():
        pred = pd.read_csv(paths.predictions / f"{ticker}_holdout_predictions.csv", index_col=0, parse_dates=True)
        res = evaluate_holdout_ticker(ticker, prices, pred, cfg, logger)
        results[ticker] = res
        logger.info("%s evaluated: %d strategies", ticker, len(res["strategies"]))

    def cat(key):
        dfs = [r[key] for r in results.values() if key in r and r[key] is not None and len(r[key])]
        return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()

    cls, trading = cat("classification"), cat("trading")
    cls.to_csv(paths.metrics / "classification_metrics.csv", index=False)
    trading.to_csv(paths.metrics / "trading_metrics_test.csv", index=False)
    for key, fname in (("cost_sensitivity", "robustness_cost_sensitivity.csv"), ("threshold_sensitivity", "robustness_threshold_sensitivity_test.csv"),
                       ("regimes", "robustness_regimes.csv"), ("bootstrap", "bootstrap_confidence_intervals.csv"),
                       ("vol_target", "robustness_vol_target.csv"), ("calibration", "calibration.csv")):
        cat(key).to_csv(paths.metrics / fname, index=False)
    save_json({t: r["thresholds"] for t, r in results.items()}, paths.metrics / "selected_thresholds.json")
    for t, r in results.items():
        for key, tab in r["threshold_tables"].items():
            tab.to_csv(paths.tuning / f"{t}_{key}_threshold_selection_val.csv", index=False)
        eq = pd.DataFrame({n: bt.frame["equity"] for n, bt in r["backtests"].items()})
        eq.to_csv(paths.predictions / f"{t}_holdout_equity_curves.csv")
        pos = pd.DataFrame({n: s["positions"] for n, s in r["strategies"].items()})
        pos.to_csv(paths.predictions / f"{t}_holdout_positions.csv")
    comparison = comparison_table(cls, trading)
    comparison.to_csv(paths.metrics / "model_comparison_test.csv", index=False)
    make_holdout_figures(cfg, frames, results, cls, trading, paths, names)
    return results


def comparison_table(cls: pd.DataFrame, trading: pd.DataFrame) -> pd.DataFrame:
    """One row per (ticker, strategy): test classification metrics of the underlying model next to
    the trading metrics of its confidence-threshold strategy. N/A for rule-based baselines."""
    test_cls = cls[cls["split"] == "test"].set_index(["ticker", "model_key"])
    rows = []
    for _, r in trading[trading["kind"].isin(["baseline", "ml_confidence", "ml_argmax"])].iterrows():
        key = (r["ticker"], r["model_key"])
        c = test_cls.loc[key] if key in test_cls.index and r["model_key"] != "buy_and_hold" and r["model_key"] != "ma_crossover" else None
        rows.append({
            "ticker": r["ticker"], "strategy": r["strategy"], "kind": r["kind"],
            "accuracy": c["accuracy"] if c is not None else np.nan,
            "balanced_accuracy": c["balanced_accuracy"] if c is not None else np.nan,
            "precision_macro": c["precision_macro"] if c is not None else np.nan,
            "recall_macro": c["recall_macro"] if c is not None else np.nan,
            "f1_macro": c["f1_macro"] if c is not None else np.nan,
            "log_loss": c["log_loss"] if c is not None else np.nan,
            **{k: r[k] for k in ("threshold", "cumulative_return", "annualized_return", "annualized_volatility", "sharpe", "sortino",
                                 "max_drawdown", "win_rate", "exposure", "turnover_annual", "n_trades", "costs_paid")},
        })
    return pd.DataFrame(rows)


def make_holdout_figures(cfg, frames, results, cls, trading, paths: OutputPaths, names):
    synthetic = cfg.get("_synthetic", False)
    tag = " [SYNTHETIC DATA]" if synthetic else ""
    split_ranges = None
    ds = load_json(paths.metrics / "dataset_summary.json") if (paths.metrics / "dataset_summary.json").exists() else {}
    first = next(iter(ds.values()), None)
    if first:
        sp = first["split"]
        split_ranges = {k: (v["start"], v["end"]) for k, v in sp.items()}
    viz.plot_price_history({t: f["close"] for t, f in frames.items()}, split_ranges, paths.figures / "price_history.png", synthetic)
    dist_path = paths.metrics / "class_distribution.csv"
    if dist_path.exists():
        viz.plot_class_distribution(pd.read_csv(dist_path), paths.figures / "class_distribution.png")
    test_cls = cls[(cls["split"] == "test")]
    viz.plot_classification_comparison(test_cls, paths.figures / "classification_comparison.png")
    main = trading[trading["kind"].isin(["baseline", "ml_confidence"])]
    viz.plot_metric_bars(main, "sharpe", "Sharpe ratio (annualised)", f"Test-period Sharpe ratio by strategy{tag}", paths.figures / "sharpe_comparison.png")
    viz.plot_metric_bars(main, "sortino", "Sortino ratio (annualised)", f"Test-period Sortino ratio by strategy{tag}", paths.figures / "sortino_comparison.png")
    viz.plot_metric_bars(main, "max_drawdown", "Maximum drawdown (%)", f"Test-period maximum drawdown by strategy{tag}", paths.figures / "max_drawdown_comparison.png", pct=True)
    viz.plot_metric_bars(main, "cumulative_return", "Cumulative return (%)", f"Test-period cumulative return by strategy{tag}", paths.figures / "cumulative_return_comparison.png", pct=True)
    cost_path = paths.metrics / "robustness_cost_sensitivity.csv"
    if cost_path.exists() and cost_path.stat().st_size > 5:
        viz.plot_cost_sensitivity(pd.read_csv(cost_path), paths.figures / "cost_sensitivity.png")
    for t, r in results.items():
        prices = frames[t]
        test_start = r["backtests"]["Buy & Hold"].frame.index[0]
        viz.plot_technical_indicators(prices, t, paths.figures / f"{t}_technical_indicators.png", start=str(test_start.date()), end=None)
        viz.plot_confusion_matrices(r["confusion_matrices"], names, f"{t}: test-period confusion matrices{tag}", paths.figures / f"{t}_confusion_matrices.png")
        main_names = [n for n, s in r["strategies"].items() if s["kind"] in ("baseline", "ml_confidence")]
        eq = {n: (r["strategies"][n]["color_key"], r["backtests"][n].frame["equity"]) for n in main_names}
        dd = {n: (r["strategies"][n]["color_key"], r["backtests"][n].frame["drawdown"]) for n in main_names}
        viz.plot_equity_curves(eq, f"{t}: test-period equity curves, net of costs{tag}", paths.figures / f"{t}_equity_curves.png",
                               cfg["backtest"]["initial_capital"])
        viz.plot_drawdowns(dd, f"{t}: test-period drawdowns{tag}", paths.figures / f"{t}_drawdowns.png")
        if "reliability_curves" in r:
            keys = {display_name(k): k for k in ML_ORDER}
            viz.plot_reliability(r["reliability_curves"], f"{t}: reliability of P(Up), test period{tag}", paths.figures / f"{t}_reliability.png", keys)
        if "threshold_sensitivity" in r and len(r["threshold_sensitivity"]):
            viz.plot_threshold_sensitivity(r["threshold_sensitivity"], r["thresholds"], f"{t}: threshold sensitivity (test, post-hoc){tag}",
                                           paths.figures / f"{t}_threshold_sensitivity.png")
        if "regimes" in r and len(r["regimes"]):
            reg = r["regimes"].pivot(index="strategy", columns="regime", values="sharpe")
            order = [c for c in ["low_vol", "high_vol", "uptrend", "downtrend"] if c in reg.columns] + sorted(c for c in reg.columns if c.isdigit())
            reg = reg.reindex(index=main_names, columns=order)
            viz.plot_regime_heatmap(reg, f"{t}: Sharpe ratio by regime and year (test){tag}", paths.figures / f"{t}_regime_sharpe.png")
        for key in ("xgboost", "hist_gradient_boosting", "random_forest", "logistic_regression"):
            f = paths.importance / f"{t}_{key}_permutation.csv"
            if f.exists():
                viz.plot_feature_importance(pd.read_csv(f), f"{t}: permutation importance, {display_name(key)} (validation){tag}",
                                            paths.figures / f"{t}_{key}_permutation_importance.png")
        hist = paths.lstm / f"{t}_lstm_history.json"
        has_lstm = any(s["model_key"] == "lstm" for s in r["strategies"].values())
        if has_lstm and hist.exists():
            viz.plot_lstm_history(load_json(hist)["history"], f"{t}: LSTM training curve{tag}", paths.figures / f"{t}_lstm_training.png")


# ---------------------------------------------------------------------------------------------
# stage 3: walk-forward
# ---------------------------------------------------------------------------------------------


def walk_forward_ticker(ticker, prices, cfg, logger=LOGGER) -> dict:
    seed = cfg["experiment"]["seed"]
    set_global_seed(seed)
    scheme = cfg["target"]["scheme"]
    names = class_names_for(scheme)
    K = len(names)
    up, down = K - 1, 0
    conv = cfg["execution"]["convention"]
    wcfg = cfg["walk_forward"]
    data, fcols, _ = prepare_ticker_dataset(prices, cfg, conv)
    gap = label_end_offset(cfg["target"]["horizon"], conv)
    folds = walk_forward_folds(data, wcfg["first_test_year"], wcfg["last_test_year"], wcfg.get("val_years", 1), cfg["split"].get("purge", True))
    if not folds:
        raise ValueError(f"{ticker}: no walk-forward fold between {wcfg['first_test_year']} and {wcfg['last_test_year']} has enough "
                         f"history (data from {data.index[0].date()}); adjust walk_forward.first_test_year")
    X = data[fcols].to_numpy(dtype=float)
    y = data["label"].to_numpy(dtype=float)
    bt_cfg = BacktestConfig.from_config(cfg)
    fold_frames, fold_info = [], []
    for year, split in folds:
        t0 = time.perf_counter()
        P = {k: data.index.get_indexer(v) for k, v in split.as_dict().items()}
        Xtr, ytr = X[P["train"]], y[P["train"]].astype(int)
        yva = y[P["val"]].astype(int)
        probas, params = {}, {}
        for name in enabled_models(cfg, [m for m in wcfg["models"] if m != "lstm"]):
            actual = resolve_model_name(name)
            grid = wcfg["reduced_grids"][name]
            best, _ = time_series_grid_search(actual, Xtr, ytr, grid, cfg["models"][name].get("fixed"), K, wcfg.get("cv_splits", 3), gap, seed)
            model = fit_estimator(actual, best, Xtr, ytr, K, seed)
            probas[actual] = {"val": predict_proba_full(model, X[P["val"]], K), "test": predict_proba_full(model, X[P["test"]], K)}
            params[actual] = {k: best[k] for k in grid}
        lcfg = cfg["models"]["lstm"]
        if "lstm" in wcfg["models"] and TORCH_AVAILABLE and lcfg.get("enabled", False):
            # same protocol as the holdout: early stopping on a purged slice at the end of the fold's
            # training years; only class weighting is tuned (architecture fixed to the config defaults)
            fit_pos, es_pos = lstm_internal_split(P["train"], lcfg.get("early_stopping_fraction", 0.2), gap)
            scaler_rows = np.arange(0, P["train"].max() + 1)
            cands = []
            for cw in lcfg.get("grid", {}).get("class_weight", [lcfg.get("class_weight")]):
                m = LSTMSequenceClassifier(n_classes=K, **lstm_params(lcfg, {"class_weight": cw}, seed)).fit(X, y, fit_pos, es_pos, scaler_rows)
                cands.append((safe_log_loss(y[es_pos], m.predict_proba(X, es_pos), K), cw, m))
            _, cw, m = min(cands, key=lambda c: c[0])
            probas["lstm"] = {"val": m.predict_proba(X, P["val"]), "test": m.predict_proba(X, P["test"])}
            params["lstm"] = {"class_weight": cw, "best_epoch": m.best_epoch}
        if cfg["models"].get("ensemble", {}).get("enabled", False) and len(probas) >= 2:
            members, _ = select_ensemble(probas, yva, K)
            probas["ensemble"] = {s: np.mean([probas[k][s] for k in members], axis=0) for s in ("val", "test")}
            params["ensemble"] = {"members": members}
        maj = MajorityClassBaseline(K).fit(ytr)
        probas["majority_class"] = {"val": maj.predict_proba(len(P["val"])), "test": maj.predict_proba(len(P["test"]))}
        f = pd.DataFrame(index=split.test)
        f["year"] = year
        f["label"] = y[P["test"]].astype(int)
        f["period_ret"] = data.loc[split.test, "period_ret"].values
        val_ret = data.loc[split.val, "period_ret"]
        thresholds = {}
        for key, pr in probas.items():
            f[proba_cols(key, K)] = pr["test"]
            if key == "majority_class":
                continue
            tau, _ = select_threshold(pr["val"], split.val, val_ret, up, down, cfg["signals"], bt_cfg)
            thresholds[key] = tau
            f[f"{key}__pos"] = positions_from_probabilities(pr["test"], split.test, up, down, tau, cfg["signals"]["mode"]).values
        fold_frames.append(f)
        fold_info.append({"ticker": ticker, "test_year": year, "train_start": split.train[0].strftime("%Y-%m-%d"),
                          "train_end": split.train[-1].strftime("%Y-%m-%d"), "val_start": split.val[0].strftime("%Y-%m-%d"),
                          "val_end": split.val[-1].strftime("%Y-%m-%d"), "test_rows": len(split.test), "train_rows": len(split.train),
                          "params": params, "thresholds": thresholds, "seconds": round(time.perf_counter() - t0, 1)})
        logger.info("  %s walk-forward %d: train %d rows, thresholds %s (%.1fs)", ticker, year, len(split.train),
                    {k: v for k, v in thresholds.items()}, time.perf_counter() - t0)
    oos = pd.concat(fold_frames)
    oos.index.name = "date"
    return {"oos": oos, "folds": fold_info, "prices": prices}


def summarize_walk_forward(ticker, wf: dict, cfg) -> dict:
    names = class_names_for(cfg["target"]["scheme"])
    K = len(names)
    oos, prices = wf["oos"], wf["prices"]
    bt_cfg = BacktestConfig.from_config(cfg)
    keys = [k[: -len("__pos")] for k in oos.columns if k.endswith("__pos")]
    keys = sorted(keys, key=lambda k: ML_ORDER.index(k) if k in ML_ORDER else 99)
    allow_short = cfg["signals"]["mode"] == "long_short"
    strategies = {
        "Buy & Hold": ("buy_and_hold", buy_and_hold_positions(oos.index)),
        f"MA Crossover ({cfg['baselines']['ma_short']}/{cfg['baselines']['ma_long']})":
            ("ma_crossover", ma_crossover_positions(prices["close"], cfg["baselines"]["ma_short"], cfg["baselines"]["ma_long"], allow_short).reindex(oos.index)),
    }
    for k in keys:
        strategies[f"{display_name(k)} (conf.)"] = (k, oos[f"{k}__pos"])
    cls_rows, trade_rows, yearly_rows = [], [], []
    for k in keys + ["majority_class"]:
        proba = oos[proba_cols(k, K)].values
        m = classification_metrics(oos["label"].values, proba.argmax(1), names, proba)
        cls_rows.append({"ticker": ticker, "model_key": k, "model": display_name(k), "period": f"{oos['year'].min()}-{oos['year'].max()}",
                         **{x: m[x] for x in ("n", "accuracy", "balanced_accuracy", "f1_macro", "log_loss", "brier")}})
        for year, part in oos.groupby("year"):
            pp = part[proba_cols(k, K)].values
            mm = classification_metrics(part["label"].values, pp.argmax(1), names, pp)
            yearly_rows.append({"ticker": ticker, "year": year, "strategy": display_name(k) + (" (conf.)" if k != "majority_class" else ""),
                                "color_key": k, "accuracy": mm["accuracy"], "f1_macro": mm["f1_macro"]})
    backtests = {}
    for name, (ck, pos) in strategies.items():
        bt = run_backtest(pos, oos["period_ret"], bt_cfg)
        backtests[name] = (ck, bt)
        trade_rows.append({"ticker": ticker, "strategy": name, "color_key": ck, **bt.metrics})
        for year, part in bt.frame.groupby(bt.frame.index.year):
            r = part["net_return"]
            invested = int((part["position"] != 0).sum())
            row = {"ticker": ticker, "year": year, "strategy": name, "color_key": ck, "invested_days": invested,
                   "sharpe": sharpe_ratio(r, bt_cfg.risk_free_rate, bt_cfg.annualization) if invested >= MIN_ACTIVE_DAYS else np.nan,
                   "annual_return": float(np.prod(1 + r.values) - 1)}
            match = [y for y in yearly_rows if y["ticker"] == ticker and y["year"] == year and y["strategy"] == name]
            if match:
                match[0].update(row)
            else:
                yearly_rows.append(row)
    return {"classification": pd.DataFrame(cls_rows), "trading": pd.DataFrame(trade_rows), "yearly": pd.DataFrame(yearly_rows),
            "backtests": backtests}


def walk_forward_stage(cfg: dict, frames: dict, paths: OutputPaths, logger=LOGGER) -> dict:
    tag = " [SYNTHETIC DATA]" if cfg.get("_synthetic") else ""
    cls, trading, yearly, folds, out = [], [], [], [], {}
    for ticker, prices in frames.items():
        logger.info("Walk-forward: %s", ticker)
        wf = walk_forward_ticker(ticker, prices, cfg, logger)
        wf["oos"].to_csv(paths.predictions / f"{ticker}_walk_forward_predictions.csv")
        s = summarize_walk_forward(ticker, wf, cfg)
        out[ticker] = s
        cls.append(s["classification"])
        trading.append(s["trading"])
        yearly.append(s["yearly"])
        folds.extend(wf["folds"])
        eq = {n: (ck, bt.frame["equity"]) for n, (ck, bt) in s["backtests"].items()}
        viz.plot_equity_curves(eq, f"{ticker}: walk-forward out-of-sample equity ({wf['oos']['year'].min()}-{wf['oos']['year'].max()}){tag}",
                               paths.figures / f"{ticker}_walk_forward_equity.png", cfg["backtest"]["initial_capital"])
    cls, trading, yearly = pd.concat(cls, ignore_index=True), pd.concat(trading, ignore_index=True), pd.concat(yearly, ignore_index=True)
    cls.to_csv(paths.metrics / "walk_forward_classification.csv", index=False)
    trading.to_csv(paths.metrics / "walk_forward_trading.csv", index=False)
    yearly.to_csv(paths.metrics / "walk_forward_yearly.csv", index=False)
    save_json(folds, paths.metrics / "walk_forward_folds.json")
    # keep NaN years (too few invested days) so that the plotted lines break instead of bridging them
    y_tr = yearly[yearly["strategy"].isin(trading["strategy"])] if "sharpe" in yearly else pd.DataFrame()
    if len(y_tr):
        viz.plot_walk_forward(y_tr, "sharpe", "Sharpe ratio", f"Walk-forward: Sharpe ratio per test year (net of costs){tag}", paths.figures / "walk_forward_sharpe.png")
    y_acc = yearly.dropna(subset=["accuracy"]) if "accuracy" in yearly else pd.DataFrame()
    if len(y_acc):
        viz.plot_walk_forward(y_acc, "accuracy", "Accuracy", f"Walk-forward: classification accuracy per test year{tag}", paths.figures / "walk_forward_accuracy.png")
    return out


# ---------------------------------------------------------------------------------------------
# stage 4: sensitivity variants (execution timing, target formulation)
# ---------------------------------------------------------------------------------------------

SENSITIVITY_VARIANTS = [
    ("default: next_open, three_class", {"execution": {"convention": "next_open"}, "target": {"scheme": "three_class"}}),
    ("market-on-close, three_class", {"execution": {"convention": "close"}, "target": {"scheme": "three_class"}}),
    ("next_open, binary target", {"execution": {"convention": "next_open"}, "target": {"scheme": "binary"}}),
]


def sensitivity_stage(cfg: dict, frames: dict, paths: OutputPaths, logger=LOGGER) -> pd.DataFrame:
    """Re-run the holdout under alternative execution / target definitions.

    For a like-for-like comparison every variant (including the default) uses the reduced
    walk-forward grids, tabular models + ensemble only (no LSTM) and no extra analyses.
    """
    rows = []
    for label, override in SENSITIVITY_VARIANTS:
        vcfg = deep_update(cfg, override)
        for name, grid in cfg["walk_forward"]["reduced_grids"].items():
            vcfg["models"][name]["grid"] = dict(grid)  # replace (not merge) the search grid
        for ticker, prices in frames.items():
            logger.info("Sensitivity [%s] %s", label, ticker)
            tr = train_holdout_ticker(ticker, prices, vcfg, None, logger, with_lstm=False, with_analysis=False)
            ev = evaluate_holdout_ticker(ticker, prices, tr["predictions"], vcfg, logger, with_analysis=False)
            c = ev["classification"]
            c = c[c["split"] == "test"].set_index("model_key")
            for _, r in ev["trading"].iterrows():
                if r["kind"] not in ("baseline", "ml_confidence"):
                    continue
                has_cls = r["kind"] == "ml_confidence" or r["model_key"] == "persistence"
                rows.append({"variant": label, "convention": vcfg["execution"]["convention"], "scheme": vcfg["target"]["scheme"],
                             "ticker": ticker, "strategy": r["strategy"], "kind": r["kind"],
                             "accuracy": c.loc[r["model_key"], "accuracy"] if has_cls else np.nan,
                             "balanced_accuracy": c.loc[r["model_key"], "balanced_accuracy"] if has_cls else np.nan,
                             "f1_macro": c.loc[r["model_key"], "f1_macro"] if has_cls else np.nan,
                             "threshold": r["threshold"], "sharpe": r["sharpe"], "cumulative_return": r["cumulative_return"],
                             "max_drawdown": r["max_drawdown"], "exposure": r["exposure"]})
            rows.append({"variant": label, "convention": vcfg["execution"]["convention"], "scheme": vcfg["target"]["scheme"],
                         "ticker": ticker, "strategy": "Majority class", "kind": "classification_baseline",
                         "accuracy": c.loc["majority_class", "accuracy"], "balanced_accuracy": c.loc["majority_class", "balanced_accuracy"],
                         "f1_macro": c.loc["majority_class", "f1_macro"]})
    df = pd.DataFrame(rows)
    df.to_csv(paths.metrics / "robustness_variants.csv", index=False)
    return df


# ---------------------------------------------------------------------------------------------
# top level
# ---------------------------------------------------------------------------------------------


def environment_info() -> dict:
    import sklearn

    return {
        "project_version": __version__,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scikit_learn": sklearn.__version__,
        "xgboost": XGBOOST_VERSION if XGBOOST_AVAILABLE else "NOT INSTALLED (HistGradientBoosting fallback used)",
        "torch": TORCH_VERSION if TORCH_AVAILABLE else "NOT INSTALLED (LSTM skipped)",
    }


def load_data(cfg: dict, logger=LOGGER):
    """Load the universe. If any ticker fell back to synthetic data, the whole run is relabelled
    SYNTHETIC and redirected to a separate results directory (never mixed with real results).
    Call this BEFORE creating :class:`OutputPaths`."""
    frames, manifest = load_universe(cfg, force_synthetic=cfg.get("_synthetic", False))
    if any(m["is_synthetic"] for m in manifest):
        logger.warning("SYNTHETIC DATA IN USE - results are NOT market results")
        if not cfg.get("_synthetic", False):
            cfg["_synthetic"] = True
            cfg["_is_default_full_run"] = False
            cfg["experiment"]["results_dir"] = cfg["experiment"]["results_dir"] + "_synthetic_fallback"
            logger.warning("Outputs redirected to %s", cfg["experiment"]["results_dir"])
    return frames, manifest


def reset_outputs(results_dir: str | Path, logger=LOGGER) -> None:
    """Delete generated artefacts of a previous run in ``results_dir`` so that a new run can never
    be mixed with stale files (only the sub-folders this project writes are touched)."""
    import shutil

    root = resolve_path(results_dir)
    for sub in ("metrics", "figures", "predictions", "reports", "models"):
        target = root / sub
        if target.is_dir():
            shutil.rmtree(target)
    logger.info("Cleared previous outputs in %s", root.name)


def run_experiment(cfg: dict, skip_walk_forward: bool = False, logger=LOGGER) -> dict:
    t0 = time.perf_counter()
    set_global_seed(cfg["experiment"]["seed"])
    frames, manifest = load_data(cfg, logger)
    reset_outputs(cfg["experiment"]["results_dir"], logger)
    paths = OutputPaths(cfg["experiment"]["results_dir"])
    save_json(manifest, paths.metrics / "data_manifest.json")
    stage_times = {}

    s = time.perf_counter()
    train_stage(cfg, frames, paths, logger)
    stage_times["train"] = time.perf_counter() - s
    s = time.perf_counter()
    evaluate_stage(cfg, frames, paths, logger)
    stage_times["evaluate"] = time.perf_counter() - s
    if cfg["walk_forward"].get("enabled", True) and not skip_walk_forward and cfg["walk_forward"].get("models"):
        s = time.perf_counter()
        walk_forward_stage(cfg, frames, paths, logger)
        stage_times["walk_forward"] = time.perf_counter() - s
    if cfg["robustness"].get("sensitivity_variants", False):
        s = time.perf_counter()
        sensitivity_stage(cfg, frames, paths, logger)
        stage_times["sensitivity_variants"] = time.perf_counter() - s

    summary = {
        "experiment": cfg["experiment"]["name"],
        "synthetic_data": bool(cfg.get("_synthetic", False)),
        # True only for `python -m scripts.run_experiment` with the unmodified config.yaml
        "is_default_full_run": bool(cfg.get("_is_default_full_run", False)) and not skip_walk_forward,
        "tickers": list(frames),
        "environment": environment_info(),
        "stage_seconds": {k: round(v, 1) for k, v in stage_times.items()},
        "total_seconds": round(time.perf_counter() - t0, 1),
        "config": {k: v for k, v in cfg.items() if not k.startswith("_")},
    }
    save_json(summary, paths.metrics / "run_summary.json")
    from .reporting import generate_report

    report = generate_report(paths.root, cfg)
    logger.info("Report written to %s", report)
    logger.info("Total runtime %.1f s", summary["total_seconds"])
    return summary
