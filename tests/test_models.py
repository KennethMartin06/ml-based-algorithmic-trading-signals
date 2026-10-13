"""Unit tests for src/models.py (tabular models + purged tuning) and src/lstm_model.py.

All data here is small and synthetic: the tests check code behaviour (shapes, invariants,
selection logic, leakage controls), not predictive skill.
"""

from __future__ import annotations

import json
import math

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_sample_weight

from src import models
from src.lstm_model import TORCH_AVAILABLE, LSTMSequenceClassifier
from src.models import (
    DISPLAY_NAMES,
    balanced_sample_weight,
    build_estimator,
    display_name,
    expand_grid,
    fit_estimator,
    predict_proba_full,
    resolve_model_name,
    safe_log_loss,
    time_series_grid_search,
    tree_importance,
)

requires_xgboost = pytest.mark.skipif(not models.XGBOOST_AVAILABLE, reason="xgboost not installed")
requires_torch = pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not installed")


def make_problem(rng: np.random.Generator, n: int = 300, d: int = 4, n_classes: int = 3, noise: float = 0.7):
    """Synthetic K-class problem whose label is a noisy, monotone function of feature 0."""
    X = rng.normal(size=(n, d))
    score = 2.0 * X[:, 0] + noise * rng.normal(size=n)
    cuts = np.quantile(score, np.linspace(0, 1, n_classes + 1)[1:-1])
    y = np.digitize(score, cuts).astype(int)
    return X, y


# ---------------------------------------------------------------------------------------------
# grid expansion and class weights
# ---------------------------------------------------------------------------------------------


def test_expand_grid_is_cartesian_product():
    """Every combination appears exactly once, keys are preserved, count = product of list sizes."""
    grid = {"a": [1, 2], "b": ["x", "y", "z"], "c": [None]}
    combos = expand_grid(grid)
    assert len(combos) == 2 * 3 * 1
    assert all(set(c) == {"a", "b", "c"} for c in combos)
    as_tuples = {(c["a"], c["b"], c["c"]) for c in combos}
    assert as_tuples == {(a, b, None) for a in (1, 2) for b in ("x", "y", "z")}
    # itertools.product order: the last key varies fastest
    assert combos[0] == {"a": 1, "b": "x", "c": None}
    assert combos[1] == {"a": 1, "b": "y", "c": None}


def test_expand_grid_empty_grid_gives_one_empty_candidate():
    """An empty grid means 'no hyperparameters to search' -> exactly one (empty) candidate."""
    assert expand_grid({}) == [{}]


@pytest.mark.parametrize("n_classes, counts", [(3, (50, 7, 23)), (3, (10, 10, 10)), (2, (90, 30))])
def test_balanced_sample_weight_matches_sklearn(rng, n_classes, counts):
    """w_i = n / (K * n_{y_i}) equals sklearn's compute_sample_weight('balanced') when every class occurs."""
    y = np.concatenate([np.full(c, k) for k, c in enumerate(counts)])
    y = rng.permutation(y)
    ours = balanced_sample_weight(y, n_classes)
    ref = compute_sample_weight("balanced", y)
    np.testing.assert_allclose(ours, ref, rtol=1e-12)
    # each class receives the same total weight n/K, and the mean weight is 1
    totals = [ours[y == k].sum() for k in range(n_classes)]
    np.testing.assert_allclose(totals, len(y) / n_classes)
    assert ours.mean() == pytest.approx(1.0)


# ---------------------------------------------------------------------------------------------
# probabilities with absent classes
# ---------------------------------------------------------------------------------------------


@pytest.mark.parametrize("name", ["logistic_regression", "random_forest", "hist_gradient_boosting"])
def test_predict_proba_full_fills_absent_class_with_zero(rng, name):
    """A class missing from the training labels gets a zero column; the other columns are the
    model's own probabilities in the right positions and every row sums to one."""
    X, y = make_problem(rng, n=240)
    y = np.where(y == 1, 2, y)  # the middle (Neutral) class never occurs in training
    params = {"n_estimators": 20} if name != "logistic_regression" else {}
    model = fit_estimator(name, params, X, y, n_classes=3)
    full = predict_proba_full(model, X[:50], n_classes=3)

    assert full.shape == (50, 3)
    np.testing.assert_array_equal(full[:, 1], 0.0)
    np.testing.assert_allclose(full.sum(axis=1), 1.0, atol=1e-12)
    raw = model.predict_proba(X[:50])  # columns for classes [0, 2]
    np.testing.assert_allclose(full[:, [0, 2]], raw)


def test_predict_proba_full_column_order_follows_class_labels(rng):
    """Column k always holds P(class k), also when the estimator saw only classes {1, 2}."""
    X, y = make_problem(rng, n=200)
    y = np.where(y == 0, 1, y)
    model = fit_estimator("logistic_regression", {}, X, y, n_classes=3)
    full = predict_proba_full(model, X, 3)
    np.testing.assert_array_equal(full[:, 0], 0.0)
    pred_class = full.argmax(axis=1)
    np.testing.assert_array_equal(pred_class, model.predict(X))


# ---------------------------------------------------------------------------------------------
# estimator construction
# ---------------------------------------------------------------------------------------------


def test_build_estimator_logistic_regression_is_scaler_pipeline():
    """LR is wrapped as Pipeline(StandardScaler -> LogisticRegression) so CV refits the scaler per fold."""
    params = {"C": 0.25, "class_weight": "balanced"}
    est = build_estimator("logistic_regression", params, n_classes=3, seed=7)
    assert isinstance(est, Pipeline)
    assert [name for name, _ in est.steps] == ["scaler", "clf"]
    assert isinstance(est.named_steps["scaler"], StandardScaler)
    clf = est.named_steps["clf"]
    assert isinstance(clf, LogisticRegression)
    assert clf.C == 0.25 and clf.class_weight == "balanced" and clf.random_state == 7
    assert params == {"C": 0.25, "class_weight": "balanced"}, "caller's params dict must not be mutated"


def test_build_estimator_random_forest_receives_params():
    """RF gets the grid/fixed params, class_weight and the seed."""
    est = build_estimator("random_forest", {"n_estimators": 15, "max_depth": 3, "class_weight": None}, 3, seed=5)
    assert isinstance(est, RandomForestClassifier)
    assert (est.n_estimators, est.max_depth, est.class_weight, est.random_state) == (15, 3, None, 5)


@requires_xgboost
@pytest.mark.parametrize("n_classes, objective", [(3, "multi:softprob"), (2, "binary:logistic")])
def test_build_estimator_xgboost_objective_and_no_class_weight_kwarg(n_classes, objective):
    """XGBoost uses softprob (3 classes) / logistic (binary); 'class_weight' is consumed, not forwarded."""
    import xgboost as xgb

    est = build_estimator("xgboost", {"max_depth": 2, "n_estimators": 10, "class_weight": "balanced"}, n_classes, seed=3)
    assert isinstance(est, xgb.XGBClassifier)
    p = est.get_params()
    assert p["objective"] == objective
    assert p["tree_method"] == "hist"
    assert p["random_state"] == 3 and p["max_depth"] == 2 and p["n_estimators"] == 10
    assert "class_weight" not in p


def test_build_estimator_hist_gradient_boosting_maps_xgboost_params():
    """The fallback maps n_estimators -> max_iter and drops XGBoost-only parameters."""
    params = {"n_estimators": 40, "max_depth": 2, "learning_rate": 0.1, "subsample": 0.8, "colsample_bytree": 0.8,
              "min_child_weight": 5, "reg_lambda": 1.0, "class_weight": "balanced"}
    est = build_estimator("hist_gradient_boosting", params, 3, seed=1)
    assert isinstance(est, HistGradientBoostingClassifier)
    assert (est.max_iter, est.max_depth, est.learning_rate, est.class_weight) == (40, 2, 0.1, "balanced")


def test_build_estimator_unknown_name_raises():
    with pytest.raises(ValueError):
        build_estimator("svm", {}, 3)


@requires_xgboost
def test_fit_estimator_xgboost_balanced_passes_sample_weights(rng, monkeypatch):
    """class_weight='balanced' for XGBoost is implemented as per-sample weights n/(K n_y);
    with class_weight=None no weights are passed."""
    import xgboost as xgb

    calls = []
    original_fit = xgb.XGBClassifier.fit

    def spy_fit(self, X, y, **kwargs):
        calls.append(kwargs.get("sample_weight"))
        return original_fit(self, X, y, **kwargs)

    monkeypatch.setattr(xgb.XGBClassifier, "fit", spy_fit)
    X, y = make_problem(rng, n=150)
    fit_estimator("xgboost", {"n_estimators": 5, "class_weight": "balanced"}, X, y, 3)
    fit_estimator("xgboost", {"n_estimators": 5, "class_weight": None}, X, y, 3)

    assert len(calls) == 2
    np.testing.assert_allclose(calls[0], compute_sample_weight("balanced", y))
    assert calls[1] is None


@requires_xgboost
def test_fit_estimator_xgboost_balanced_shifts_probability_to_minority_classes(rng):
    """Behavioural check: on imbalanced, uninformative labels the balanced XGBoost model predicts
    the minority classes with clearly higher average probability than the unweighted one."""
    n = 600
    X = rng.normal(size=(n, 3))
    y = rng.choice(3, size=n, p=[0.8, 0.1, 0.1])
    params = {"n_estimators": 30, "max_depth": 2, "learning_rate": 0.3}
    plain = predict_proba_full(fit_estimator("xgboost", {**params, "class_weight": None}, X, y, 3), X, 3).mean(axis=0)
    balanced = predict_proba_full(fit_estimator("xgboost", {**params, "class_weight": "balanced"}, X, y, 3), X, 3).mean(axis=0)
    assert balanced[1] > plain[1] + 0.1
    assert balanced[2] > plain[2] + 0.1
    assert balanced[0] < plain[0] - 0.2


# ---------------------------------------------------------------------------------------------
# purged time-series grid search
# ---------------------------------------------------------------------------------------------


def _manual_cv_log_loss(X, y, C, class_weight, n_splits, gap, tol):
    """Independent re-implementation of the CV score for one LR parameter set."""
    losses = []
    for tr, va in TimeSeriesSplit(n_splits=n_splits, gap=gap).split(X):
        model = Pipeline([("scaler", StandardScaler()),
                          ("clf", LogisticRegression(C=C, class_weight=class_weight, max_iter=5000, tol=tol, random_state=0))])
        model.fit(X[tr], y[tr])
        losses.append(log_loss(y[va], model.predict_proba(X[va]), labels=[0, 1, 2]))
    return float(np.mean(losses))


def test_time_series_grid_search_selects_lowest_cv_log_loss(rng):
    """The returned params are the grid point with the lowest mean purged-CV log loss; the table
    is sorted ascending, has one row per grid point and its scores match an independent
    TimeSeriesSplit(gap) computation (so the gap/purge is really applied)."""
    X, y = make_problem(rng, n=360, noise=1.0)
    grid = {"C": [1e-4, 1.0, 1e-2], "class_weight": [None, "balanced"]}
    fixed = {"tol": 1e-6}
    n_splits, gap = 3, 2
    best, table = time_series_grid_search("logistic_regression", X, y, grid, fixed, n_classes=3,
                                          n_splits=n_splits, gap=gap, seed=0)

    assert len(table) == len(expand_grid(grid))
    assert table["cv_log_loss_mean"].is_monotonic_increasing
    assert {"cv_log_loss_mean", "cv_log_loss_std", "cv_macro_f1_mean", "fit_seconds"} <= set(table.columns)

    expected = {}
    for p in expand_grid(grid):
        expected[(p["C"], str(p["class_weight"]))] = _manual_cv_log_loss(X, y, p["C"], p["class_weight"], n_splits, gap, fixed["tol"])
    for _, row in table.iterrows():
        assert row["cv_log_loss_mean"] == pytest.approx(expected[(row["C"], row["class_weight"])], rel=1e-7)

    best_key = min(expected, key=expected.get)
    assert (best["C"], str(best["class_weight"])) == best_key
    assert best["class_weight"] is None or best["class_weight"] == "balanced"  # real None, not the string "None"
    assert best["tol"] == fixed["tol"], "fixed params are merged into the best params"
    assert (table.loc[0, "C"], table.loc[0, "class_weight"]) == best_key
    assert best["C"] != 1e-4, "a heavily regularised C=1e-4 model (near-uniform probabilities) cannot win on an informative problem"

    # sensitivity of the check itself: ignoring the purge gap would give different scores
    no_gap = _manual_cv_log_loss(X, y, best["C"], best["class_weight"], n_splits, 0, fixed["tol"])
    assert no_gap != pytest.approx(table.loc[0, "cv_log_loss_mean"], rel=1e-7)


def test_time_series_grid_search_trains_only_on_purged_past(rng, monkeypatch):
    """Every CV fit sees a chronological prefix of the training data that ends `gap` rows
    before its validation fold (TimeSeriesSplit with gap = label horizon)."""
    X, y = make_problem(rng, n=200)
    seen = []
    original = models.fit_estimator

    def spy(name, params, Xf, yf, n_classes, seed=42):
        seen.append(Xf)
        return original(name, params, Xf, yf, n_classes, seed)

    monkeypatch.setattr(models, "fit_estimator", spy)
    gap, n_splits = 3, 4
    time_series_grid_search("logistic_regression", X, y, {"C": [1.0]}, None, 3, n_splits=n_splits, gap=gap)
    folds = list(TimeSeriesSplit(n_splits=n_splits, gap=gap).split(X))
    assert len(seen) == n_splits
    for Xf, (_, va) in zip(seen, folds):
        np.testing.assert_array_equal(Xf, X[: len(Xf)])  # a prefix: no future rows, no shuffling
        assert len(Xf) == va.min() - gap  # purge gap between last training row and validation fold


# ---------------------------------------------------------------------------------------------
# XGBoost fallback naming
# ---------------------------------------------------------------------------------------------


def test_resolve_model_name_uses_documented_fallback(monkeypatch):
    """Without xgboost, 'xgboost' resolves to HistGradientBoosting and is labelled as a fallback,
    never as XGBoost; building an XGBoost estimator then fails loudly."""
    monkeypatch.setattr(models, "XGBOOST_AVAILABLE", False)
    assert resolve_model_name("xgboost") == "hist_gradient_boosting"
    for other in ("logistic_regression", "random_forest", "lstm"):
        assert resolve_model_name(other) == other
    label = display_name(resolve_model_name("xgboost"))
    assert "fallback" in label.lower() and "HistGradientBoosting" in label
    assert label != DISPLAY_NAMES["xgboost"]
    with pytest.raises(ImportError):
        build_estimator("xgboost", {}, 3)


def test_resolve_model_name_keeps_xgboost_when_available(monkeypatch):
    monkeypatch.setattr(models, "XGBOOST_AVAILABLE", True)
    assert resolve_model_name("xgboost") == "xgboost"
    assert display_name("xgboost") == "XGBoost"


# ---------------------------------------------------------------------------------------------
# safe log loss and importances
# ---------------------------------------------------------------------------------------------


def test_safe_log_loss_renormalises_rows(rng):
    """Probability rows that do not sum to one (sub-normalised, or float32 rounding) are
    renormalised before scoring, giving the log loss of the normalised distribution."""
    y = rng.integers(0, 3, size=40)
    p = rng.dirichlet([2.0, 2.0, 2.0], size=40)
    reference = -np.mean(np.log(p[np.arange(40), y]))

    shrunk = p * rng.uniform(0.5, 0.95, size=(40, 1))  # every row sums to something < 1
    assert np.all(shrunk.sum(axis=1) < 0.96)
    assert safe_log_loss(y, shrunk, 3) == pytest.approx(reference, rel=1e-10)
    assert safe_log_loss(y, p.astype(np.float32), 3) == pytest.approx(reference, rel=1e-6)
    # sklearn's log_loss on the same normalised probabilities agrees
    assert safe_log_loss(y, shrunk, 3) == pytest.approx(log_loss(y, p, labels=[0, 1, 2]), rel=1e-10)


def test_safe_log_loss_zero_probability_and_missing_class_in_y(rng):
    """A zero probability on the true class is clipped (finite loss) and y_true need not contain every class."""
    y = np.array([0, 2, 2, 0])  # class 1 absent from y_true
    p = np.array([[0.7, 0.2, 0.1], [0.0, 0.5, 0.5], [0.1, 0.1, 0.8], [0.0, 0.0, 1.0]])
    loss = safe_log_loss(y, p, 3)
    assert math.isfinite(loss)
    clipped = np.clip(p, 1e-12, 1.0)
    clipped = clipped / clipped.sum(axis=1, keepdims=True)
    assert loss == pytest.approx(-np.mean(np.log(clipped[np.arange(4), y])), rel=1e-10)


@pytest.mark.parametrize("name", ["random_forest", "logistic_regression",
                                  pytest.param("xgboost", marks=requires_xgboost)])
def test_tree_importance_is_normalised_and_finds_signal(rng, name):
    """Importances are non-negative, indexed by feature name, sum to 1 and rank the informative feature first."""
    X, y = make_problem(rng, n=300, noise=0.3)
    params = {"n_estimators": 30, "max_depth": 3} if name != "logistic_regression" else {}
    model = fit_estimator(name, params, X, y, 3)
    names = [f"f{i}" for i in range(X.shape[1])]
    imp = tree_importance(model, names)
    assert isinstance(imp, pd.Series)
    assert list(imp.index) == names
    assert (imp >= 0).all()
    assert imp.sum() == pytest.approx(1.0)
    assert imp.idxmax() == "f0"


def test_tree_importance_none_for_model_without_importances(rng):
    """The HistGradientBoosting fallback exposes no built-in importance -> None (permutation importance is used instead)."""
    X, y = make_problem(rng, n=120)
    model = fit_estimator("hist_gradient_boosting", {"n_estimators": 5}, X, y, 3)
    assert tree_importance(model, [f"f{i}" for i in range(X.shape[1])]) is None


# ---------------------------------------------------------------------------------------------
# LSTM
# ---------------------------------------------------------------------------------------------

LSTM_KW = dict(n_classes=3, seq_len=5, hidden_size=16, num_layers=1, dropout=0.0, learning_rate=0.02, batch_size=16,
               max_epochs=10, patience=10, class_weight=None, seed=0)
TRAIN_POS, VAL_POS, TEST_POS = np.arange(0, 100), np.arange(100, 130), np.arange(130, 160)


@pytest.fixture(scope="module")
def lstm_data():
    """160 chronological rows, 4 features, 3-class labels (pure noise -> the net overfits quickly)."""
    g = np.random.default_rng(0)
    X = g.normal(size=(160, 4))
    y = g.integers(0, 3, size=160).astype(float)
    return X, y


@pytest.fixture(scope="module")
def fitted_lstm(lstm_data):
    X, y = lstm_data
    return LSTMSequenceClassifier(**LSTM_KW).fit(X, y, TRAIN_POS, VAL_POS)


@requires_torch
def test_lstm_predict_proba_shape_and_rows_sum_to_one(lstm_data, fitted_lstm):
    """One probability row per requested decision row, K columns, each row a distribution."""
    X, _ = lstm_data
    p = fitted_lstm.predict_proba(X, TEST_POS)
    assert p.shape == (len(TEST_POS), 3)
    assert np.all(p >= 0) and np.all(p <= 1)
    np.testing.assert_allclose(p.sum(axis=1), 1.0, atol=1e-5)


@requires_torch
def test_lstm_is_deterministic_for_a_fixed_seed(lstm_data, fitted_lstm):
    """Re-fitting with the same seed reproduces the probabilities and history exactly; another seed differs."""
    X, y = lstm_data
    again = LSTMSequenceClassifier(**LSTM_KW).fit(X, y, TRAIN_POS, VAL_POS)
    np.testing.assert_allclose(again.predict_proba(X, TEST_POS), fitted_lstm.predict_proba(X, TEST_POS), rtol=0, atol=1e-7)
    assert [h["val_loss"] for h in again.history] == pytest.approx([h["val_loss"] for h in fitted_lstm.history], abs=1e-7)
    other = LSTMSequenceClassifier(**{**LSTM_KW, "seed": 1}).fit(X, y, TRAIN_POS, VAL_POS)
    assert not np.allclose(other.predict_proba(X, TEST_POS), fitted_lstm.predict_proba(X, TEST_POS))


@requires_torch
def test_lstm_history_and_best_epoch_weights_restored(lstm_data, fitted_lstm):
    """History has one record per epoch; best_val_loss is the minimum validation loss, best_epoch
    is where it occurred, and the returned network carries THOSE weights (not the last epoch's)."""
    X, y = lstm_data
    hist = fitted_lstm.history
    assert len(hist) == LSTM_KW["max_epochs"]  # patience == max_epochs -> no early stop
    assert [h["epoch"] for h in hist] == list(range(1, len(hist) + 1))
    assert {"train_loss", "val_loss", "val_accuracy"} <= set(hist[0])
    val_losses = [h["val_loss"] for h in hist]
    assert fitted_lstm.best_val_loss == pytest.approx(min(val_losses), abs=1e-12)
    assert fitted_lstm.best_epoch == int(np.argmin(val_losses)) + 1
    # precondition that makes the restore check meaningful: training continued past the best epoch
    assert fitted_lstm.best_epoch < len(hist)
    assert val_losses[-1] > fitted_lstm.best_val_loss + 1e-3

    # class_weight=None -> the monitored loss is plain cross-entropy, i.e. the validation log loss
    restored = safe_log_loss(y[VAL_POS].astype(int), fitted_lstm.predict_proba(X, VAL_POS), 3)
    assert restored == pytest.approx(fitted_lstm.best_val_loss, abs=1e-5)
    assert restored != pytest.approx(val_losses[-1], abs=1e-4)


@requires_torch
def test_lstm_early_stopping_respects_patience(lstm_data):
    """Training stops exactly `patience` epochs after the last improvement (or at max_epochs)."""
    X, y = lstm_data
    m = LSTMSequenceClassifier(**{**LSTM_KW, "max_epochs": 30, "patience": 2}).fit(X, y, TRAIN_POS, VAL_POS)
    assert len(m.history) < 30
    assert len(m.history) == m.best_epoch + 2


@requires_torch
def test_lstm_scaler_fitted_on_training_rows_only(lstm_data, fitted_lstm):
    """The StandardScaler sees rows 0..last training decision row only (no validation/test rows)."""
    X, _ = lstm_data
    upto = X[: TRAIN_POS.max() + 1]
    np.testing.assert_allclose(fitted_lstm.scaler.mean_, upto.mean(axis=0))
    np.testing.assert_allclose(fitted_lstm.scaler.scale_, upto.std(axis=0))


@requires_torch
def test_lstm_prediction_uses_only_past_rows(lstm_data, fitted_lstm):
    """The window for decision row p is rows p-L+1..p: changing rows after p cannot change P(row p)."""
    X, _ = lstm_data
    p = 140
    X_future_changed = X.copy()
    X_future_changed[p + 1:] = 100.0
    rows = np.arange(130, p + 1)
    np.testing.assert_array_equal(fitted_lstm.predict_proba(X_future_changed, rows), fitted_lstm.predict_proba(X, rows))
    with pytest.raises(ValueError):
        fitted_lstm.predict_proba(X, np.array([LSTM_KW["seq_len"] - 2]))  # not enough history


@requires_torch
def test_lstm_save_writes_weights_and_history(tmp_path, fitted_lstm):
    """save() writes the state dict and a JSON history with params and best epoch."""
    fitted_lstm.save(tmp_path, "unit")
    assert (tmp_path / "unit.pt").exists()
    meta = json.loads((tmp_path / "unit_history.json").read_text(encoding="utf-8"))
    assert meta["best_epoch"] == fitted_lstm.best_epoch
    assert meta["params"]["seq_len"] == LSTM_KW["seq_len"]
    assert len(meta["history"]) == len(fitted_lstm.history)


@requires_torch
def test_lstm_unfitted_predict_raises(lstm_data):
    X, _ = lstm_data
    with pytest.raises(RuntimeError):
        LSTMSequenceClassifier(n_classes=3).predict_proba(X, TEST_POS)
