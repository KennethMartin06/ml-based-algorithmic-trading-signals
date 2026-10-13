"""Unit tests for classification metrics, calibration, block bootstrap, feature analysis and
regime labelling (src/evaluation.py) plus the majority-class benchmark (src/baselines.py).

Reference values come from scikit-learn or from hand calculations, never from the function
under test itself.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_recall_fscore_support,
)

from src.baselines import MajorityClassBaseline
from src.evaluation import (
    accuracy_difference_stat,
    bootstrap_ci,
    circular_block_bootstrap_indices,
    classification_metrics,
    expected_calibration_error,
    multiclass_brier,
    mutual_information_frame,
    regime_labels,
    reliability_curve,
    sharpe_stat,
)
from src.features import compute_features

CLASSES = ["Down", "Neutral", "Up"]


def _random_proba(rng, n: int, k: int = 3) -> np.ndarray:
    p = rng.dirichlet(np.ones(k), size=n)
    return p


# ---------------------------------------------------------------------------------------------
# classification metrics
# ---------------------------------------------------------------------------------------------


def test_classification_metrics_match_sklearn(rng):
    """Accuracy, balanced accuracy, macro/weighted F1, confusion matrix and per-class scores agree
    with scikit-learn on a random 3-class problem."""
    y_true = rng.integers(0, 3, 400)
    y_pred = np.where(rng.random(400) < 0.5, y_true, rng.integers(0, 3, 400))
    m = classification_metrics(y_true, y_pred, CLASSES)

    labels = [0, 1, 2]
    assert m["n"] == 400
    assert m["accuracy"] == pytest.approx(accuracy_score(y_true, y_pred))
    assert m["balanced_accuracy"] == pytest.approx(balanced_accuracy_score(y_true, y_pred))
    assert m["f1_macro"] == pytest.approx(f1_score(y_true, y_pred, average="macro", labels=labels))
    assert m["f1_weighted"] == pytest.approx(f1_score(y_true, y_pred, average="weighted", labels=labels))
    assert m["confusion_matrix"] == confusion_matrix(y_true, y_pred, labels=labels).tolist()

    p, r, f, s = precision_recall_fscore_support(y_true, y_pred, labels=labels, zero_division=0)
    for i, name in enumerate(CLASSES):
        pc = m["per_class"][name]
        assert pc["precision"] == pytest.approx(p[i])
        assert pc["recall"] == pytest.approx(r[i])
        assert pc["f1"] == pytest.approx(f[i])
        assert pc["support"] == s[i]
    assert sum(m["predicted_class_share"].values()) == pytest.approx(1.0)
    assert m["predicted_class_share"]["Up"] == pytest.approx(np.mean(y_pred == 2))


def test_classification_metrics_keep_all_classes_when_one_is_absent():
    """A class that never occurs (here Neutral) still gets a full 3x3 confusion matrix row/column
    and zero scores instead of an error or a silently shrunken matrix."""
    y_true = np.array([0, 0, 2, 2, 2])
    y_pred = np.array([0, 2, 2, 2, 0])
    m = classification_metrics(y_true, y_pred, CLASSES)
    assert m["confusion_matrix"] == [[1, 0, 1], [0, 0, 0], [1, 0, 2]]
    assert m["per_class"]["Neutral"] == {"precision": 0.0, "recall": 0.0, "f1": 0.0, "support": 0}
    assert m["accuracy"] == pytest.approx(3 / 5)
    # macro F1 averages over all three labels, including the empty Neutral class
    f_down = 2 * (1 / 2) * (1 / 2) / (1 / 2 + 1 / 2)
    f_up = 2 * (2 / 3) * (2 / 3) / (2 / 3 + 2 / 3)
    assert m["f1_macro"] == pytest.approx((f_down + 0.0 + f_up) / 3)


def test_log_loss_and_brier_reported_only_with_probabilities(rng):
    """Without probabilities log loss and Brier are NaN; with valid probabilities they equal
    sklearn's log loss and the multiclass Brier score."""
    y_true = rng.integers(0, 3, 200)
    y_pred = rng.integers(0, 3, 200)
    proba = _random_proba(rng, 200)

    m_hard = classification_metrics(y_true, y_pred, CLASSES)
    assert np.isnan(m_hard["log_loss"]) and np.isnan(m_hard["brier"])

    m_soft = classification_metrics(y_true, y_pred, CLASSES, proba=proba)
    assert m_soft["log_loss"] == pytest.approx(log_loss(y_true, proba, labels=[0, 1, 2]), rel=1e-9)
    assert m_soft["brier"] == pytest.approx(multiclass_brier(y_true, proba, 3), rel=1e-9)


def test_log_loss_is_finite_for_zero_probability_on_true_class():
    """Probabilities are clipped before the log, so a confident miss gives a large but finite loss."""
    y_true = np.array([0, 2])
    proba = np.array([[0.0, 0.0, 1.0], [0.0, 0.0, 1.0]])
    m = classification_metrics(y_true, np.array([2, 2]), CLASSES, proba=proba)
    assert np.isfinite(m["log_loss"]) and m["log_loss"] > 5.0


def test_multiclass_brier_hand_values():
    """Brier = mean_i sum_k (p_ik - 1[y_i = k])^2: perfect forecasts -> 0, uniform 3-class
    forecasts -> (2/3)^2 + 2*(1/3)^2 = 2/3, confident misses -> 2, and a hand example -> 0.2."""
    y = np.array([0, 1, 2, 2])
    assert multiclass_brier(y, np.eye(3)[y], 3) == 0.0
    assert multiclass_brier(y, np.full((4, 3), 1 / 3), 3) == pytest.approx(2 / 3)
    assert multiclass_brier(np.array([0]), np.array([[0.0, 0.0, 1.0]]), 3) == pytest.approx(2.0)
    # sample 0: 0.3^2+0.2^2+0.1^2 = 0.14 ; sample 1: 0.1^2+0.3^2+0.4^2 = 0.26 ; mean = 0.20
    proba = np.array([[0.7, 0.2, 0.1], [0.1, 0.3, 0.6]])
    assert multiclass_brier(np.array([0, 2]), proba, 3) == pytest.approx(0.20)


# ---------------------------------------------------------------------------------------------
# calibration
# ---------------------------------------------------------------------------------------------


def test_reliability_curve_and_ece_for_perfectly_calibrated_forecasts(rng):
    """If outcomes are drawn as Bernoulli(p), observed frequency tracks predicted probability in
    every bin and the expected calibration error is close to 0."""
    p = rng.random(200_000)
    y = (rng.random(200_000) < p).astype(int)
    curve = reliability_curve(p, y, n_bins=10)
    assert list(curve.columns) == ["bin", "mean_predicted", "observed_frequency", "count"]
    assert curve["count"].sum() == len(p)
    np.testing.assert_allclose(curve["observed_frequency"], curve["mean_predicted"], atol=0.02)
    assert expected_calibration_error(p, y, n_bins=10) < 0.01


def test_ece_detects_overconfident_forecasts(rng):
    """Always predicting 0.9 for an event that happens half the time has ECE = |0.9 - 0.5| = 0.4."""
    y = rng.integers(0, 2, 20_000)
    ece = expected_calibration_error(np.full(len(y), 0.9), y, n_bins=10)
    assert ece == pytest.approx(0.4, abs=0.02)


def test_ece_hand_example_and_bin_edges():
    """Two populated bins of equal weight: |0.15 - 0| and |0.85 - 0.5| -> ECE = 0.5*0.15 + 0.5*0.35.
    Probabilities of exactly 0 and 1 fall into the first and last bin rather than out of range."""
    p = np.array([0.15, 0.15, 0.85, 0.85])
    y = np.array([0, 0, 1, 0])
    assert expected_calibration_error(p, y, n_bins=10) == pytest.approx(0.5 * 0.15 + 0.5 * 0.35)
    edges = reliability_curve(np.array([0.0, 1.0]), np.array([0, 1]), n_bins=10)
    assert list(edges["bin"]) == [0, 9]
    assert np.isnan(expected_calibration_error(np.array([]), np.array([])))


# ---------------------------------------------------------------------------------------------
# circular block bootstrap
# ---------------------------------------------------------------------------------------------


def test_block_bootstrap_indices_shape_and_range():
    """The index matrix has shape (n_boot, n) and every index is a valid row number 0..n-1, also
    when n is not a multiple of the block length and when block_len exceeds n."""
    n, L, B = 23, 5, 200
    idx = circular_block_bootstrap_indices(n, L, B, seed=1)
    assert idx.shape == (B, n)
    assert idx.min() >= 0 and idx.max() <= n - 1
    big = circular_block_bootstrap_indices(7, 50, 10, seed=1)
    assert big.shape == (10, 7)
    assert big.min() >= 0 and big.max() <= 6


def test_block_bootstrap_blocks_are_contiguous_modulo_n():
    """Within each block of length L consecutive indices increase by 1 and wrap from n-1 to 0
    (the series is treated as a circle); wrap-arounds must actually occur."""
    n, L, B = 23, 5, 200
    idx = circular_block_bootstrap_indices(n, L, B, seed=3)
    wraps = 0
    for row in idx:
        for start in range(0, n, L):
            block = row[start : min(start + L, n)]
            np.testing.assert_array_equal(np.diff(block) % n, 1)
            wraps += int(np.any(np.diff(block) == -(n - 1)))
    assert wraps > 0


def test_block_bootstrap_is_deterministic_given_seed():
    """Same seed -> identical resamples (reproducible CIs); different seed -> different resamples."""
    a = circular_block_bootstrap_indices(100, 10, 50, seed=42)
    b = circular_block_bootstrap_indices(100, 10, 50, seed=42)
    c = circular_block_bootstrap_indices(100, 10, 50, seed=43)
    np.testing.assert_array_equal(a, b)
    assert not np.array_equal(a, c)


def test_bootstrap_ci_returns_nan_for_too_short_input(rng):
    """With fewer than 2 * block_len observations no meaningful block resample exists -> NaN."""
    out = bootstrap_ci(rng.normal(size=39), np.mean, block_len=20, n_boot=100)
    assert np.isnan(out["estimate"]) and np.isnan(out["low"]) and np.isnan(out["high"])


def test_bootstrap_ci_estimate_is_full_sample_statistic(rng):
    """The point estimate is the statistic on the full (un-resampled) data and lies inside the
    percentile interval; results are reproducible for a fixed seed."""
    x = rng.normal(0.1, 1.0, 400)
    out = bootstrap_ci(x, np.mean, block_len=10, n_boot=500, ci=0.9, seed=7)
    assert out["estimate"] == pytest.approx(float(np.mean(x)), rel=0, abs=1e-15)
    assert out["low"] < out["estimate"] < out["high"]
    assert 0.0 <= out["share_positive"] <= 1.0
    assert bootstrap_ci(x, np.mean, block_len=10, n_boot=500, ci=0.9, seed=7) == out


def test_bootstrap_ci_resamples_rows_of_paired_data(rng):
    """For 2-column paired data the statistic sees whole rows: the accuracy-difference estimate
    equals mean(col0) - mean(col1) and the Sharpe statistic is finite on a random return series."""
    a = np.column_stack([rng.random(300) < 0.6, rng.random(300) < 0.5]).astype(float)
    out = bootstrap_ci(a, accuracy_difference_stat(), block_len=10, n_boot=200, seed=0)
    assert out["estimate"] == pytest.approx(a[:, 0].mean() - a[:, 1].mean())
    assert out["low"] <= out["estimate"] <= out["high"]
    sr = bootstrap_ci(rng.normal(0.001, 0.01, 300), sharpe_stat(252), block_len=10, n_boot=200)
    assert np.isfinite(sr["estimate"]) and sr["low"] < sr["high"]


# ---------------------------------------------------------------------------------------------
# feature analysis
# ---------------------------------------------------------------------------------------------


def test_mutual_information_frame_sorted_descending(rng):
    """MI table has one row per feature, is sorted by MI descending, and ranks the feature that
    determines the label first and pure noise features last."""
    n = 2000
    signal = rng.normal(size=n)
    y = (signal > 0).astype(int) + (signal > 1).astype(int)  # 3 classes driven by `signal`
    X = np.column_stack([rng.normal(size=n), signal, rng.normal(size=n)])
    names = ["noise_a", "signal", "noise_b"]
    mi = mutual_information_frame(X, y, names, seed=0)
    assert list(mi.columns) == ["feature", "mutual_information"]
    assert sorted(mi["feature"]) == sorted(names)
    assert mi["mutual_information"].is_monotonic_decreasing
    assert mi["feature"].iloc[0] == "signal"
    assert mi["mutual_information"].iloc[0] > 5 * mi["mutual_information"].iloc[1]
    assert list(mi.index) == [0, 1, 2]


# ---------------------------------------------------------------------------------------------
# naive classification baseline
# ---------------------------------------------------------------------------------------------


def test_majority_class_baseline_prior_and_predictions():
    """The prior equals training class frequencies (including a class with zero count) and every
    prediction is the most frequent training class."""
    y = np.array([0, 0, 2, 2, 2, 2])
    base = MajorityClassBaseline(n_classes=3).fit(y)
    np.testing.assert_allclose(base.prior_, [2 / 6, 0.0, 4 / 6])
    assert base.majority_ == 2
    np.testing.assert_array_equal(base.predict(4), [2, 2, 2, 2])
    proba = base.predict_proba(4)
    assert proba.shape == (4, 3)
    np.testing.assert_allclose(proba, np.tile([2 / 6, 0.0, 4 / 6], (4, 1)))
    np.testing.assert_allclose(proba.sum(axis=1), 1.0)


# ---------------------------------------------------------------------------------------------
# regimes
# ---------------------------------------------------------------------------------------------


def test_regime_labels_use_training_median_only():
    """The volatility threshold is the median over TRAIN rows only (here 3). Using the full-sample
    median (52.5) would label the values 4 and 5 as low_vol, i.e. leak test-period information."""
    idx = pd.bdate_range("2024-01-01", periods=10)
    feats = pd.DataFrame(
        {
            "vol_20": [1.0, 2.0, 3.0, 4.0, 5.0, 100.0, 100.0, 100.0, 100.0, 100.0],
            "trend_regime": [0.1, -0.1, 0.0, 0.2, -0.2, 0.3, -0.3, 0.0, 0.05, -0.05],
        },
        index=idx,
    )
    out = regime_labels(feats, train_index=idx[:5])
    assert list(out["volatility_regime"]) == ["low_vol"] * 3 + ["high_vol"] * 7  # 3 == median -> low
    assert list(out["trend_regime"]) == [
        "uptrend", "downtrend", "downtrend", "uptrend", "downtrend",
        "uptrend", "downtrend", "downtrend", "uptrend", "downtrend",
    ]
    assert out.index.equals(idx)


def test_regime_labels_on_pipeline_features_ignore_post_train_data(prices, small_cfg):
    """On real pipeline features, inflating post-train volatility tenfold must not move the
    threshold: every row is still classified against the unchanged TRAIN median, so the labels of
    all training rows are identical before and after the shock."""
    feats = compute_features(prices, small_cfg["features"]).dropna(subset=["vol_20", "trend_regime"])
    half = len(feats) // 2
    train_idx = feats.index[:half]
    train_median = feats.loc[train_idx, "vol_20"].median()

    shocked = feats.copy()
    shocked.loc[shocked.index[half:], "vol_20"] *= 10.0
    assert shocked["vol_20"].median() != pytest.approx(train_median)  # a full-sample median would differ

    base = regime_labels(feats, train_idx)
    out = regime_labels(shocked, train_idx)
    for frame, labels in ((feats, base), (shocked, out)):
        expected = np.where(frame["vol_20"] > train_median, "high_vol", "low_vol")
        np.testing.assert_array_equal(labels["volatility_regime"].values, expected)
    pd.testing.assert_series_equal(base.loc[train_idx, "volatility_regime"], out.loc[train_idx, "volatility_regime"])
