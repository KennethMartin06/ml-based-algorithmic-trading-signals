"""Dataset assembly, chronological splitting with purging, walk-forward folds and sequence windows.

Leakage controls implemented here
---------------------------------
* Splits are by date only - no shuffling of observations across time.
* Purging: a row whose label window ends after its split boundary is dropped from that split.
  Example: under next-open execution the label of 2019-12-31 depends on the open of 2020-01-03,
  which lies in the validation period, so that row is removed from the training set.
* Scalers are fitted on training rows only (see :func:`fit_train_scaler`; sklearn models use a
  ``Pipeline`` so cross-validation also refits the scaler inside each fold).
* LSTM windows end at the decision row t and contain rows ``t-L+1 .. t`` only.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

META_COLUMNS = ["fwd_ret", "label", "label_end_date", "period_ret"]


def assemble_dataset(features: pd.DataFrame, targets: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Join features and targets and drop indicator warm-up rows.

    Rows at the end of the sample keep valid features but have NaN labels (their future is not
    observed); they are retained because they can still serve as LSTM context, but are never used
    as labelled samples.
    """
    data = features.join(targets, how="inner")
    feature_cols = list(features.columns)
    complete = data[feature_cols].notna().all(axis=1)
    first_valid = complete.idxmax() if complete.any() else None
    report = {
        "rows_before_warmup_drop": int(len(data)),
        "first_complete_feature_date": first_valid.strftime("%Y-%m-%d") if first_valid is not None else None,
    }
    if first_valid is None:
        all_nan = [c for c in feature_cols if data[c].isna().all()]
        if all_nan:  # e.g. instruments without volume data (FX, indices) -> volume features undefined
            raise ValueError(f"Features with no valid value at all: {all_nan}. Check the input data (e.g. zero volume).")
        raise ValueError(f"No row has a complete feature vector ({len(data)} rows); the history is too short for the windows used.")
    data = data.loc[first_valid:]
    # after warm-up, any remaining NaN feature row is dropped (and counted) rather than imputed
    incomplete_after = ~data[feature_cols].notna().all(axis=1)
    report["incomplete_rows_after_warmup_dropped"] = int(incomplete_after.sum())
    data = data[~incomplete_after]
    report["rows"] = int(len(data))
    report["labelled_rows"] = int(data["label"].notna().sum())
    return data, report


@dataclass
class Split:
    train: pd.DatetimeIndex
    val: pd.DatetimeIndex
    test: pd.DatetimeIndex

    def as_dict(self) -> dict[str, pd.DatetimeIndex]:
        return {"train": self.train, "val": self.val, "test": self.test}

    def describe(self) -> dict:
        out = {}
        for name, idx in self.as_dict().items():
            out[name] = {
                "rows": int(len(idx)),
                "start": idx[0].strftime("%Y-%m-%d") if len(idx) else None,
                "end": idx[-1].strftime("%Y-%m-%d") if len(idx) else None,
            }
        return out


def _segment(data: pd.DataFrame, start, end, purge: bool) -> pd.DatetimeIndex:
    labelled = data[data["label"].notna()]
    mask = pd.Series(True, index=labelled.index)
    if start is not None:
        mask &= labelled.index > pd.Timestamp(start)
    if end is not None:
        mask &= labelled.index <= pd.Timestamp(end)
        if purge:
            mask &= labelled["label_end_date"] <= pd.Timestamp(end)
    return labelled.index[mask.values]


def chronological_split(data: pd.DataFrame, train_end: str, val_end: str, purge: bool = True) -> Split:
    """Train (<= train_end), validation (train_end, val_end], test (> val_end)."""
    if pd.Timestamp(train_end) >= pd.Timestamp(val_end):
        raise ValueError("train_end must precede val_end")
    return Split(
        train=_segment(data, None, train_end, purge),
        val=_segment(data, train_end, val_end, purge),
        test=_segment(data, val_end, None, purge),
    )


def walk_forward_folds(
    data: pd.DataFrame,
    first_test_year: int,
    last_test_year: int,
    val_years: int = 1,
    purge: bool = True,
    min_train_rows: int = 500,
) -> list[tuple[int, Split]]:
    """Expanding-window folds: train = all years before the validation block, validation =
    ``val_years`` years immediately preceding the test year, test = one calendar year."""
    folds = []
    for year in range(first_test_year, last_test_year + 1):
        train_end = f"{year - val_years - 1}-12-31"
        val_end = f"{year - 1}-12-31"
        test_end = f"{year}-12-31"
        split = Split(
            train=_segment(data, None, train_end, purge),
            val=_segment(data, train_end, val_end, purge),
            # test rows are never trained on, so their labels may extend past Dec 31
            test=_segment(data, val_end, test_end, purge=False),
        )
        if len(split.train) >= min_train_rows and len(split.val) > 0 and len(split.test) > 0:
            folds.append((year, split))
    return folds


def fit_train_scaler(X: pd.DataFrame, train_index: pd.DatetimeIndex) -> StandardScaler:
    """StandardScaler fitted on training rows only."""
    return StandardScaler().fit(X.loc[train_index].values)


def build_sequences(X: np.ndarray, end_positions: np.ndarray, seq_len: int) -> np.ndarray:
    """Stack windows ``X[p-seq_len+1 : p+1]`` for each end position p.

    The window for decision row p contains only rows at or before p, so no future feature enters.
    """
    end_positions = np.asarray(end_positions, dtype=int)
    if (end_positions < seq_len - 1).any():
        raise ValueError("Some end positions do not have seq_len rows of history")
    if (end_positions >= len(X)).any():
        raise ValueError("End position outside the feature matrix")
    offsets = np.arange(-seq_len + 1, 1)
    return X[end_positions[:, None] + offsets[None, :]]
