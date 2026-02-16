"""Train / validation / test splitting for time-series and tabular data."""

from __future__ import annotations

from typing import Generator

import numpy as np
import pandas as pd
from sklearn.model_selection import BaseCrossValidator


class RollingWindowSplitter(BaseCrossValidator):
    """Rolling-window splitter compatible with :mod:`sklearn`.

    For time-series data (when *groups* = date index), windows are defined by
    calendar duration.  For plain tabular data (*groups* = None), observation
    counts are used instead.

    Parameters
    ----------
    train_len, test_len, step : str or int
        Duration strings (``'365D'``, ``'92D'``, …) for time-series mode,
        or integers for tabular mode.
    min_train_obs, min_test_obs : int
        Minimum number of observations required in each fold.
    """

    def __init__(
        self,
        train_len: str | int = "365D",
        test_len: str | int = "92D",
        step: str | int = "30D",
        min_train_obs: int = 180,
        min_test_obs: int = 40,
    ) -> None:
        self.train_len = train_len
        self.test_len = test_len
        self.step = step
        self.min_train_obs = min_train_obs
        self.min_test_obs = min_test_obs

    # ---- sklearn interface ---------------------------------------------------

    def split(self, X, y=None, groups=None):  # type: ignore[override]
        """Yield ``(train_idx, test_idx)`` arrays.

        Parameters
        ----------
        X : array-like, shape (n, …)
        y : ignored.
        groups : array-like of datetime, optional.
            If provided, switches to calendar-based windowing.
        """
        if groups is not None:
            yield from self._split_ts(np.asarray(groups), len(X))
        else:
            yield from self._split_tabular(len(X))

    def get_n_splits(self, X=None, y=None, groups=None) -> int:
        if X is None:
            return 0
        return sum(1 for _ in self.split(X, y, groups))

    # ---- extended interface --------------------------------------------------

    def split_with_info(
        self,
        X,
        groups=None,
    ) -> Generator[tuple[pd.Timestamp | int, np.ndarray, np.ndarray], None, None]:
        """Like :meth:`split` but also yields the train-end boundary.

        Yields ``(train_end, train_idx, test_idx)`` where *train_end* is the
        last training date (or index) useful for labelling windows.
        """
        if groups is not None:
            dates = pd.DatetimeIndex(groups)
        else:
            dates = None

        for tr_idx, te_idx in self.split(X, groups=groups):
            if dates is not None:
                train_end = dates[tr_idx[-1]]
            else:
                train_end = int(tr_idx[-1])
            yield train_end, tr_idx, te_idx

    # ---- private -------------------------------------------------------------

    def _split_ts(self, dates_arr, n: int):
        idx = pd.DatetimeIndex(dates_arr)
        train_delta = pd.Timedelta(str(self.train_len))
        test_delta = pd.Timedelta(str(self.test_len))
        step_delta = pd.Timedelta(str(self.step))

        start = idx.min()
        last_date = idx.max()
        while True:
            train_end = start + train_delta
            test_end = train_end + test_delta
            if test_end > last_date:
                break
            tr_mask = (idx >= start) & (idx < train_end)
            te_mask = (idx >= train_end) & (idx < test_end)
            tr_idx = np.where(tr_mask)[0]
            te_idx = np.where(te_mask)[0]
            if len(tr_idx) >= self.min_train_obs and len(te_idx) >= self.min_test_obs:
                yield tr_idx, te_idx
            start += step_delta

    def _split_tabular(self, n: int):
        train_size = (
            int(self.train_len)
            if isinstance(self.train_len, int)
            else self.min_train_obs
        )
        test_size = (
            int(self.test_len) if isinstance(self.test_len, int) else self.min_test_obs
        )
        step_size = (
            int(self.step) if isinstance(self.step, int) else max(1, test_size // 3)
        )

        start = 0
        while start + train_size + test_size <= n:
            tr = np.arange(start, start + train_size)
            te = np.arange(start + train_size, start + train_size + test_size)
            if len(tr) >= self.min_train_obs and len(te) >= self.min_test_obs:
                yield tr, te
            start += step_size


class NestedTimeSeriesSplit:
    """Single nested split: first portion inner-train, rest validation.

    Parameters
    ----------
    inner_frac : float
        Fraction-based split point (for tabular data).
    inner_duration : str
        Duration from start of training to end of inner-train block
        (for time-series data), e.g. ``'270D'``.
    """

    def __init__(
        self,
        inner_frac: float = 0.75,
        inner_duration: str = "270D",
    ) -> None:
        self.inner_frac = inner_frac
        self.inner_duration = inner_duration

    def split(
        self,
        X: np.ndarray,
        dates: np.ndarray | pd.DatetimeIndex | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return ``(inner_idx, val_idx)`` arrays.

        Parameters
        ----------
        X : array-like, shape (n, p)
        dates : date index for time-series-aware split.

        Returns
        -------
        inner_idx, val_idx : 1-D int arrays.
        """
        n = len(X)
        if dates is not None:
            tr_dates = pd.DatetimeIndex(pd.to_datetime(dates))
            inner_end = tr_dates.min() + pd.Timedelta(self.inner_duration)
            inner_mask = tr_dates < inner_end
            val_mask = tr_dates >= inner_end
            return np.where(inner_mask)[0], np.where(val_mask)[0]
        split_pt = int(n * self.inner_frac)
        return np.arange(split_pt), np.arange(split_pt, n)
