"""Tests for grpca.splitters — RollingWindowSplitter and NestedTimeSeriesSplit."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from grpca.splitters import NestedTimeSeriesSplit, RollingWindowSplitter

# ---------------------------------------------------------------------------
# RollingWindowSplitter — time-series mode
# ---------------------------------------------------------------------------


class TestRollingWindowSplitterTS:
    @pytest.fixture
    def dates_500(self):
        return pd.bdate_range("2022-01-03", periods=500)

    @pytest.fixture
    def X_500(self, rng: np.random.Generator):
        return rng.normal(size=(500, 5))

    def test_yields_windows(self, X_500, dates_500):
        splitter = RollingWindowSplitter(
            train_len="365D", test_len="30D", step="30D",
            min_train_obs=100, min_test_obs=10,
        )
        windows = list(splitter.split(X_500, groups=dates_500))
        assert len(windows) >= 1

    def test_train_test_no_overlap(self, X_500, dates_500):
        splitter = RollingWindowSplitter(
            train_len="180D", test_len="30D", step="30D",
            min_train_obs=50, min_test_obs=5,
        )
        for tr_idx, te_idx in splitter.split(X_500, groups=dates_500):
            assert len(np.intersect1d(tr_idx, te_idx)) == 0

    def test_train_before_test(self, X_500, dates_500):
        splitter = RollingWindowSplitter(
            train_len="180D", test_len="30D", step="30D",
            min_train_obs=50, min_test_obs=5,
        )
        for tr_idx, te_idx in splitter.split(X_500, groups=dates_500):
            assert tr_idx[-1] < te_idx[0]

    def test_minimum_obs_filtering(self, X_500, dates_500):
        splitter = RollingWindowSplitter(
            train_len="60D", test_len="10D", step="10D",
            min_train_obs=999, min_test_obs=5,
        )
        windows = list(splitter.split(X_500, groups=dates_500))
        assert len(windows) == 0

    def test_get_n_splits(self, X_500, dates_500):
        splitter = RollingWindowSplitter(
            train_len="180D", test_len="30D", step="30D",
            min_train_obs=50, min_test_obs=5,
        )
        n = splitter.get_n_splits(X_500, groups=dates_500)
        assert n == len(list(splitter.split(X_500, groups=dates_500)))

    def test_split_with_info_yields_date(self, X_500, dates_500):
        splitter = RollingWindowSplitter(
            train_len="180D", test_len="30D", step="30D",
            min_train_obs=50, min_test_obs=5,
        )
        for end_dt, tr_idx, te_idx in splitter.split_with_info(X_500, groups=dates_500):
            assert isinstance(end_dt, pd.Timestamp)
            assert len(tr_idx) >= 50
            assert len(te_idx) >= 5


# ---------------------------------------------------------------------------
# RollingWindowSplitter — tabular mode
# ---------------------------------------------------------------------------


class TestRollingWindowSplitterTabular:
    def test_integer_params(self, rng: np.random.Generator):
        X = rng.normal(size=(200, 3))
        splitter = RollingWindowSplitter(
            train_len=100, test_len=30, step=20,
            min_train_obs=50, min_test_obs=10,
        )
        windows = list(splitter.split(X))
        assert len(windows) >= 1
        for tr, te in windows:
            assert len(tr) == 100
            assert len(te) == 30
            assert tr[-1] < te[0]


# ---------------------------------------------------------------------------
# NestedTimeSeriesSplit
# ---------------------------------------------------------------------------


class TestNestedTimeSeriesSplit:
    def test_fraction_split(self):
        X = np.zeros((100, 4))
        splitter = NestedTimeSeriesSplit(inner_frac=0.75)
        inner, val = splitter.split(X)
        assert len(inner) == 75
        assert len(val) == 25
        assert inner[-1] < val[0]

    def test_duration_split(self):
        dates = pd.bdate_range("2020-01-01", periods=300)
        X = np.zeros((300, 3))
        splitter = NestedTimeSeriesSplit(inner_duration="180D")
        inner, val = splitter.split(X, dates=dates)
        assert len(inner) > 0
        assert len(val) > 0
        # inner dates should be before inner_end
        inner_end = dates.min() + pd.Timedelta("180D")
        assert all(dates[i] < inner_end for i in inner)
        assert all(dates[i] >= inner_end for i in val)

    def test_no_overlap(self):
        X = np.zeros((80, 2))
        splitter = NestedTimeSeriesSplit(inner_frac=0.6)
        inner, val = splitter.split(X)
        assert len(np.intersect1d(inner, val)) == 0

    def test_covers_all_indices(self):
        X = np.zeros((50, 3))
        splitter = NestedTimeSeriesSplit(inner_frac=0.8)
        inner, val = splitter.split(X)
        combined = np.sort(np.concatenate([inner, val]))
        assert np.array_equal(combined, np.arange(50))
