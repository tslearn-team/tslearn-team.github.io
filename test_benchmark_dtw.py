import inspect

import numpy as np
import pytest
import torch

from tslearn.metrics import dtw
from tslearn.metrics import cdist_dtw
from tslearn.barycenters import dtw_barycenter_averaging_petitjean


@pytest.mark.parametrize("sz", [100, 1000, 2000, 10000, 20000])
def test_dtw_numpy(benchmark, sz):
    X = np.random.rand(sz, 2)
    benchmark(dtw, X, X)


@pytest.mark.parametrize("sz", [10, 100, 1000])
def test_dtw_torch_cpu(benchmark, sz):
    X = torch.rand((sz, 2))
    benchmark(dtw, X, X)


@pytest.mark.parametrize("n_ts, sz, n_jobs", [(10, 100, 1), (10, 1000, 1), (10, 1000, -1), (50, 5000, -1)])
def test_cdist_dtw_numpy(benchmark, n_ts, sz, n_jobs):
    X = np.random.rand(n_ts, sz, 2)
    benchmark(cdist_dtw, X, X, n_jobs=n_jobs)


@pytest.mark.parametrize("n_ts, sz, n_jobs", [(10, 100, 1), (10, 100, -1)])
def test_cdist_dtw_torch_cpu(benchmark, n_ts, sz, n_jobs):
    X = torch.rand((n_ts, sz, 2))
    benchmark(cdist_dtw, X, X, n_jobs=n_jobs)


@pytest.mark.parametrize("n_ts, sz, n_jobs", [(10, 100, 1), (100, 1000, -1)])
def test_dtw_barycenter(benchmark, n_ts, sz, n_jobs):
    rng = np.random.default_rng(0)
    X = rng.random((n_ts, sz, 2))
    if 'n_jobs' in inspect.signature(dtw_barycenter_averaging_petitjean).parameters:
        benchmark(dtw_barycenter_averaging_petitjean, X, max_iter=1, n_jobs=n_jobs)
    else:
        benchmark(dtw_barycenter_averaging_petitjean, X, max_iter=1)

