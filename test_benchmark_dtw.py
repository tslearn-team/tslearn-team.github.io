import inspect

import numpy as np
import pytest
import torch

from tslearn.metrics import dtw
from tslearn.metrics import cdist_dtw
from tslearn.barycenters import dtw_barycenter_averaging_petitjean


@pytest.mark.parametrize("sz", [100, 1000, 2000, 4000, 8000])
def test_dtw_numpy(benchmark, sz):
    # Mem: sz * 64 * 3 (inputs duplicated twice) + sz * sz * 8 (mask) + sz *sz * 64 (acc matrix)
    # Care for swap, check runner mem
    X = np.random.rand(sz, 2)
    benchmark(dtw, X, X)


@pytest.mark.parametrize("sz", [10, 100, 1000])
def test_dtw_torch_cpu(benchmark, sz):
    X = torch.rand((sz, 2))
    benchmark(dtw, X, X)


@pytest.mark.parametrize("n_ts, sz, n_jobs", [(8, 1000, 1), (8, 2000, 1), (8, 2000, -1), (16, 2000, -1)])
def test_cdist_dtw_numpy(benchmark, n_ts, sz, n_jobs):
    # Mem: dtw_mem * n_jobs_effective + n_ts * n_ts *64
    # Care for swap, check runner mem
    X = np.random.rand(n_ts, sz, 2)
    benchmark(cdist_dtw, X, X, n_jobs=n_jobs)


@pytest.mark.parametrize("n_ts, sz, n_jobs", [(8, 100, 1)])
def test_cdist_dtw_torch_cpu(benchmark, n_ts, sz, n_jobs):
    # TODO: check threading on python free threading, not relevant otherwise
    X = torch.rand((n_ts, sz, 2))
    benchmark(cdist_dtw, X, X, n_jobs=n_jobs)


@pytest.mark.parametrize("n_ts, sz, n_jobs", [(7, 100, 1), (7, 1000, 1), (7, 1000, -1), (14, 1000, 1)])
def test_dtw_barycenter(benchmark, n_ts, sz, n_jobs):
    rng = np.random.default_rng(0)
    X = rng.random((n_ts, sz, 2))
    if 'n_jobs' in inspect.signature(dtw_barycenter_averaging_petitjean).parameters:
        benchmark(dtw_barycenter_averaging_petitjean, X, max_iter=1, n_jobs=n_jobs)
    else:
        benchmark(dtw_barycenter_averaging_petitjean, X, max_iter=1)

