import inspect

import numpy as np
import pytest
import torch

from tslearn.metrics import soft_dtw
from tslearn.metrics import cdist_soft_dtw
from tslearn.metrics import SoftDTWLossPyTorch
from tslearn.barycenters import softdtw_barycenter


@pytest.mark.parametrize("sz", [100, 1000, 10000])
def test_soft_dtw_numpy(benchmark, sz):
    X = np.random.rand(sz, 2)
    benchmark(soft_dtw, X, X)


@pytest.mark.parametrize("sz", [100, 1000])
def test_soft_dtw_torch_cpu(benchmark, sz):
    X = torch.rand((sz, 2))
    kwargs = {}
    if 'compute_with_backend' in inspect.signature(soft_dtw).parameters:
        kwargs.update({'compute_with_backend': True})
    benchmark(soft_dtw, X, X, **kwargs)


@pytest.mark.parametrize("n_ts, sz, n_jobs", [(10, 100, 1), (10, 1000, 1), (10, 1000, -1)])
def test_cdist_soft_dtw_numpy(benchmark, n_ts, sz, n_jobs):
    X = np.random.rand(n_ts, sz, 2)
    kwargs = {}
    if 'n_jobs' in inspect.signature(cdist_soft_dtw).parameters:
        kwargs.update({'n_jobs': n_jobs})
    benchmark(cdist_soft_dtw, X, X, **kwargs)


@pytest.mark.parametrize("n_ts, sz, n_jobs", [(10, 100, 1), (10, 100, -1)])
def test_cdist_soft_dtw_torch_cpu(benchmark, n_ts, sz, n_jobs):
    X = torch.rand((n_ts, sz, 2))
    kwargs = {}
    if 'compute_with_backend' in inspect.signature(soft_dtw).parameters:
        kwargs.update({'compute_with_backend': True})
    if 'n_jobs' in inspect.signature(cdist_soft_dtw).parameters:
        kwargs.update({'n_jobs': n_jobs})
    benchmark(cdist_soft_dtw, X, X, **kwargs)


@pytest.mark.parametrize("n_ts, sz, n_jobs", [(10, 100, 1), (10, 1000, -1)])
def test_soft_dtw_barycenter(benchmark, n_ts, sz, n_jobs):
    rng = np.random.default_rng(0)
    X = rng.random((n_ts, sz, 2))
    if 'n_jobs' in inspect.signature(softdtw_barycenter).parameters:
        benchmark(softdtw_barycenter, X, max_iter=1, n_jobs=n_jobs)
    else:
        benchmark(softdtw_barycenter, X, max_iter=1)


@pytest.mark.parametrize("n_ts, sz", [(10, 100), (10, 1000), (100, 1000)])
def test_SoftDTWLossPyTorch(benchmark, n_ts, sz):
    X = torch.rand((n_ts, sz, 2))
    benchmark(SoftDTWLossPyTorch().forward, X, X)
