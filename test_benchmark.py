from tslearn.metrics import dtw
from tslearn.metrics import cdist_dtw

import numpy as np
import torch


def test_dtw_numpy(benchmark):
    X = np.random.rand(10, 2)
    benchmark(dtw, X, X)


def test_dtw_torch_cpu(benchmark):
    X = torch.rand((10, 2))
    benchmark(dtw, X, X)


def test_cdist_dtw_numpy(benchmark):
    X = np.random.rand(10, 100, 2)
    benchmark(cdist_dtw, X)


def test_cdist_dtw_torch_cpu(benchmark):
    X = torch.rand((10, 100, 2))
    benchmark(cdist_dtw, X)