"""
Unit tests for AsyncDMADataPrefetcher and Asynchronous Stream Double-Buffering [AO-24].
"""

import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from arbiter_omni.training.prefetcher import AsyncDMADataPrefetcher


def test_async_dma_prefetcher_full_iteration():
    """Validates that AsyncDMADataPrefetcher streams all batches in order without dropping items."""
    X = torch.arange(40).reshape(10, 4)
    y = torch.arange(10)
    dataset = TensorDataset(X, y)
    dataloader = DataLoader(dataset, batch_size=3, shuffle=False)

    prefetcher = AsyncDMADataPrefetcher(dataloader, device="cpu", queue_size=2)
    assert len(prefetcher) == len(dataloader)

    collected_X = []
    collected_y = []
    for batch_X, batch_y in prefetcher:
        collected_X.append(batch_X)
        collected_y.append(batch_y)

    cat_X = torch.cat(collected_X, dim=0)
    cat_y = torch.cat(collected_y, dim=0)

    assert torch.equal(cat_X, X)
    assert torch.equal(cat_y, y)


def test_async_dma_prefetcher_device_dispatch():
    """Validates tensor device transfers during prefetching."""
    dataset = TensorDataset(torch.randn(8, 4), torch.zeros(8))
    dataloader = DataLoader(dataset, batch_size=2)

    device = torch.device("cpu")
    prefetcher = AsyncDMADataPrefetcher(dataloader, device=device, queue_size=2)

    for bX, by in prefetcher:
        assert bX.device == device
        assert by.device == device


def test_async_dma_prefetcher_early_exit_close():
    """Validates clean termination when consumer breaks out of loop early."""
    dataset = TensorDataset(torch.randn(20, 4))
    dataloader = DataLoader(dataset, batch_size=2)

    prefetcher = AsyncDMADataPrefetcher(dataloader, device="cpu", queue_size=2)
    count = 0
    for _ in prefetcher:
        count += 1
        if count >= 3:
            break

    prefetcher.close()
    assert prefetcher.worker_thread is None
