"""Utilities module."""

import numpy as np
from scipy import sparse
import healpy as hp
import torch
from torch import nn


def extend_indices(indices, nside_in, nside_out, nest=True):
    """
    Minimally extends a set of indices such that it can be reduced to nside_out in a healpy fashion, always four pixels
    reduce naturally to a higher order pixel. Note that this function supports the ring ordering, however, since almost
    no other function does so, nest ordering is strongly recommended.
    :param indices: 1d array of integer pixel ids
    :param nside_in: nside of the input
    :param nside_out: nside of the output
    :param nest: indices are ordered in the "NEST" ordering scheme
    :return: returns a set of indices in the same ordering as the input.
    """
    # figire out the ordering
    if nest:
        ordering = "NEST"
    else:
        ordering = "RING"

    # get the map to reduce
    m_in = np.zeros(hp.nside2npix(nside_in))
    m_in[indices] = 1.0

    # reduce
    m_in = hp.ud_grade(map_in=m_in, nside_out=nside_out, order_in=ordering, order_out=ordering)

    # expand
    m_in = hp.ud_grade(map_in=m_in, nside_out=nside_in, order_in=ordering, order_out=ordering)

    # get the new indices
    return np.arange(hp.nside2npix(nside_in))[m_in > 1e-12]


def rescale_L(L, lmax=2, scale=1):
    """Rescale the Laplacian eigenvalues in [-scale,scale]."""
    M, M = L.shape
    I = sparse.identity(M, format="csr", dtype=L.dtype)
    L *= 2 * scale / lmax
    L -= I
    return L


def split_sparse_dense_matmul(sparse_tensor, dense_tensor, n_splits=1):
    """
    Multiply a rank-2 PyTorch sparse tensor by a rank-2 dense tensor.

    Splits axis 1 of ``dense_tensor`` when requested so large GPU sparse matrix
    multiplications can be run in smaller chunks.
    :param sparse_tensor: Input PyTorch sparse tensor of rank 2.
    :param dense_tensor: Input dense tensor of rank 2.
    :param n_splits: Integer number of chunks applied to axis 1 of dense_tensor.
    """
    if n_splits < 1:
        raise ValueError("n_splits must be at least 1")

    sparse_tensor = sparse_tensor.to(device=dense_tensor.device, dtype=dense_tensor.dtype)

    if n_splits > 1:
        dense_splits = torch.tensor_split(dense_tensor, n_splits, dim=1)
        result = [torch.sparse.mm(sparse_tensor, dense_split) for dense_split in dense_splits]
        return torch.cat(result, dim=1)

    return torch.sparse.mm(sparse_tensor, dense_tensor)


class GaussianNoiseLayer(nn.Module):
    """
    A layer that adds Gaussian noise to the input, where the standard deviation of the Gaussian can be set channel-wise.
    """

    def __init__(self, stddev):
        super(GaussianNoiseLayer, self).__init__()
        self.register_buffer("stddev", torch.as_tensor(stddev, dtype=torch.float32))

    def _stddev_for(self, inputs):
        if self.stddev.ndim == 0:
            return self.stddev.to(device=inputs.device, dtype=inputs.dtype)
        if self.stddev.ndim != 1:
            raise ValueError("stddev must be a scalar or a 1D tensor of per-channel standard deviations")
        if self.stddev.shape[0] != inputs.shape[-1]:
            raise ValueError("Length of stddev does not match the number of input channels")
        return self.stddev.to(device=inputs.device, dtype=inputs.dtype).view(*([1] * (inputs.ndim - 1)), -1)

    def forward(self, inputs):
        inputs = torch.as_tensor(inputs)
        stddev = self._stddev_for(inputs)
        noise = torch.randn_like(inputs) * stddev
        return inputs + noise

    def call(self, inputs):
        return self.forward(inputs)
