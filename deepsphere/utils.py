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
    Splits axis 1 of the dense_tensor for sparse PyTorch matrix multiplication.
    :param sparse_tensor: Input sparse tensor of rank 2.
    :param dense_tensor: Input dense tensor of rank 2.
    :param n_splits: Integer number of splits applied to axis 1 of dense_tensor.

    This is retained as a PyTorch replacement for the historical TensorFlow
    helper that split very large sparse/dense matmul calls.
    """
    if n_splits > 1:
        print(
            f"Due to tensor size, torch.sparse.mm is executed over {n_splits} splits."
            f" Beware of the resulting performance penalty."
        )
        dense_splits = torch.chunk(dense_tensor, n_splits, dim=1)
        result = []
        for dense_split in dense_splits:
            result.append(torch.sparse.mm(sparse_tensor, dense_split))
        result = torch.cat(result, dim=1)
    else:
        result = torch.sparse.mm(sparse_tensor, dense_tensor)

    return result


class GaussianNoiseLayer(nn.Module):
    """
    A layer that adds Gaussian noise to the input, where the standard deviation of the Gaussian can be set channel-wise
    """

    def __init__(self, stddev, **kwargs):
        super(GaussianNoiseLayer, self).__init__()
        self.register_buffer("stddev", torch.as_tensor(stddev, dtype=torch.float32))

    def build(self, input_shape):
        if len(self.stddev.shape) == 0:
            self.stddev = torch.ones((input_shape[-1],), device=self.stddev.device) * self.stddev
        elif self.stddev.shape[0] != input_shape[-1]:
            raise ValueError("Length of stddev does not match the number of input channels")

    def forward(self, inputs):
        inputs = inputs if torch.is_tensor(inputs) else torch.as_tensor(inputs, dtype=torch.get_default_dtype())
        stddev = self.stddev.to(device=inputs.device, dtype=inputs.dtype)
        if len(stddev.shape) == 0:
            stddev = torch.ones((inputs.shape[-1],), device=inputs.device, dtype=inputs.dtype) * stddev
        noise = torch.randn_like(inputs) * stddev

        return inputs + noise

    call = forward
