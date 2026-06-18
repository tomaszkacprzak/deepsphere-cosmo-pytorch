import numpy as np
import healpy as hp

from deepsphere import utils


def test_extend_indices():
    # defs
    nside_in = 4
    nside_out = 2

    # create a set of indices
    indices = np.arange(hp.nside2npix(nside_in))[::4]

    # get the expanded set
    new_indices = utils.extend_indices(indices, nside_in=nside_in, nside_out=nside_out)

    assert len(new_indices) == hp.nside2npix(nside_in)

    # this should also work reorderd
    m_nest = np.zeros(hp.nside2npix(nside_in))
    m_nest[::4] = 1.0
    m_ring = hp.reorder(map_in=m_nest, n2r=True)

    # get the indices
    indices = np.arange(hp.nside2npix(nside_in))[m_ring > 0.0]

    # get the expanded set
    new_indices = utils.extend_indices(indices, nside_in=nside_in, nside_out=nside_out, nest=False)

    assert len(new_indices) == hp.nside2npix(nside_in)


def test_split_sparse_dense_matmul_matches_torch_sparse_mm_with_splits():
    import torch

    indices = torch.tensor([[0, 1, 1], [1, 0, 2]])
    values = torch.tensor([2.0, 3.0, 4.0])
    sparse_tensor = torch.sparse_coo_tensor(indices, values, (2, 3)).coalesce()
    dense_tensor = torch.arange(15, dtype=torch.float32).reshape(3, 5)

    expected = torch.sparse.mm(sparse_tensor, dense_tensor)
    actual = utils.split_sparse_dense_matmul(sparse_tensor, dense_tensor, n_splits=2)

    assert torch.allclose(actual, expected)


def test_gaussian_noise_layer_registers_buffer_and_broadcasts_channels():
    import torch

    torch.manual_seed(11)
    inputs = torch.zeros((2, 4, 3))
    layer = utils.GaussianNoiseLayer([0.0, 1.0, 2.0])

    assert "stddev" in dict(layer.named_buffers())
    outputs = layer(inputs)

    assert torch.allclose(outputs[..., 0], torch.zeros_like(outputs[..., 0]))
    assert torch.any(outputs[..., 1] != 0)
    assert torch.any(outputs[..., 2] != 0)
