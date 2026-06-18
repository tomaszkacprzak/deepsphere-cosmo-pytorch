"""Fixtures documenting the PyTorch port's public tensor layout.

All public DeepSphere layers accept and return ``(batch, nodes, channels)``.
Only internal adapters around channel-first PyTorch layers transpose to
``(batch, channels, nodes)``.
"""

import numpy as np
import pytest
import torch
from scipy import sparse

from deepsphere import gnn_layers, gnn_transformers, healpy_layers, utils


@pytest.fixture
def public_layout_tensor():
    """A documented public-layout fixture: ``(batch, nodes, channels)``."""

    return torch.randn(2, 16, 3, generator=torch.Generator().manual_seed(123))


def _laplacian(nodes):
    return np.eye(nodes, dtype=np.float64)


def test_layout_helpers_are_inverse(public_layout_tensor):
    channel_first = utils.nodes_channels_to_channels_nodes(public_layout_tensor)
    assert channel_first.shape == (2, 3, 16)

    public = utils.channels_nodes_to_nodes_channels(channel_first)
    assert public.shape == public_layout_tensor.shape
    torch.testing.assert_close(public, public_layout_tensor)


@pytest.mark.parametrize(
    "layer,expected_nodes,expected_channels",
    [
        (healpy_layers.HealpyPool(1, pool_type="MAX"), 4, 3),
        (healpy_layers.HealpyPool(1, pool_type="AVG"), 4, 3),
        (healpy_layers.HealpyPseudoConv(1, 5), 4, 5),
        (healpy_layers.HealpyPseudoConv_Transpose(1, 5), 64, 5),
    ],
)
def test_healpy_layers_preserve_public_layout(public_layout_tensor, layer, expected_nodes, expected_channels):
    out = layer(public_layout_tensor)
    assert out.shape == (2, expected_nodes, expected_channels)


@pytest.mark.parametrize(
    "layer",
    [
        gnn_layers.Chebyshev(L=_laplacian(16), K=2, Fout=5),
        gnn_layers.Monomial(L=_laplacian(16), K=2, Fout=5),
        gnn_layers.Bernstein(L=_laplacian(16), K=2, Fout=5),
    ],
)
def test_graph_convolution_layers_preserve_public_layout(public_layout_tensor, layer):
    out = layer(public_layout_tensor)
    assert out.shape == (2, 16, 5)


def test_residual_layer_preserves_public_layout(public_layout_tensor):
    layer = gnn_layers.GCNN_ResidualLayer(
        layer_type="CHEBY", layer_kwargs={"L": _laplacian(16), "K": 2}, activation="relu"
    )
    out = layer(public_layout_tensor)
    assert out.shape == public_layout_tensor.shape


def test_transformer_layers_preserve_public_layout(public_layout_tensor):
    vit = gnn_transformers.Graph_ViT(p=2, key_dim=2, num_heads=2, n_layers=1)
    vit_out = vit(public_layout_tensor)
    assert vit_out.shape == (2, 1, 4)

    transformer = gnn_transformers.Graph_Transformer(A=sparse.eye(16, format="csc"), key_dim=2, num_heads=2, n_layers=1)
    transformer_out = transformer(public_layout_tensor)
    assert transformer_out.shape == (2, 16, 4)
