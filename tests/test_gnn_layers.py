import pytest
import numpy as np
import torch
import healpy as hp

from deepsphere import gnn_layers


def _sym_laplacian():
    rng = np.random.default_rng(11)
    L = rng.normal(size=(3, 3))
    return L @ L.T


def _input():
    return torch.tensor(np.arange(30, dtype=np.float32).reshape(2, 3, 5) / 10)


def _constant_initializer(value):
    def init(shape):
        return torch.full(shape, value, dtype=torch.float32)

    return init


@pytest.mark.parametrize(
    "layer_cls,k_terms", [(gnn_layers.Chebyshev, 4), (gnn_layers.Monomial, 4), (gnn_layers.Bernstein, 5)]
)
def test_graph_layers_output_shapes_and_options(layer_cls, k_terms):
    L = _sym_laplacian()
    x = torch.randn(5, 3, 7, generator=torch.Generator().manual_seed(12))
    layer = layer_cls(L=L, Fout=3, K=4, initializer=_constant_initializer(0.1), activation="linear")
    out = layer(x)
    assert out.shape == (5, 3, 3)
    assert layer.kernel.shape == (k_terms * 7, 3)

    layer = layer_cls(
        L=L, Fout=3, K=4, initializer=_constant_initializer(0.1), activation="elu", use_bias=True, use_bn=True
    )
    out = layer(x)
    assert out.shape == (5, 3, 3)
    assert isinstance(layer.sparse_L, torch.Tensor)
    assert layer.sparse_L.is_sparse


def test_chebyshev_depthwise_shape():
    layer = gnn_layers.Chebyshev(L=_sym_laplacian(), K=3, initializer=_constant_initializer(0.25), depth_wise=True)
    out = layer(_input())
    assert out.shape == _input().shape
    assert layer.kernel.shape == (5, 3)


def test_monomial_known_numpy_fixture_with_identity_laplacian():
    x = torch.tensor([[[1.0, 2.0], [3.0, 4.0]]])
    layer = gnn_layers.Monomial(L=np.eye(2), K=2, Fout=1, initializer=_constant_initializer(1.0), activation="linear")
    out = layer(x).detach().cpu().numpy()

    L = layer.sparse_L.to_dense().detach().cpu().numpy()
    x0 = x.detach().cpu().numpy()[0]
    expected_basis = np.concatenate([x0, L @ x0], axis=1)
    expected = expected_basis.sum(axis=1, keepdims=True)[None, :, :]
    np.testing.assert_allclose(out, expected, rtol=1e-6, atol=1e-6)


def test_activations_known_values():
    assert torch.equal(gnn_layers._activation("linear")(torch.tensor([-1.0, 1.0])), torch.tensor([-1.0, 1.0]))
    assert torch.equal(gnn_layers._activation("relu")(torch.tensor([-1.0, 1.0])), torch.tensor([0.0, 1.0]))
    np.testing.assert_allclose(
        gnn_layers._activation("elu")(torch.tensor([-1.0])).detach().cpu().numpy(), np.array([-0.63212055])
    )
    with pytest.raises(ValueError):
        gnn_layers._activation("gelu")


def test_GCNN_ResidualLayer():
    n_pix = hp.nside2npix(4)
    rng = np.random.default_rng(11)
    m_in = torch.tensor(rng.normal(size=[3, n_pix, 7]), dtype=torch.float32)

    with pytest.raises(IOError):
        _ = gnn_layers.GCNN_ResidualLayer("juhu", dict())

    layer_kwargs = {"L": np.eye(n_pix, dtype=np.float64), "K": 5, "activation": "relu"}
    res_layer = gnn_layers.GCNN_ResidualLayer(layer_type="CHEBY", layer_kwargs=layer_kwargs, activation="relu")
    assert res_layer(m_in).detach().cpu().numpy().shape == (3, n_pix, 7)

    res_layer = gnn_layers.GCNN_ResidualLayer(
        layer_type="CHEBY", layer_kwargs=layer_kwargs, activation="relu", use_bn=True
    )
    assert res_layer(m_in).detach().cpu().numpy().shape == (3, n_pix, 7)

    res_layer = gnn_layers.GCNN_ResidualLayer(
        layer_type="CHEBY",
        layer_kwargs=layer_kwargs,
        activation="relu",
        use_bn=True,
        norm_type="layer_norm",
        bn_kwargs={"axis": (1, 2)},
    )
    assert res_layer(m_in).detach().cpu().numpy().shape == (3, n_pix, 7)

    with pytest.raises(ValueError):
        gnn_layers.GCNN_ResidualLayer(
            layer_type="CHEBY", layer_kwargs=layer_kwargs, activation="relu", use_bn=True, norm_type="moving_norm"
        )


def test_sparse_laplacian_buffer_moves_with_model_device():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    layer = gnn_layers.Chebyshev(L=_sym_laplacian(), K=2, Fout=2).to(device)
    assert "sparse_L" in dict(layer.named_buffers())
    assert layer.sparse_L.device == device
    assert layer.sparse_L.dtype == torch.float32
    assert layer.sparse_L.indices().dtype == torch.long
    x = torch.randn(2, 3, 4, device=device)
    out = layer(x)
    assert out.device == device


def test_lazy_graph_and_residual_parameters_follow_input_device_and_dtype():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float64
    x = torch.randn(2, 3, 4, device=device, dtype=dtype)

    layer = gnn_layers.Chebyshev(L=_sym_laplacian(), K=2, Fout=5, use_bias=True, use_bn=True).to(
        device=device, dtype=dtype
    )
    out = layer(x)
    assert out.device == device
    assert out.dtype == dtype
    assert layer.kernel.device == device
    assert layer.kernel.dtype == dtype
    assert layer.bias.device == device
    assert layer.bias.dtype == dtype

    res_layer = gnn_layers.GCNN_ResidualLayer(
        layer_type="CHEBY",
        layer_kwargs={"L": _sym_laplacian(), "K": 2},
        use_bn=True,
        norm_type="layer_norm",
    ).to(device=device, dtype=dtype)
    res_out = res_layer(x)
    assert res_out.device == device
    assert res_out.dtype == dtype
    assert res_layer.bn1.weight.device == device
    assert res_layer.bn1.weight.dtype == dtype
    assert res_layer.bn2.weight.device == device
    assert res_layer.bn2.weight.dtype == dtype
