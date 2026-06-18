import pytest
import numpy as np
import torch
import torch.nn.functional as F
import healpy as hp

from deepsphere import healpy_layers


def test_HealpyPool():
    # we get a random map to pool
    n_pix = hp.nside2npix(4)
    np.random.seed(11)
    m_in = np.random.normal(size=n_pix)

    # check exception
    with pytest.raises(IOError):
        avg_layer = healpy_layers.HealpyPool(0, pool_type="MAX")
    with pytest.raises(IOError):
        avg_layer = healpy_layers.HealpyPool(2, pool_type="HUHU")

    # we pool with healpy
    m_avg = hp.ud_grade(map_in=m_in, nside_out=2, order_in="NEST", order_out="NEST", power=None)

    # avg layer
    avg_layer = healpy_layers.HealpyPool(1, pool_type="AVG")
    m_avg_torch = avg_layer(m_in[None, :, None])

    assert np.all(np.abs(m_avg - m_avg_torch.detach().cpu().numpy().ravel()) < 1e-5)

    # maxpool normal
    m_max = np.max(m_in.reshape((n_pix//4, 4)), axis=1)

    # max layer
    max_layer = healpy_layers.HealpyPool(1, pool_type="MAX")
    m_max_torch = max_layer(m_in[None, :, None])

    assert np.all(np.abs(m_max - m_max_torch.detach().cpu().numpy().ravel()) < 1e-5)


def test_HealpyPseudoConv():
    # we get a random map to conv
    n_pix = hp.nside2npix(8)
    np.random.seed(11)
    m_in = np.random.normal(size=n_pix)

    # layer
    hp_conv = healpy_layers.HealpyPseudoConv(3, 5)
    m_conv_torch = hp_conv(m_in[None, :, None])

    assert m_conv_torch.shape == (1, n_pix//int(4**3), 5)


def test_HealpyPseudoConv_Transpose():
    # we get a random map to conv
    n_pix = hp.nside2npix(8)
    np.random.seed(11)
    m_in = np.random.normal(size=n_pix)

    # layer
    hp_conv = healpy_layers.HealpyPseudoConv_Transpose(3, 5)
    m_conv_torch = hp_conv(m_in[None, :, None])

    assert m_conv_torch.shape == (1, n_pix * int(4 ** 3), 5)


def test_HealpyChebyshev():
    # create the layer
    torch.manual_seed(11)
    L = torch.randn((3, 3), generator=torch.Generator().manual_seed(11))
    # make sym
    L = L @ L.T
    x = torch.randn((5, 3, 7), generator=torch.Generator().manual_seed(12))
    Fout = 3
    K = 4

    # create the layer
    stddev = 1 / np.sqrt(7 * (K + 0.5) / 2)
    initializer = lambda shape: torch.randn(shape, generator=torch.Generator().manual_seed(13)) * stddev
    cheb = healpy_layers.HealpyChebyshev(Fout=Fout, K=K, initializer=initializer)
    cheb = cheb._get_layer(L)
    new = cheb(x)

    cheb = healpy_layers.HealpyChebyshev(Fout=Fout, K=K, initializer=initializer, use_bn=True, use_bias=True)
    cheb = cheb._get_layer(L)
    new = cheb(x)


def test_HealpyMonomial():

    # create the layer
    torch.manual_seed(11)
    L = torch.randn((3, 3), generator=torch.Generator().manual_seed(11))
    # make sym
    L = L @ L.T
    x = torch.randn((5, 3, 7), generator=torch.Generator().manual_seed(12))
    Fout = 3
    K = 4

    # create the layer
    stddev = 0.1
    initializer = lambda shape: torch.randn(shape, generator=torch.Generator().manual_seed(13)) * stddev
    mon = healpy_layers.HealpyMonomial(Fout=Fout, K=K, initializer=initializer,
                                       activation="linear")
    mon = mon._get_layer(L)
    new = mon(x)

    mon = healpy_layers.HealpyMonomial(Fout=Fout, K=K, initializer=initializer,
                                       activation="linear",
                                       use_bias=True,
                                       use_bn=True)
    mon = mon._get_layer(L)
    new = mon(x)


def test_Healpy_ResidualLayer():
    # we get a random map
    n_pix = hp.nside2npix(4)
    np.random.seed(11)
    m_in = np.random.normal(size=[3, n_pix, 7])

    # layer definition
    layer_type = "CHEBY"
    layer_kwargs = {"K": 5,
                    "activation": "relu"}

    res_layer = healpy_layers.Healpy_ResidualLayer(layer_type=layer_type,
                                                   layer_kwargs=layer_kwargs,
                                                   activation=F.relu)
    res_layer = res_layer._get_layer(np.eye(n_pix, dtype=np.float64))
    out = res_layer(m_in)

    assert out.numpy().shape == (3, n_pix, 7)


def test_healpy_smoothing_buffers_move_with_model_device():
    import torch

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    nside = 1
    indices = np.arange(hp.nside2npix(nside))
    layer = healpy_layers.HealpySmoothing(nside=nside, indices=indices, sigma=1.0, arcmin=False).to(device)
    assert layer.indices.device == device
    assert layer.indices.dtype == torch.long
    assert layer.sparse_kernel.device == device
    assert layer.sparse_kernel.dtype == torch.float32
    assert layer.sparse_kernel.indices().dtype == torch.long
    out = layer(torch.randn(2, len(indices), 1, device=device))
    assert out.device == device
