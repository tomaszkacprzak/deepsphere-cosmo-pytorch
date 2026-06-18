import numpy as np
import torch
import healpy as hp

from pygsp.graphs import SphereHealpix

from deepsphere import gnn_transformers


def test_Graph_ViT():
    # create the input
    nside = 32
    n_pix = hp.nside2npix(nside)
    np.random.seed(11)
    m_in = np.random.normal(size=[3, n_pix, 7])

    # create the layer
    torch.manual_seed(11)
    p = 2
    key_dim = 16
    num_heads = 4
    graph_ViT = gnn_transformers.Graph_ViT(p=p, key_dim=key_dim, num_heads=num_heads, n_layers=3)
    output = graph_ViT(m_in)

    assert output.shape == (3,n_pix//4**p,num_heads*key_dim)

    # second eager call smoke test
    output = graph_ViT(m_in)

    assert output.shape == (3, n_pix // 4 ** p, num_heads * key_dim)


def test_Graph_Transformer():
    # create the input
    nside = 8
    n_pix = hp.nside2npix(nside)
    np.random.seed(11)
    m_in = np.random.normal(size=[3, n_pix, 7])
    A = SphereHealpix(subdivisions=8, nest=True, k=20, lap_type='normalized').A

    # create the layer
    torch.manual_seed(11)
    p = 2
    key_dim = 16
    num_heads = 4
    graph_ViT = gnn_transformers.Graph_Transformer(A=A, key_dim=key_dim, num_heads=num_heads, n_layers=3)
    output = graph_ViT(m_in)

    assert output.shape == (3, n_pix, num_heads*key_dim)

    # second eager call smoke test
    output = graph_ViT(m_in)

    assert output.numpy().shape == (3, n_pix, num_heads * key_dim)


def test_graph_transformer_sparse_indices_buffer_moves_with_model_device():
    import torch
    from scipy import sparse

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    A = sparse.csr_matrix(np.eye(4, dtype=np.float32))
    layer = gnn_transformers.Graph_Transformer(A=A, key_dim=2, num_heads=2, n_layers=1).to(device)
    assert "sparse_A_indices" in dict(layer.named_buffers())
    assert layer.sparse_A_indices.device == device
    assert layer.sparse_A_indices.dtype == torch.long
    out = layer(torch.randn(2, 4, 3, device=device))
    assert out.device == device
