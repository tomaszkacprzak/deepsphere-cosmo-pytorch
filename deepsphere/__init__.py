__version__ = "0.2"

import torch

# Explicitly opt out of PyTorch sparse invariant checks. This keeps the
# current performance-oriented behavior while silencing PyTorch warnings about
# the implicit default.
torch.sparse.check_sparse_tensor_invariants.disable()

from deepsphere.gnn_layers import Bernstein, Chebyshev, GCNN_ResidualLayer, Monomial
from deepsphere.healpy_layers import (
    HealpyBernstein,
    HealpyChebyshev,
    HealpyMonomial,
    HealpyPool,
    HealpyPseudoConv,
    HealpyPseudoConv_Transpose,
    Healpy_ResidualLayer,
    Healpy_Transformer,
    Healpy_ViT,
)
from deepsphere.healpy_networks import HealpyGCNN

__all__ = [
    "Bernstein",
    "Chebyshev",
    "GCNN_ResidualLayer",
    "HealpyBernstein",
    "HealpyChebyshev",
    "HealpyGCNN",
    "HealpyMonomial",
    "HealpyPool",
    "HealpyPseudoConv",
    "HealpyPseudoConv_Transpose",
    "Healpy_ResidualLayer",
    "Healpy_Transformer",
    "Healpy_ViT",
    "Monomial",
]
