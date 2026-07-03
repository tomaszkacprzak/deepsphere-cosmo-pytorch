import torch

from deepsphere import healpy_layers
from examples.map_only_regression import build_map_only_regressor


def test_healpy_pseudoconv_accepts_activation():
    layer = healpy_layers.HealpyPseudoConv(p=1, Fout=3, activation=torch.relu)
    x = -torch.ones((2, 4, 1))

    y = layer(x)

    assert y.shape == (2, 1, 3)
    assert torch.all(y >= 0)


def test_map_only_regression_translation_forward_small():
    model, x_batch = build_map_only_regressor(
        n_side=32,
        indices=range(12 * 32**2),
        batch_size=2,
        n_channels=2,
        out_features=6,
        base_channels=4,
        residual_layers=1,
        n_neighbors=8,
    )

    y_pred = model(x_batch)

    assert y_pred.shape == (2, 6)
