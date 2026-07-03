"""PyTorch HEALPix graph-network container.

Examples
--------
Save and restore weights with the standard PyTorch API::

    torch.save(model.state_dict(), "healpy_gcnn.pt")

    restored = HealpyGCNN(nside=nside, indices=indices, layers=layers)
    restored.initialize((batch_size, n_pixels, n_features))  # one-time init for lazy layers
    restored.load_state_dict(torch.load("healpy_gcnn.pt", map_location="cpu"))

For detailed model summaries, install ``torchinfo`` and call::

    from torchinfo import summary
    summary(model, input_size=(batch_size, n_pixels, n_features))
"""

import math

import healpy as hp
import matplotlib.pyplot as plt
import numpy as np
import torch
from pygsp import filters
from pygsp.graphs import SphereHealpix
from torch import nn

from . import gnn_layers as gnn
from . import healpy_layers as hp_nn
from . import plot


class HealpyGCNN(nn.Module):
    """
    A graph convolutional network using PyTorch modules and the DeepSphere healpy layers.

    The public class name and constructor signature are preserved from the
    previous implementation. ``build(input_shape=...)`` is kept as a
    shape-validation/eager-initialization compatibility hook; normal PyTorch
    use can rely on eager construction and lazy initialization on the first
    ``forward`` pass.
    """

    def __init__(self, nside, indices, layers, n_neighbors=8, max_batch_size=None, initial_Fin=None):
        """
        Initializes a graph convolutional neural network using the healpy pixelization scheme
        :param nside: integeger, the nside of the input
        :param indices: 1d array of inidices, corresponding to the pixel ids of the input of the NN
        :param layers: a list of layers that will make up the neural network
        :param n_neighbors: Number of neighbors considered when building the graph, currently supported values are:
                            8 (default), 20, 40 and 60.
        :param max_batch_size: Maximal batch size this network is supposed to handle. This determines the number of
                                sparse-matrix multiplication splits, which are subsequently applied independent of the
                                actual batch size. Defaults to None, then no such precautions are taken, which may
                                cause an error.
        :param initial_Fin: Initial number of input features. Defaults to None.
        """
        super().__init__()

        print("WARNING: This network assumes that everything concerning healpy is in NEST ordering...", flush=True)

        if n_neighbors not in [8, 20, 40, 60]:
            raise NotImplementedError(
                f"The requested number of neighbors {n_neighbors} is nor supported. Choose either 8, 20, 40 or 60."
            )

        self.nside_in = nside
        self.indices_in = indices
        self.layers_in = list(layers)
        self.n_neighbors = n_neighbors
        self._initialized = False

        self.reduction_fac = 1.0
        for layer in self.layers_in:
            if isinstance(layer, (hp_nn.HealpyPool, hp_nn.HealpyPseudoConv, hp_nn.Healpy_ViT)):
                self.reduction_fac *= 2 ** (layer.p)
            if isinstance(layer, hp_nn.HealpyPseudoConv_Transpose):
                self.reduction_fac /= 2 ** (layer.p)

        self.nside_out = int(self.nside_in // self.reduction_fac)
        if self.nside_out < 1:
            raise ValueError(
                "With the given input, the layers would reduce the nside below zero!"
                "Use less layers that reduce the nside, e.g. HealpyPool or HealpyPseudoConv..."
            )
        if not hp.isnsideok(self.nside_out, nest=True):
            raise ValueError(f"The ouput of the network does not have a valid nside {self.nside_out}...")

        print(
            f"Detected a reduction factor of {self.reduction_fac}, the input with nside {self.nside_in} will be "
            f"transformed to {self.nside_out} during a forward pass. Checking for consistency with indices...",
            flush=True,
        )

        mask_in = np.zeros(hp.nside2npix(self.nside_in))
        mask_in[indices] = 1.0
        mask_out = hp.ud_grade(map_in=mask_in, nside_out=self.nside_out, order_in="NEST", order_out="NEST")
        mask_out[mask_out > 1e-12] = 1.0
        mask_in = hp.ud_grade(map_in=mask_out, nside_out=self.nside_in, order_in="NEST", order_out="NEST")
        transformed_indices = np.arange(hp.nside2npix(self.nside_in))[mask_in > 1e-12]

        if not np.all(np.sort(transformed_indices.astype(int)) == np.sort(self.indices_in.astype(int))):
            raise ValueError(
                "With the given indices it would not be possible to properly reduce the input maps "
                "with the reduction factor determined by the layers. Use the function "
                "<extend_indices> from utils with the determined minimal nside to make your set of "
                "indices compatible..."
            )
        print("indices seem consistent...", flush=True)

        # now we build the actual layers
        layers_use = []
        current_nside = self.nside_in
        current_indices = indices
        current_Fin = initial_Fin

        for layer in self.layers_in:
            if isinstance(
                layer,
                (
                    hp_nn.HealpyChebyshev,
                    hp_nn.HealpyMonomial,
                    hp_nn.Healpy_ResidualLayer,
                    hp_nn.Healpy_Transformer,
                    hp_nn.HealpyBernstein,
                ),
            ):
                sphere = SphereHealpix(
                    subdivisions=current_nside,
                    indexes=current_indices,
                    nest=True,
                    k=self.n_neighbors,
                    lap_type="normalized",
                )
                if isinstance(layer, hp_nn.Healpy_Transformer):
                    actual_layer = layer._get_layer(sphere.A)
                else:
                    if (max_batch_size is not None) and (current_Fin is not None):
                        n_edges = len(sphere.L.indices) if hasattr(sphere.L, "indices") else sphere.L.nnz
                        n_matmul_splits = max(1, math.ceil(max_batch_size * current_Fin * n_edges / 2**31))
                        while max_batch_size * current_Fin % n_matmul_splits != 0:
                            n_matmul_splits += 1
                        actual_layer = layer._get_layer(sphere.L, n_matmul_splits)
                    else:
                        actual_layer = layer._get_layer(sphere.L)
                layers_use.append(actual_layer)
            elif isinstance(layer, (hp_nn.HealpyPool, hp_nn.HealpyPseudoConv, hp_nn.Healpy_ViT)):
                new_nside = int(current_nside // 2**layer.p)
                current_indices = self._transform_indices(current_nside, new_nside, current_indices)
                current_nside = new_nside
                layers_use.append(layer)
            elif isinstance(layer, hp_nn.HealpyPseudoConv_Transpose):
                new_nside = int(current_nside * 2**layer.p)
                current_indices = self._transform_indices(current_nside, new_nside, current_indices)
                current_nside = new_nside
                layers_use.append(layer)
            else:
                if not isinstance(layer, nn.Module):
                    raise TypeError(f"Expected torch.nn.Module or HEALPix wrapper, got {type(layer)!r}.")
                layers_use.append(layer)

            if hasattr(layer, "Fout"):
                current_Fin = layer.Fout

        self.layers_use = nn.ModuleList(layers_use)

    def save_weights(self, *args, **kwargs):
        raise RuntimeError(
            "save_weights/load_weights are not available in the PyTorch port. "
            "use torch.save(model.state_dict(), path) and model.load_state_dict(torch.load(path))."
        )

    def load_weights(self, *args, **kwargs):
        raise RuntimeError(
            "save_weights/load_weights are not available in the PyTorch port. "
            "use torch.save(model.state_dict(), path) and model.load_state_dict(torch.load(path))."
        )

    def forward(self, x, training=None):
        """Run a PyTorch forward pass. ``training`` is accepted for migration compatibility."""
        if training is not None:
            self.train(bool(training))
        if not torch.is_tensor(x):
            x = torch.as_tensor(x, dtype=torch.get_default_dtype())
        for layer in self.layers_use:
            x = layer(x)
        return x

    call = forward

    def initialize(self, input_shape, device=None, dtype=None):
        """Materialize lazy parameters with a one-time dummy forward pass.

        This supports ``build(input_shape=...)`` expectations. Provide a
        full ``(batch, nodes, channels)`` input shape before creating an
        optimizer or before loading a saved ``state_dict`` into a fresh model.
        """
        if device is None:
            try:
                device = next(self.parameters()).device
            except StopIteration:
                try:
                    device = next(self.buffers()).device
                except StopIteration:
                    device = None
        if dtype is None:
            try:
                dtype = next(self.parameters()).dtype
            except StopIteration:
                try:
                    dtype = next(buffer for buffer in self.buffers() if buffer.is_floating_point()).dtype
                except StopIteration:
                    dtype = torch.get_default_dtype()
        dummy = torch.zeros(tuple(input_shape), device=device, dtype=dtype)
        was_training = self.training
        self.eval()
        with torch.no_grad():
            self(dummy)
        self.train(was_training)
        self._initialized = True
        return self

    def build(self, input_shape=None, **kwargs):
        """Materialize lazy parameters using a full ``(batch, nodes, channels)`` input shape."""
        if input_shape is None:
            input_shape = kwargs.get("input_shape")
        if input_shape is None:
            return self
        return self.initialize(input_shape)

    def summary(self, input_shape=None, line_length=120):
        """Print a lightweight summary; use ``torchinfo.summary`` for full details."""
        lines = [self.__class__.__name__, "=" * min(line_length, 80)]
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        for i, layer in enumerate(self.layers_use):
            params = sum(p.numel() for p in layer.parameters())
            lines.append(f"{i:>3}  {self._layer_name(layer):<32} {layer.__class__.__name__:<28} params={params}")
        lines.append(f"Total params: {total}")
        lines.append(f"Trainable params: {trainable}")
        text = "\n".join(lines)
        print(text)
        return text

    def _transform_indices(self, nside_in, nside_out, indices):
        if nside_in == nside_out:
            return indices
        mask_in = np.zeros(hp.nside2npix(nside_in))
        mask_in[indices] = 1.0
        mask_out = hp.ud_grade(map_in=mask_in, nside_out=nside_out, order_in="NEST", order_out="NEST")
        return np.arange(hp.nside2npix(nside_out))[mask_out > 1e-12]

    def _layer_name(self, layer):
        if hasattr(layer, "name") and layer.name:
            return layer.name
        if isinstance(layer, gnn.GCNN_ResidualLayer):
            return "gcnn__residual_layer"
        name = layer.__class__.__name__
        return "".join([("_" + c.lower()) if c.isupper() else c for c in name]).lstrip("_")

    def get_layer(self, index=None, name=None):
        if index is not None:
            return self.layers_use[index]
        if name is not None:
            for layer in self.layers_use:
                if self._layer_name(layer) == name:
                    return layer
            raise ValueError(f"No layer named {name!r}.")
        raise ValueError("Provide either index or name.")

    def _get_filter_coeffs(self, layer: gnn.Chebyshev, ind_in=None, ind_out=None):
        K, Fout = layer.K, layer.Fout
        if layer.kernel is None:
            raise RuntimeError(
                "Layer parameters are not initialized. Run a forward pass or initialize(input_shape) first."
            )
        trained_weights = layer.kernel.detach().cpu().numpy()
        if Fout is None:
            Fout = int(np.sqrt(np.prod(trained_weights.shape) // K))
        trained_weights = trained_weights.reshape((-1, K, Fout)).transpose([1, 2, 0])
        if ind_in is not None:
            trained_weights = trained_weights[:, :, ind_in]
        if ind_out is not None:
            trained_weights = trained_weights[:, ind_out, :]
        return trained_weights

    def get_gsp_filters(self, layer, ind_in=None, ind_out=None, return_weights=False):
        if isinstance(layer, int):
            torch_layer = self.get_layer(index=layer)
        elif isinstance(layer, str):
            torch_layer = self.get_layer(name=layer)
        else:
            raise ValueError("layer should be either string or int.")

        # check if the layer is actually the right type
        if isinstance(torch_layer, gnn.GCNN_ResidualLayer):
            if not (isinstance(torch_layer.layer1, gnn.Chebyshev) and isinstance(torch_layer.layer2, gnn.Chebyshev)):
                raise ValueError(
                    f"The requested layer ({layer}) is of type {type(torch_layer)}, but only "
                    f"Chebyshev5 or GCNN_ResidualLayer layers (with Chebyshev5 sublayers) "
                    f"are supported..."
                )
        elif not isinstance(torch_layer, gnn.Chebyshev):
            raise ValueError(
                f"The requested layer ({layer}) is of type {type(torch_layer)}, but only "
                f"Chebyshev5 or GCNN_ResidualLayer layers (with Chebyshev5 sublayers) "
                f"are supported..."
            )

        # we get the weights
        if isinstance(torch_layer, gnn.GCNN_ResidualLayer):
            # get the weights
            # print(torch_layer.layer1.kernel)
            weight1 = self._get_filter_coeffs(torch_layer.layer1, ind_in=ind_in, ind_out=ind_out)
            weight2 = self._get_filter_coeffs(torch_layer.layer2, ind_in=ind_in, ind_out=ind_out)
            weights = [weight1, weight2]

            # get the size of the features
            n_features = torch_layer.layer1.L_shape[0]

        else:
            # get the weights and reshape
            weight1 = self._get_filter_coeffs(torch_layer, ind_in=ind_in, ind_out=ind_out)
            weights = [weight1]
            # get the size of the features
            n_features = torch_layer.L_shape[0]

        if return_weights:
            return weights

        nside = len(self.indices_in) // n_features
        reduction_fac = 0
        while nside != 1:
            nside = nside // 4
            reduction_fac += 1
        nside = int(self.nside_in // 2 ** (reduction_fac))

        gsp_filters = []
        for weight in weights:
            pygsp_graph = SphereHealpix(
                subdivisions=nside,
                indexes=np.arange(hp.nside2npix(nside)),
                nest=True,
                k=self.n_neighbors,
                lap_type="normalized",
            )
            pygsp_graph.estimate_lmax()
            gsp_filters.append(filters.Chebyshev(pygsp_graph, weight))
        return gsp_filters

    def plot_chebyshev_coeffs(
        self, layer, ind_in=None, ind_out=None, ax=None, title="Chebyshev coefficients - layer {}"
    ):
        weights = self.get_gsp_filters(layer, ind_in, ind_out, return_weights=True)
        if ax is None:
            ax = plt.gca()
        for weight in weights:
            K, Fout, Fin = weight.shape
            ax.plot(weight.reshape((K, Fin * Fout)), ".")
            ax.set_title(title.format(layer))
        return ax

    def plot_filters_spectral(self, layer, ind_in=None, ind_out=None, ax=None, **kwargs):
        gsp_filters = self.get_gsp_filters(layer, ind_in=ind_in, ind_out=ind_out)
        if ax is None:
            ax = plt.gca()
        for gsp_filter in gsp_filters:
            gsp_filter.plot(sum=False, ax=ax, **kwargs)
        return ax

    def _filter_order(self, layer):
        torch_layer = self.get_layer(index=layer) if isinstance(layer, int) else self.get_layer(name=layer)
        return torch_layer.K if isinstance(torch_layer, gnn.Chebyshev) else torch_layer.layer1.K

    def plot_filters_section(self, layer, ind_in=None, ind_out=None, ax=None, **kwargs):
        gsp_filters = self.get_gsp_filters(layer, ind_in=ind_in, ind_out=ind_out)
        K = self._filter_order(layer)
        return [plot.plot_filters_section(gsp_filter, order=K, **kwargs) for gsp_filter in gsp_filters]

    def plot_filters_gnomonic(self, layer, ind_in=None, ind_out=None, **kwargs):
        gsp_filters = self.get_gsp_filters(layer, ind_in=ind_in, ind_out=ind_out)
        K = self._filter_order(layer)
        return [plot.plot_filters_gnomonic(gsp_filter, order=K, **kwargs) for gsp_filter in gsp_filters]
