import numpy as np
from scipy import sparse
from scipy.sparse.linalg import eigsh
from scipy.special import comb
import torch
from torch import nn
from torch.nn import functional as F

from . import utils

_UNSUPPORTED_TF_KWARGS = {
    "regularizer",
    "kernel_regularizer",
    "bias_regularizer",
    "activity_regularizer",
    "kernel_constraint",
    "bias_constraint",
}


def _raise_for_unsupported_tf_kwargs(kwargs):
    unsupported = sorted(set(kwargs).intersection(_UNSUPPORTED_TF_KWARGS))
    if unsupported:
        raise TypeError(
            "Legacy Keras-style arguments are not supported by the PyTorch port: "
            f"{', '.join(unsupported)}. Use native PyTorch modules and apply regularization "
            "in the training loop/optimizer (for example optimizer weight_decay) or by "
            "adding explicit penalty terms to the loss."
        )


def _activation(activation):
    if activation is None:
        return None
    if callable(activation):
        return activation
    mapping = {"linear": lambda x: x, "relu": F.relu, "elu": F.elu}
    if activation in mapping:
        return mapping[activation]
    raise ValueError(f"Could not find activation <{activation}> in supported torch activations...")


def _to_scipy_csr(matrix):
    if sparse.issparse(matrix):
        return matrix.tocsr()
    if torch.is_tensor(matrix):
        matrix = matrix.detach().cpu().numpy()
    elif hasattr(matrix, "numpy"):
        matrix = matrix.numpy()
    return sparse.csr_matrix(matrix)


def _to_sparse_tensor(L, scale=1.0, dtype=torch.float32):
    """Convert a SciPy/NumPy sparse Laplacian to a coalesced torch sparse COO tensor."""
    L = _to_scipy_csr(L)
    lmax = 1.02 * eigsh(L, k=1, which="LM", return_eigenvectors=False)[0]
    L = utils.rescale_L(L, lmax=lmax, scale=scale).tocoo()
    indices = torch.as_tensor(np.vstack((L.row, L.col)), dtype=torch.long)
    values = torch.as_tensor(L.data, dtype=dtype)
    return torch.sparse_coo_tensor(indices, values, L.shape, dtype=dtype, check_invariants=False).coalesce()


def _sparse_mm(sparse_matrix, dense_matrix):
    if sparse_matrix.device != dense_matrix.device or sparse_matrix.dtype != dense_matrix.dtype:
        sparse_matrix = sparse_matrix.to(device=dense_matrix.device, dtype=dense_matrix.dtype)
    return torch.sparse.mm(sparse_matrix, dense_matrix)


def _init_tensor(shape, initializer, default_stddev, device=None, dtype=torch.float32):
    tensor = torch.empty(*shape, device=device, dtype=dtype)
    if initializer is None:
        nn.init.trunc_normal_(tensor, std=default_stddev)
    elif callable(initializer):
        try:
            value = initializer(shape=shape)
        except TypeError:
            try:
                value = initializer(shape)
            except TypeError:
                value = initializer(tensor)
        if isinstance(value, torch.Tensor):
            tensor = value.detach().clone().to(device=device, dtype=dtype)
        elif value is not None:
            tensor = torch.as_tensor(value, device=device, dtype=dtype)
    else:
        tensor = torch.as_tensor(initializer, device=device, dtype=dtype)
    return tensor


class NodeBatchNorm1d(nn.Module):
    """BatchNorm1d adapter for (batch, nodes, channels) tensors."""

    def __init__(self, num_features, momentum=0.1, eps=1e-5, affine=False):
        super().__init__()
        self.bn = nn.BatchNorm1d(num_features, eps=eps, momentum=momentum, affine=affine)

    def forward(self, x):
        return utils.channels_nodes_to_nodes_channels(self.bn(utils.nodes_channels_to_channels_nodes(x)))


class _GraphPolynomial(nn.Module):
    def __init__(self, L, K, Fout=None, initializer=None, activation=None, use_bias=False, use_bn=False, **kwargs):
        super().__init__()
        self.L_shape = tuple(_to_scipy_csr(L).shape)
        self.K = K
        self.Fout = Fout
        self.initializer = initializer
        self.activation = _activation(activation)
        self.use_bias = use_bias
        self.use_bn = use_bn
        self.kwargs = kwargs
        self.kernel = None
        self.bias = None
        self.bn = None

    def _build(self, Fin, Fout, kernel_shape, default_stddev, device=None, dtype=None):
        if self.kernel is not None:
            return
        dtype = dtype or torch.get_default_dtype()
        self.kernel = nn.Parameter(
            _init_tensor(kernel_shape, self.initializer, default_stddev, device=device, dtype=dtype)
        )
        if self.use_bias:
            self.bias = nn.Parameter(torch.zeros(1, 1, Fout, device=device, dtype=dtype))
        if self.use_bn:
            self.bn = NodeBatchNorm1d(Fout, momentum=0.1, eps=1e-5, affine=False).to(device=device, dtype=dtype)

    def _finalize(self, x, M, Fout):
        kernel = self.kernel.to(device=x.device, dtype=x.dtype)
        x = torch.matmul(x, kernel).reshape(-1, M, Fout)
        if self.bn is not None:
            self.bn.to(device=x.device, dtype=x.dtype)
            x = self.bn(x)
        if self.bias is not None:
            x = x + self.bias.to(device=x.device, dtype=x.dtype)
        if self.activation is not None:
            x = self.activation(x)
        return x


class Chebyshev(_GraphPolynomial):
    """A graph convolutional layer using the Chebyshev approximation."""

    def __init__(
        self,
        L,
        K,
        Fout=None,
        initializer=None,
        activation=None,
        use_bias=False,
        use_bn=False,
        n_matmul_splits=1,
        depth_wise=False,
        **kwargs,
    ):
        super().__init__(L, K, Fout, initializer, activation, use_bias, use_bn, **kwargs)
        self.n_matmul_splits = n_matmul_splits
        self.depth_wise = depth_wise
        self.register_buffer("sparse_L", _to_sparse_tensor(L, scale=0.75))

    def build(self, input_shape, device=None, dtype=None):
        Fin = int(input_shape[-1])
        Fout = Fin if self.Fout is None else self.Fout
        if self.depth_wise and Fout != Fin:
            raise AssertionError("For depthwise convolutions, Fout has to be None or equal to Fin")
        stddev = 1 / np.sqrt(Fin * (self.K + 0.5) / 2)
        shape = (Fin, self.K) if self.depth_wise else (self.K * Fin, Fout)
        self._build(Fin, Fout, shape, stddev, device=device, dtype=dtype)
        return self

    def forward(self, input_tensor):
        input_tensor = torch.as_tensor(input_tensor)
        N, M, Fin = input_tensor.shape
        Fout = Fin if self.Fout is None else self.Fout
        if self.depth_wise and Fout != Fin:
            raise AssertionError("For depthwise convolutions, Fout has to be None or equal to Fin")
        if self.kernel is None:
            self.build(input_tensor.shape, device=input_tensor.device, dtype=input_tensor.dtype)
        x0 = input_tensor.permute(1, 2, 0).reshape(M, -1)
        stack = [x0]
        if self.K > 1:
            x1 = _sparse_mm(self.sparse_L, x0)
            stack.append(x1)
        for _ in range(2, self.K):
            x2 = 2 * _sparse_mm(self.sparse_L, x1) - x0
            stack.append(x2)
            x0, x1 = x1, x2
        x = torch.stack(stack, dim=0).reshape(self.K, M, Fin, -1).permute(3, 1, 2, 0)
        if self.depth_wise:
            kernel = self.kernel.to(device=x.device, dtype=x.dtype)
            x = torch.einsum("ijkl,kl->ijk", x, kernel)
            if self.bn is not None:
                self.bn.to(device=x.device, dtype=x.dtype)
                x = self.bn(x)
            if self.bias is not None:
                x = x + self.bias.to(device=x.device, dtype=x.dtype)
            return self.activation(x) if self.activation is not None else x
        return self._finalize(x.reshape(-1, Fin * self.K), M, Fout)


class Monomial(_GraphPolynomial):
    """A graph convolutional layer using monomials."""

    def __init__(
        self,
        L,
        K,
        Fout=None,
        initializer=None,
        activation=None,
        use_bias=False,
        use_bn=False,
        n_matmul_splits=1,
        **kwargs,
    ):
        super().__init__(L, K, Fout, initializer, activation, use_bias, use_bn, **kwargs)
        self.n_matmul_splits = n_matmul_splits
        self.register_buffer("sparse_L", _to_sparse_tensor(L))

    def build(self, input_shape, device=None, dtype=None):
        Fin = int(input_shape[-1])
        Fout = Fin if self.Fout is None else self.Fout
        self._build(Fin, Fout, (self.K * Fin, Fout), 0.1, device=device, dtype=dtype)
        return self

    def forward(self, input_tensor):
        input_tensor = torch.as_tensor(input_tensor)
        N, M, Fin = input_tensor.shape
        Fout = Fin if self.Fout is None else self.Fout
        if self.kernel is None:
            self.build(input_tensor.shape, device=input_tensor.device, dtype=input_tensor.dtype)
        x0 = input_tensor.permute(1, 2, 0).reshape(M, -1)
        stack = [x0]
        for _ in range(1, self.K):
            x0 = _sparse_mm(self.sparse_L, x0)
            stack.append(x0)
        x = torch.stack(stack, dim=0).reshape(self.K, M, Fin, -1).permute(3, 1, 2, 0)
        return self._finalize(x.reshape(-1, Fin * self.K), M, Fout)


class GCNN_ResidualLayer(nn.Module):
    """A generic residual layer: in -> layer -> layer -> out + alpha*in."""

    def __init__(
        self,
        layer_type,
        layer_kwargs,
        activation=None,
        act_before=False,
        use_bn=False,
        norm_type="batch_norm",
        bn_kwargs=None,
        alpha=1.0,
    ):
        super().__init__()
        self.activation = _activation(activation)
        self.act_before = act_before
        self.use_bn = use_bn
        self.alpha = alpha
        layer_kwargs = dict(layer_kwargs)
        _raise_for_unsupported_tf_kwargs(layer_kwargs)
        if layer_type == "CHEBY":
            self.layer1 = Chebyshev(**layer_kwargs)
            self.layer2 = Chebyshev(**layer_kwargs)
        elif layer_type == "MONO":
            self.layer1 = Monomial(**layer_kwargs)
            self.layer2 = Monomial(**layer_kwargs)
        else:
            raise IOError(f"Layertype not understood: {layer_type}")
        self.bn1 = self.bn2 = None
        if use_bn:
            if norm_type not in {"layer_norm", "batch_norm"}:
                raise ValueError(f"norm_type <{norm_type}> not understood!")
            self.norm_type = norm_type

    def build(self, input_shape, device=None, dtype=None):
        batch, nodes, Fin = tuple(input_shape)
        self.layer1.build((batch, nodes, Fin), device=device, dtype=dtype)
        Fmid = Fin if self.layer1.Fout is None else self.layer1.Fout
        if self.use_bn:
            if self.norm_type == "layer_norm":
                self.bn1 = nn.LayerNorm((nodes, Fmid)).to(device=device, dtype=dtype)
            else:
                self.bn1 = NodeBatchNorm1d(Fmid).to(device=device, dtype=dtype)
        self.layer2.build((batch, nodes, Fmid), device=device, dtype=dtype)
        Fout = Fmid if self.layer2.Fout is None else self.layer2.Fout
        if self.use_bn:
            if self.norm_type == "layer_norm":
                self.bn2 = nn.LayerNorm((nodes, Fout)).to(device=device, dtype=dtype)
            else:
                self.bn2 = NodeBatchNorm1d(Fout).to(device=device, dtype=dtype)
        return self

    def _norm(self, name, x):
        module = getattr(self, name)
        if module is None:
            module = nn.LayerNorm(x.shape[1:]) if self.norm_type == "layer_norm" else NodeBatchNorm1d(x.shape[-1])
            module = module.to(device=x.device, dtype=x.dtype)
            setattr(self, name, module)
        module.to(device=x.device, dtype=x.dtype)
        return module(x)

    def forward(self, input_tensor):
        input_is_tensor = torch.is_tensor(input_tensor)
        input_tensor = torch.as_tensor(input_tensor)
        x = self.layer1(input_tensor)
        if self.use_bn:
            x = self._norm("bn1", x)
        x = self.layer2(x)
        if self.use_bn:
            x = self._norm("bn2", x)
        if self.activation is None:
            output = x + input_tensor
        else:
            output = (
                self.activation(x) + self.alpha * input_tensor
                if self.act_before
                else self.activation(x + self.alpha * input_tensor)
            )
        return output if input_is_tensor else output.detach()


class Bernstein(_GraphPolynomial):
    """A graph convolutional layer using the Bernstein approximation."""

    def __init__(
        self,
        L,
        K,
        Fout=None,
        initializer=None,
        activation=None,
        use_bias=False,
        use_bn=False,
        n_matmul_splits=1,
        **kwargs,
    ):
        super().__init__(L, K, Fout, initializer, activation, use_bias, use_bn, **kwargs)
        self.n_matmul_splits = n_matmul_splits
        self.register_buffer("sparse_L", _to_sparse_tensor(L, scale=0.75))

    def build(self, input_shape, device=None, dtype=None):
        Fin = int(input_shape[-1])
        Fout = Fin if self.Fout is None else self.Fout
        self._build(
            Fin,
            Fout,
            ((self.K + 1) * Fin, Fout),
            np.sqrt(6 / (Fin + Fout)),
            device=device,
            dtype=dtype,
        )
        return self

    def forward(self, input_tensor):
        input_tensor = torch.as_tensor(input_tensor)
        N, M, Fin = input_tensor.shape
        Fout = Fin if self.Fout is None else self.Fout
        if self.kernel is None:
            self.build(input_tensor.shape, device=input_tensor.device, dtype=input_tensor.dtype)
        x0 = input_tensor.permute(1, 2, 0).reshape(M, -1)
        stack = []
        for i in range(self.K + 1):
            x1 = x0
            theta = comb(self.K, i) / (2**self.K)
            for _ in range(i):
                x1 = _sparse_mm(self.sparse_L, x1)
            x2 = x1
            for _ in range(self.K - i):
                x2 = 2 * x2 - _sparse_mm(self.sparse_L, x2)
            stack.append(theta * x2)
        x = torch.stack(stack, dim=0).reshape(self.K + 1, M, Fin, -1).permute(3, 1, 2, 0)
        return self._finalize(x.reshape(-1, Fin * (self.K + 1)), M, Fout)
