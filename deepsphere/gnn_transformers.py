import math

import numpy as np
import torch
from scipy import sparse
from torch import nn

from .utils import channels_nodes_to_nodes_channels, nodes_channels_to_channels_nodes

# Helper Functions
##################


def _as_tensor(inputs):
    """Convert numpy-like inputs to tensors while leaving tensors untouched."""
    if torch.is_tensor(inputs):
        return inputs
    return torch.as_tensor(inputs, dtype=torch.get_default_dtype())


def _to_scipy_coo(matrix):
    if sparse.issparse(matrix):
        return matrix.tocoo()
    if torch.is_tensor(matrix):
        matrix = matrix.detach().cpu().numpy()
    elif hasattr(matrix, "numpy"):
        matrix = matrix.numpy()
    return sparse.csr_matrix(matrix).tocoo()


def _activation_module(activation):
    if activation is None or activation == "linear":
        return nn.Identity()
    if isinstance(activation, nn.Module):
        return activation
    if callable(activation) and not isinstance(activation, str):

        class _CallableActivation(nn.Module):
            def forward(self, x):
                return activation(x)

        return _CallableActivation()
    name = str(activation).lower()
    if name == "relu":
        return nn.ReLU()
    if name == "gelu":
        return nn.GELU()
    if name == "tanh":
        return nn.Tanh()
    if name == "sigmoid":
        return nn.Sigmoid()
    raise ValueError(f"Unsupported activation: {activation!r}")


def scaled_dot_product_attention(q, k, v, mask=None):
    """Calculate dense scaled dot-product attention with PyTorch tensors.

    ``q``, ``k`` and ``v`` are expected to have shape
    ``(..., seq_len, depth)``. ``mask`` follows standard PyTorch masking
    semantics: boolean masks mark entries to suppress, while floating masks are
    added to the logits (for example, use ``-inf`` for masked logits). The mask
    must be broadcastable to ``(..., seq_len_q, seq_len_k)``.
    """
    matmul_qk = torch.matmul(q, k.transpose(-2, -1))
    scaled_attention_logits = matmul_qk / math.sqrt(k.shape[-1])

    if mask is not None:
        mask = mask.to(device=scaled_attention_logits.device)
        if mask.dtype == torch.bool:
            scaled_attention_logits = scaled_attention_logits.masked_fill(mask, float("-inf"))
        else:
            scaled_attention_logits = scaled_attention_logits + mask.to(dtype=scaled_attention_logits.dtype)

    attention_weights = torch.softmax(scaled_attention_logits, dim=-1)
    output = torch.matmul(attention_weights, v)
    return output, attention_weights


def _segment_softmax(values, segment_ids, num_segments):
    """Softmax over entries sharing the same segment id.

    This local helper uses ``scatter_reduce_``/``scatter_add_`` so the sparse
    attention path does not require the optional ``torch-scatter`` package.
    """
    if values.numel() == 0:
        return values

    expand_shape = (num_segments,) + values.shape[1:]
    index = segment_ids.view(-1, *([1] * (values.dim() - 1))).expand_as(values)

    maxima = torch.full(expand_shape, -torch.inf, dtype=values.dtype, device=values.device)
    maxima.scatter_reduce_(0, index, values, reduce="amax", include_self=True)

    shifted = values - maxima.index_select(0, segment_ids)
    exp_values = torch.exp(shifted)

    denom = torch.zeros(expand_shape, dtype=values.dtype, device=values.device)
    denom.scatter_add_(0, index, exp_values)
    return exp_values / denom.index_select(0, segment_ids).clamp_min(torch.finfo(values.dtype).tiny)


def scaled_dot_product_sparse_attention(q, k, v, mask):
    """Calculate sparse scaled dot-product attention from adjacency indices.

    ``q``, ``k`` and ``v`` have shape ``(batch, heads, nodes, depth)``. ``mask``
    is a two-column tensor of ``(query_node, key_node)`` sparse adjacency
    indices. A grouped softmax over each query node is implemented with native
    PyTorch scatter operations.
    """
    if mask.numel() == 0:
        return torch.zeros_like(q)

    mask = mask.to(device=q.device, dtype=torch.long)
    row = mask[:, 0]
    col = mask[:, 1]
    num_nodes = q.shape[-2]

    q_part = q.index_select(2, row).permute(2, 0, 1, 3)
    k_part = k.index_select(2, col).permute(2, 0, 1, 3)
    logits = (q_part * k_part).sum(dim=-1, keepdim=True) / math.sqrt(k.shape[-1])

    weights = _segment_softmax(logits, row, num_nodes)
    v_part = v.index_select(2, col).permute(2, 0, 1, 3)
    weighted_values = v_part * weights

    output_seq_first = torch.zeros(
        (num_nodes,) + weighted_values.shape[1:], dtype=weighted_values.dtype, device=weighted_values.device
    )
    index = row.view(-1, 1, 1, 1).expand_as(weighted_values)
    output_seq_first.scatter_add_(0, index, weighted_values)
    return output_seq_first.permute(1, 2, 0, 3)


# Layers
########


class AddPositionEmbs(nn.Module):
    """Adds learned positional embeddings to the inputs."""

    def __init__(self, posemb_init=None, **kwargs):
        super().__init__()
        self.posemb_init = posemb_init
        self.pos_embedding = None

    def build(self, inputs_shape, device=None, dtype=None):
        pos_emb_shape = (1, int(inputs_shape[1]), int(inputs_shape[2]))
        self.pos_embedding = nn.Parameter(torch.empty(pos_emb_shape, device=device, dtype=dtype))
        if self.posemb_init is not None:
            with torch.no_grad():
                initialized = self.posemb_init(self.pos_embedding)
                if initialized is not None:
                    self.pos_embedding.copy_(initialized)
        else:
            nn.init.trunc_normal_(self.pos_embedding, std=0.02)

    def forward(self, inputs):
        if self.pos_embedding is None:
            self.build(inputs.shape, device=inputs.device, dtype=inputs.dtype)
        return inputs + self.pos_embedding.to(dtype=inputs.dtype, device=inputs.device)

    call = forward


class MultiHeadAttention(nn.Module):
    """A simple multi-head attention layer followed by a single-layer MLP."""

    def __init__(self, d_model, num_heads, use_norm=True, activation="relu", sparse_A_indices=None):
        super().__init__()
        self.num_heads = num_heads
        self.d_model = d_model
        self.use_norm = use_norm
        if sparse_A_indices is not None:
            self.register_buffer("sparse_A_indices", torch.as_tensor(sparse_A_indices, dtype=torch.long))
        else:
            self.register_buffer("sparse_A_indices", None)

        assert d_model % self.num_heads == 0
        self.depth = d_model // self.num_heads

        self.wq = nn.Linear(d_model, d_model)
        self.wk = nn.Linear(d_model, d_model)
        self.wv = nn.Linear(d_model, d_model)

        self.layer_norm1 = nn.LayerNorm(d_model) if self.use_norm else nn.Identity()
        self.layer_norm2 = nn.LayerNorm(d_model) if self.use_norm else nn.Identity()
        self.activation_layer = _activation_module(activation)
        self.dense = nn.Linear(d_model, d_model)

    def split_heads(self, x, batch_size):
        x = x.reshape(batch_size, -1, self.num_heads, self.depth)
        return x.permute(0, 2, 1, 3)

    def forward(self, inputs, mask=None):
        inputs = _as_tensor(inputs).to(next(self.parameters()).device)
        batch_size = inputs.shape[0]
        inputs = self.layer_norm1(inputs)

        q = self.split_heads(self.wq(inputs), batch_size)
        k = self.split_heads(self.wk(inputs), batch_size)
        v = self.split_heads(self.wv(inputs), batch_size)

        if self.sparse_A_indices is None:
            scaled_attention, _ = scaled_dot_product_attention(q, k, v, mask)
        else:
            scaled_attention = scaled_dot_product_sparse_attention(q, k, v, self.sparse_A_indices)

        scaled_attention = scaled_attention.permute(0, 2, 1, 3)
        concat_attention = scaled_attention.reshape(batch_size, -1, self.d_model)
        concat_attention = inputs + concat_attention
        output = self.layer_norm2(concat_attention)
        output = self.dense(output)
        output = self.activation_layer(output)
        return output + concat_attention

    call = forward


class Graph_ViT(nn.Module):
    """A visual transformer layer for (healpy) graphs."""

    def __init__(self, p, key_dim, num_heads, positional_encoding=True, n_layers=1, activation="relu", layer_norm=True):
        super().__init__()
        if not p > 1:
            raise IOError("The super pixel size factor p has to be at least 1!")
        else:
            print(f"Every patch consists of {4**p} HEALPix pixels")

        self.p = p
        self.embed_filter_size = int(4**p)
        self.key_dim = key_dim
        self.num_heads = num_heads
        self.embedding_size = self.key_dim * self.num_heads
        self.positional_encoding = positional_encoding
        self.n_layers = n_layers
        self.activation = activation
        self.layer_norm = layer_norm

        self.embed = nn.LazyConv1d(
            self.embedding_size, kernel_size=self.embed_filter_size, stride=self.embed_filter_size, padding=0
        )
        self.pos_encoder = AddPositionEmbs() if self.positional_encoding else None
        assert n_layers >= 1, "Number of attention layers should be at least 1"
        self.mha_layers = nn.ModuleList(
            [
                MultiHeadAttention(
                    d_model=self.embedding_size,
                    num_heads=self.num_heads,
                    use_norm=self.layer_norm,
                    activation=self.activation,
                )
                for _ in range(n_layers)
            ]
        )

    def build(self, input_shape):
        n_nodes = int(input_shape[1])
        if n_nodes % self.embed_filter_size != 0:
            raise IOError(
                f"Input shape {input_shape} not compatible with the embedding filter size {self.embed_filter_size}"
            )

    def forward(self, inputs):
        inputs = _as_tensor(inputs)
        if inputs.shape[1] % self.embed_filter_size != 0:
            raise IOError(
                f"Input shape {tuple(inputs.shape)} not compatible with the embedding filter size {self.embed_filter_size}"
            )
        x = channels_nodes_to_nodes_channels(self.embed(nodes_channels_to_channels_nodes(inputs)))
        if self.pos_encoder is not None:
            x = self.pos_encoder(x)
        for mha in self.mha_layers:
            x = mha(x)
        return x

    call = forward


class Graph_Transformer(nn.Module):
    """A graph transformer layer that takes edge information from an adjacency matrix."""

    def __init__(self, A, key_dim, num_heads, positional_encoding=True, n_layers=1, activation="relu", layer_norm=True):
        super().__init__()
        self.A_shape = tuple(_to_scipy_coo(A).shape)
        self.key_dim = key_dim
        self.num_heads = num_heads
        self.embedding_size = self.key_dim * self.num_heads
        self.positional_encoding = positional_encoding
        self.n_layers = n_layers
        self.activation = activation
        self.layer_norm = layer_norm

        sparse_A = _to_scipy_coo(A)
        sparse_A_indices = np.stack([sparse_A.row, sparse_A.col], axis=1)
        self.register_buffer("sparse_A_indices", torch.as_tensor(sparse_A_indices, dtype=torch.long))

        self.embed = nn.LazyLinear(self.embedding_size)
        self.pos_encoder = AddPositionEmbs() if self.positional_encoding else None
        assert n_layers >= 1, "Number of attention layers should be at least 1"
        self.mha_layers = nn.ModuleList(
            [
                MultiHeadAttention(
                    d_model=self.embedding_size,
                    num_heads=self.num_heads,
                    use_norm=self.layer_norm,
                    activation=self.activation,
                    sparse_A_indices=self.sparse_A_indices,
                )
                for _ in range(n_layers)
            ]
        )

    def build(self, input_shape):
        return None

    def forward(self, inputs):
        x = self.embed(_as_tensor(inputs))
        if self.pos_encoder is not None:
            x = self.pos_encoder(x)
        for mha in self.mha_layers:
            x = mha(x)
        return x

    call = forward
