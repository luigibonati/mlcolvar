import functools
import math
import torch
from torch import nn
import torch_scatter
from torch_geometric.nn import MessagePassing
from typing import Tuple, Callable, Optional, List, Dict

from mlcolvar.core.nn.graph.gnn import BaseGNN

"""
The Geometric Vector Perceptron (GVP) layer. This module is taken from:
https://github.com/chaitjo/geometric-gnn-dojo/blob/main/models/layers/py,
and made compilable.
"""

__all__ = ['GVPModel', 'GVPConvLayer', 'LayerNorm', 'Dropout']


class GVPModel(BaseGNN):
    """
    The Geometric Vector Perceptron (GVP) model [1, 2] with vector gate [2].

    Parameters
    ----------
    n_out: int
        Size of the output node features.
    cutoff: float
        Cutoff radius of the basis functions. Should be the same as the cutoff
        radius used to build the graphs.
    atomic_numbers: List[int]
        The atomic numbers mapping, e.g. the `atomic_numbers` attribute of a
        `mlcolvar.graph.data.GraphDataSet` instance.
    n_bases: int
        Size of the basis set.
    n_polynomials: bool
        Order of the polynomials in the basis functions.
    n_layers: int
        Number of the graph convolution layers.
    n_messages: int
        Number of GVPs to use in the message functions.
    n_feedforwards: int
        Number of GVPs to use in the feedforward functions.
    n_scalars_node: int
        Size of the scalar channel of the node embedding in hidden layers.
    n_vectors_node: int
        Size of the vector channel of the node embedding in hidden layers.
    n_scalars_edge: int
        Size of the scalar channel of the edge embedding in hidden layers.
    drop_rate: int
        Drop probability in all dropout layers.
    activation: str
        Name of the activation function to use in the GVPs (case sensitive).
    basis_type: str
        Type of the basis function.
    smooth: bool
        If use the smoothed GVPConv.

    References
    ----------
    .. [1] Jing, Bowen, et al.
           "Learning from protein structure with geometric vector perceptrons."
           International Conference on Learning Representations. 2020.
    .. [2] Jing, Bowen, et al.
           "Equivariant graph neural networks for 3d macromolecular structure."
           arXiv preprint arXiv:2106.03843 (2021).
    """
    def __init__(
        self,
        n_out: int,
        cutoff: float,
        atomic_numbers: List[int],
        n_bases: int = 8,
        n_polynomials: int = 6,
        n_layers: int = 1,
        n_messages: int = 2,
        n_feedforwards: int = 2,
        n_scalars_node: int = 8,
        n_vectors_node: int = 8,
        n_scalars_edge: int = 8,
        drop_rate: int = 0.1,
        activation: str = 'SiLU',
        basis_type: str = 'bessel',
        smooth: bool = False,
    ) -> None:
        super().__init__(
            n_out, cutoff, atomic_numbers, n_bases, n_polynomials, basis_type
        )

        self.W_e = nn.ModuleList([
            LayerNorm((n_bases, 1)),
            GVP(in_dims=(n_bases, 1),
                out_dims=(n_scalars_edge, 1),
                activations=(None, None),
                vector_gate=True
                )
        ])

        self.W_v = nn.ModuleList([
            LayerNorm((len(atomic_numbers), 0)),
            GVP(in_dims=(len(atomic_numbers), 0),
                out_dims=(n_scalars_node, n_vectors_node),
                activations=(None, None),
                vector_gate=True
                )
        ])

        self.layers = nn.ModuleList(
            GVPConvLayer(node_dims=(n_scalars_node, n_vectors_node),
                         edge_dims=(n_scalars_edge, 1),
                         n_message=n_messages,
                         n_feedforward=n_feedforwards,
                         drop_rate=drop_rate,
                         activations=(eval(f'torch.nn.{activation}')(), None),
                         vector_gate=True,
                         cutoff=(cutoff if smooth else -1)
                         )
            for _ in range(n_layers)
        )

        self.W_out = nn.ModuleList([
            LayerNorm((n_scalars_node, n_vectors_node)),
            GVP(in_dims=(n_scalars_node, n_vectors_node),
                out_dims=(n_out, 0),
                activations=(None, None),
                vector_gate=True)
        ])

    def forward(
        self, data: Dict[str, torch.Tensor], scatter_mean: bool = True
    ) -> torch.Tensor:
        """
        The forward pass.

        Parameters
        ----------
        data: Dict[str, torch.Tensor]
            The data dict. Usually came from the `to_dict` method of a
            `torch_geometric.data.Batch` object.
        scatter_mean: bool
            If perform the scatter mean to the model output.
        """
        h_V = (data['node_attrs'], None)
        for w in self.W_v:
            h_V = w(h_V)
        h_V_1, h_V_2 = h_V
        assert h_V_2 is not None
        h_V = (h_V_1, h_V_2)

        h_E = self.embed_edge(data)
        lengths = h_E[0]
        h_E = (h_E[1], h_E[2].unsqueeze(-2))
        for w in self.W_e:
            h_E = w(h_E)
        h_E_1, h_E_2 = h_E
        assert h_E_2 is not None
        h_E = (h_E_1, h_E_2)

        batch_id = data['batch']

        for layer in self.layers:
            h_V = layer(h_V, data['edge_index'], h_E, lengths)

        for w in self.W_out:
            h_V = w(h_V)
        out = h_V[0]

        if scatter_mean:
            if 'system_masks' not in data.keys():
                out = torch_scatter.scatter_mean(out, batch_id, dim=0)
            else:
                out = out * data['system_masks']
                out = torch_scatter.scatter_sum(out, batch_id, dim=0)
                out = out / data['n_system']

        return out


class GVP(nn.Module):
    """
    Geometric Vector Perceptron. See manuscript and README.md
    for more details.

    :param in_dims: tuple (n_scalar, n_vector)
    :param out_dims: tuple (n_scalar, n_vector)
    :param h_dim: intermediate number of vector channels, optional
    :param activations: tuple of functions (scalar_act, vector_act)
    :param vector_gate: whether to use vector gating.
           (vector_act will be used as sigma^+ in vector gating if `True`)
    """

    def __init__(
        self,
        in_dims: Tuple[int, Optional[int]],
        out_dims: Tuple[int, Optional[int]],
        h_dim: Tuple[int, Optional[int]] = None,
        activations: Tuple[
            Optional[Callable], Optional[Callable]
        ] = (nn.functional.relu, torch.sigmoid),
        vector_gate: bool = True,
    ) -> None:
        super(GVP, self).__init__()
        self.si, self.vi = in_dims
        self.so, self.vo = out_dims
        self.vector_gate = vector_gate
        if self.vi:
            self.h_dim = h_dim or max(self.vi, self.vo)
            self.wh = nn.Linear(self.vi, self.h_dim, bias=False)
            self.ws = nn.Linear(self.h_dim + self.si, self.so)
            if self.vo:
                self.wv = nn.Linear(self.h_dim, self.vo, bias=False)
                if self.vector_gate:
                    self.wsv = nn.Linear(self.so, self.vo)
            else:
                self.wv = None
                self.wsv = None
        else:
            self.wh = None
            self.wv = None
            self.wsv = None
            self.ws = nn.Linear(self.si, self.so)

        self.scalar_act, self.vector_act = activations
        self.dummy_param = nn.Parameter(torch.empty(0))

    def forward(
        self,
        x: Tuple[torch.Tensor, Optional[torch.Tensor]]
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        :param x: tuple (s, V) of `torch.Tensor`,
               or (if vectors_in is 0), a single `torch.Tensor`
        :return: tuple (s, V) of `torch.Tensor`,
               or (if vectors_out is 0), a single `torch.Tensor`
        """
        s, v = x
        if v is not None:
            assert self.wh is not None
            v = torch.transpose(v, -1, -2)
            vh = self.wh(v)
            vn = _norm_no_nan(vh, axis=-2)
            s = self.ws(torch.cat([s, vn], -1))
            if self.vo:
                assert self.wv is not None
                v = self.wv(vh)
                v = torch.transpose(v, -1, -2)
                if self.vector_gate:
                    assert self.wsv is not None
                    gate = (
                        self.wsv(self.vector_act(s))
                        if self.vector_act is not None
                        else self.wsv(s)
                    )
                    v = v * torch.sigmoid(gate).unsqueeze(-1)
                elif self.vector_act is not None:
                    v = v * self.vector_act(
                        _norm_no_nan(v, axis=-1, keepdims=True)
                    )
        else:
            s = self.ws(s)
            if self.vo:
                v = torch.zeros(
                    s.shape[0],
                    self.vo,
                    3,
                    device=self.dummy_param.device,
                    dtype=s.dtype
                )
            else:
                v = None

        if self.scalar_act is not None:
            s = self.scalar_act(s)

        return s, v


class GVPConv(MessagePassing):
    """
    Graph convolution / message passing with Geometric Vector Perceptrons.
    Takes in a graph with node and edge embeddings,
    and returns new node embeddings.

    This does NOT do residual updates and pointwise feedforward layers
    --- see `GVPConvLayer`.

    :param in_dims: input node embedding dimensions (n_scalar, n_vector)
    :param out_dims: output node embedding dimensions (n_scalar, n_vector)
    :param edge_dims: input edge embedding dimensions (n_scalar, n_vector)
    :param n_layers: number of GVPs in the message function
    :param activations:
           tuple of functions (scalar_act, vector_act) to use in GVPs
    :param vector_gate: whether to use vector gating.
           (vector_act will be used as sigma^+ in vector gating if `True`)
    """
    propagate_type = {
        's': torch.Tensor,
        'v': torch.Tensor,
        'edge_attr_s': torch.Tensor,
        'edge_attr_v': torch.Tensor,
        'edge_lengths': torch.Tensor,
    }

    def __init__(
        self,
        in_dims,
        out_dims,
        edge_dims,
        n_layers=3,
        aggr='mean',
        activations=(nn.functional.relu, torch.sigmoid),
        vector_gate=True,
        cutoff: float = -1.0,
    ) -> None:
        super(GVPConv, self).__init__(aggr=aggr)
        self.si, self.vi = in_dims
        self.so, self.vo = out_dims
        self.se, self.ve = edge_dims
        self.cutoff = cutoff

        GVP_ = functools.partial(
            GVP, activations=activations, vector_gate=vector_gate
        )

        self._module_list = torch.nn.ModuleList()
        if n_layers == 1:
            self._module_list.append(
                GVP_(in_dims=(2 * self.si + self.se, 2 * self.vi + self.ve),
                    out_dims=(self.so, self.vo),
                    activations=(None, None))
            )
        else:
            self._module_list.append(
                GVP_(in_dims=(2 * self.si + self.se, 2 * self.vi + self.ve),
                     out_dims=out_dims)
            )
            for i in range(n_layers - 2):
                self._module_list.append(GVP_(out_dims, out_dims))
            self._module_list.append(
                GVP_(in_dims=out_dims, 
                     out_dims=out_dims, 
                     activations=(None, None))
            )

    def forward(
        self,
        x: Tuple[torch.Tensor, torch.Tensor],
        edge_index: torch.Tensor,
        edge_attr: Tuple[torch.Tensor, torch.Tensor],
        edge_lengths: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        :param x: tuple (s, V) of `torch.Tensor`
        :param edge_index: array of shape [2, n_edges]
        :param edge_attr: tuple (s, V) of `torch.Tensor`
        """
        x_s, x_v = x
        assert x_v is not None
        message = self.propagate(
            edge_index,
            s=x_s,
            v=x_v.contiguous().view(x_v.shape[0], x_v.shape[1] * 3),
            edge_attr_s=edge_attr[0],
            edge_attr_v=edge_attr[1],
            edge_lengths=edge_lengths,
        )
        return _split(message, self.vo)

    def message(
        self,
        s_i: torch.Tensor,
        v_i: torch.Tensor,
        s_j: torch.Tensor,
        v_j: torch.Tensor,
        edge_attr_s: torch.Tensor,
        edge_attr_v: torch.Tensor,
        edge_lengths: torch.Tensor,
    ) -> torch.Tensor:
        assert edge_attr_s is not None
        assert edge_attr_v is not None
        v_j = v_j.view(v_j.shape[0], v_j.shape[1] // 3, 3)
        v_i = v_i.view(v_i.shape[0], v_i.shape[1] // 3, 3)
        message = _tuple_cat(
            (s_j, v_j), (edge_attr_s, edge_attr_v), (s_i, v_i)
        )
        message = self.message_func(message)
        message_merged = _merge(*message)
        if self.cutoff > 0:
            # apply SchNet-style cutoff function
            c = 0.5 * (torch.cos(edge_lengths * math.pi / self.cutoff) + 1.0)
            message_merged = message_merged * c.view(-1, 1)
        return message_merged

    def message_func(
        self, x: Tuple[torch.Tensor, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        for m in self._module_list:
            x = m(x)
        output_1, output_2 = x
        assert output_2 is not None
        return output_1, output_2


class GVPConvLayer(nn.Module):
    """
    Full graph convolution / message passing layer with
    Geometric Vector Perceptrons. Residually updates node embeddings with
    aggregated incoming messages, applies a pointwise feedforward
    network to node embeddings, and returns updated node embeddings.

    To only compute the aggregated messages, see `GVPConv`.

    :param node_dims: node embedding dimensions (n_scalar, n_vector)
    :param edge_dims: input edge embedding dimensions (n_scalar, n_vector)
    :param n_message: number of GVPs to use in message function
    :param n_feedforward: number of GVPs to use in feedforward function
    :param drop_rate: drop probability in all dropout layers
    :param activations:
           tuple of functions (scalar_act, vector_act) to use in GVPs
    :param vector_gate: whether to use vector gating.
           (vector_act will be used as sigma^+ in vector gating if `True`)
    """

    def __init__(
        self,
        node_dims,
        edge_dims,
        n_message=3,
        n_feedforward=2,
        drop_rate=0.1,
        activations=(nn.functional.relu, torch.sigmoid),
        vector_gate=True,
        residual=True,
        cutoff: float = -1.0,
    ) -> None:
        super(GVPConvLayer, self).__init__()
        self.conv = GVPConv(
            node_dims,
            node_dims,
            edge_dims,
            n_message,
            aggr='mean',
            activations=activations,
            vector_gate=vector_gate,
            cutoff=cutoff,
        )
        GVP_ = functools.partial(
            GVP, activations=activations, vector_gate=vector_gate
        )
        self.norm = nn.ModuleList([LayerNorm(node_dims) for _ in range(2)])
        self.dropout = nn.ModuleList([Dropout(drop_rate) for _ in range(2)])

        self._module_list = nn.ModuleList()
        if n_feedforward == 1:
            self._module_list.append(
                GVP_(in_dims=node_dims, 
                     out_dims=node_dims, 
                     activations=(None, None))
            )
        else:
            hid_dims = 4 * node_dims[0], 2 * node_dims[1]
            self._module_list.append(GVP_(node_dims, hid_dims))
            self._module_list.extend(
                GVP_(in_dims=hid_dims, out_dims=hid_dims) for _ in range(n_feedforward - 2)
            )
            self._module_list.append(
                GVP_(in_dims=hid_dims, out_dims=node_dims, activations=(None, None))
            )
        self.residual = residual

    def forward(
        self,
        x: Tuple[torch.Tensor, torch.Tensor],
        edge_index: torch.Tensor,
        edge_attr: Tuple[torch.Tensor, torch.Tensor],
        edge_lengths: torch.Tensor,
        node_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        :param x: tuple (s, V) of `torch.Tensor`
        :param edge_index: array of shape [2, n_edges]
        :param edge_attr: tuple (s, V) of `torch.Tensor`
        :param node_mask: array of type `bool` to index into the first
               dim of node embeddings (s, V). If not `None`, only
               these nodes will be updated.
        """

        dh = self.conv(x, edge_index, edge_attr, edge_lengths)

        x_ = x
        if node_mask is not None:
            x, dh = _tuple_index(x, node_mask), _tuple_index(dh, node_mask)

        if self.residual:
            input_1, input_2 = self.dropout[0](dh)
            assert input_2 is not None
            output_1, output_2 = self.norm[0](
                _tuple_sum(x, (input_1, input_2))
            )
            assert output_2 is not None
            x = (output_1, output_2)
        else:
            x = dh

        dh = self.ff_func(x)
        if self.residual:
            input_1, input_2 = self.dropout[1](dh)
            assert input_2 is not None
            output_1, output_2 = self.norm[1](
                _tuple_sum(x, (input_1, input_2))
            )
            assert output_2 is not None
            x = (output_1, output_2)
        else:
            x = dh

        if node_mask is not None:
            x_[0][node_mask], x_[1][node_mask] = x[0], x[1]
            x = x_
        return x

    def ff_func(
        self, x: Tuple[torch.Tensor, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        for m in self._module_list:
            x = m(x)
        output_1 = x[0]
        output_2 = x[1]
        assert output_2 is not None
        return output_1, output_2


class LayerNorm(nn.Module):
    """
    Combined LayerNorm for tuples (s, V).
    Takes tuples (s, V) as input and as output.
    """

    def __init__(self, dims) -> None:
        super(LayerNorm, self).__init__()
        self.s, self.v = dims
        self.scalar_norm = nn.LayerNorm(self.s)

    def forward(
        self,
        x: Tuple[torch.Tensor, Optional[torch.Tensor]]
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        :param x: tuple (s, V) of `torch.Tensor`,
               or single `torch.Tensor`
               (will be assumed to be scalar channels)
        """
        s, v = x
        if not self.v:
            return self.scalar_norm(s), None
        else:
            assert v is not None
            vn = _norm_no_nan(v, axis=-1, keepdims=True, sqrt=False)
            vn = torch.sqrt(torch.mean(vn, dim=-2, keepdim=True))
            return self.scalar_norm(s), v / vn


class Dropout(nn.Module):
    """
    Combined dropout for tuples (s, V).
    Takes tuples (s, V) as input and as output.
    """

    def __init__(self, drop_rate) -> None:
        super(Dropout, self).__init__()
        self.sdropout = nn.Dropout(drop_rate)
        self.vdropout = _VDropout(drop_rate)

    def forward(
        self,
        x: Tuple[torch.Tensor, Optional[torch.Tensor]]
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        :param x: tuple (s, V) of `torch.Tensor`,
               or single `torch.Tensor`
               (will be assumed to be scalar channels)
        """
        s, v = x
        if v is None:
            return self.sdropout(s), None
        else:
            assert v is not None
            return self.sdropout(s), self.vdropout(v)


class _VDropout(nn.Module):
    """
    Vector channel dropout where the elements of each
    vector channel are dropped together.
    """

    def __init__(self, drop_rate) -> None:
        super(_VDropout, self).__init__()
        self.drop_rate = drop_rate
        self.dummy_param = nn.Parameter(torch.empty(0))

    def forward(self, x) -> torch.Tensor:
        """
        :param x: `torch.Tensor` corresponding to vector channels
        """
        device = self.dummy_param.device
        if not self.training:
            return x
        mask = torch.bernoulli(
            (1 - self.drop_rate) * torch.ones(x.shape[:-1], device=device)
        ).unsqueeze(-1)
        x = mask * x / (1 - self.drop_rate)
        return x


def _tuple_sum(
    input_1: Tuple[torch.Tensor, torch.Tensor],
    input_2: Tuple[torch.Tensor, torch.Tensor]
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Sums any number of tuples (s, V) elementwise.
    """
    out = [i + j for i, j in zip(input_1, input_2)]
    return out[0], out[1]


@torch.jit.script
def _tuple_cat(
    input_1: Tuple[torch.Tensor, torch.Tensor],
    input_2: Tuple[torch.Tensor, torch.Tensor],
    input_3: Tuple[torch.Tensor, torch.Tensor],
    dim: int = -1
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Concatenates any number of tuples (s, V) elementwise.

    :param dim: dimension along which to concatenate when viewed
           as the `dim` index for the scalar-channel tensors.
           This means that `dim=-1` will be applied as
           `dim=-2` for the vector-channel tensors.
    """
    dim = int(dim % len(input_1[0].shape))
    s_args, v_args = list(zip(input_1, input_2, input_3))
    return torch.cat(s_args, dim=dim), torch.cat(v_args, dim=dim)


@torch.jit.script
def _tuple_index(
    x: Tuple[torch.Tensor, torch.Tensor], idx: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Indexes into a tuple (s, V) along the first dimension.

    :param idx: any object which can be used to index into a `torch.Tensor`
    """
    return x[0][idx], x[1][idx]


@torch.jit.script
def _norm_no_nan(
    x: torch.Tensor,
    axis: int = -1,
    keepdims: bool = False,
    eps: float = 1e-8,
    sqrt: bool = True
) -> torch.Tensor:
    """
    L2 norm of tensor clamped above a minimum value `eps`.

    :param sqrt: if `False`, returns the square of the L2 norm
    """
    out = torch.clamp(torch.sum(torch.square(x), axis, keepdims), min=eps)
    return torch.sqrt(out) if sqrt else out


@torch.jit.script
def _split(x: torch.Tensor, nv: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Splits a merged representation of (s, V) back into a tuple.
    Should be used only with `_merge(s, V)` and only if the tuple
    representation cannot be used.

    :param x: the `torch.Tensor` returned from `_merge`
    :param nv: the number of vector channels in the input to `_merge`
    """
    s = x[..., :-3 * nv]
    v = x[..., -3 * nv:].contiguous().view(x.shape[0], nv, 3)
    return s, v


@torch.jit.script
def _merge(s: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """
    Merges a tuple (s, V) into a single `torch.Tensor`, where the
    vector channels are flattened and appended to the scalar channels.
    Should be used only if the tuple representation cannot be used.
    Use `_split(x, nv)` to reverse.
    """
    v = v.contiguous().view(v.shape[0], v.shape[1] * 3)
    return torch.cat([s, v], -1)


def test_gvp() -> None:
    from mlcolvar.core.nn.graph.utils import _test_get_data
    torch.manual_seed(0)
    torch.set_default_dtype(torch.float64)

    model = GVPModel(
        n_out=2,
        cutoff=0.1,
        atomic_numbers=[1, 8],
        n_bases=6,
        n_polynomials=6,
        n_layers=2,
        n_messages=2,
        n_feedforwards=1,
        n_scalars_node=16,
        n_vectors_node=8,
        n_scalars_edge=16,
        drop_rate=0,
        activation='SiLU',
    )

    data = _test_get_data()
    assert (
        torch.abs(
            model(data) -
            torch.tensor([[0.6100070244145421, -0.2559670171962067]] * 6)
        ) < 1E-12
    ).all()
    torch.set_default_dtype(torch.float32)

if __name__ == '__main__':
    test_gvp()