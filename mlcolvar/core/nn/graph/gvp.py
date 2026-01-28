import functools
import math
import torch
from torch import nn
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
        pooling_operation : str = 'mean',
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
        """Initializes a Geometric Vector Perceptron (GVP) model.

        Parameters
        ----------
        n_out: int
            Number of the output scalar node features.
        cutoff: float
            Cutoff radius of the basis functions. Should be the same as the cutoff
            radius used to build the graphs.
        atomic_numbers: List[int]
            The atomic numbers mapping
        pooling_operation : str
            Type of pooling operation to combine node-level features into graph-level features, either mean or sum, by default 'mean'
        n_bases: int
            Size of the basis set used for the embedding, by default 8.
        n_polynomials: bool
            Order of the polynomials in the basis functions, by default 6.
        n_layers: int
            Number of the graph convolution layers, by default 1.
        n_messages: int
            Number of GVP layers to be used in the message functions, by default 2.
        n_feedforwards: int
            Number of GVP layers to be used in the feedforward functions, by default 2.
        n_scalars_node: int
            Size of the scalar channel of the node embedding in hidden layers, by default 8.
        n_vectors_node: int
            Size of the vector channel of the node embedding in hidden layers, by default 8.
        n_scalars_edge: int
            Size of the scalar channel of the edge embedding in hidden layers, by default 8.
        drop_rate: int
            Drop probability in all dropout layers, by default 0.1.
        activation: str
            Name of the activation function to be used in the GVPs (case sensitive), by default SiLU.
        basis_type: str
            Type of the basis function, by default bessel.
        smooth: bool
            If use the smoothed GVPConv, by default False.
        """
        super().__init__(
            n_out=n_out, 
            cutoff=cutoff, 
            atomic_numbers=atomic_numbers, 
            pooling_operation=pooling_operation, 
            n_bases=n_bases, 
            n_polynomials=n_polynomials, 
            basis_type=basis_type
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
        self, data: Dict[str, torch.Tensor], pool: bool = True
    ) -> torch.Tensor:
        """The forward pass.

        Parameters
        ----------
        data: Dict[str, torch.Tensor]
            The data dict. Usually came from the `to_dict` method of a
            `torch_geometric.data.Batch` object.
        pool: bool
            If perform the pooling to the model output, by default True.
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

        for layer in self.layers:
            h_V = layer(h_V, data['edge_index'], h_E, lengths)

        for w in self.W_out:
            h_V = w(h_V)
        out = h_V[0]

        if pool:
            out = self.pooling(input=out, data=data)

        return out


class GVP(nn.Module):
    """
    Geometric Vector Perceptron (GVP) layer from [1, 2] with vector gate [2].

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
        in_dims: Tuple[int, Optional[int]],
        out_dims: Tuple[int, Optional[int]],
        h_dim: Tuple[int, Optional[int]] = None,
        activations: Tuple[
            Optional[Callable], Optional[Callable]
        ] = (nn.functional.relu, torch.sigmoid),
        vector_gate: bool = True,
    ) -> None:
        r"""Geometric Vector Perceptron layer.

        Updates the scalar node feature s as:
        .. math:: bm{s}^n \leftarrow \sigma \left(\bm{s}'\right) \quad\text{with}\quad  \bm{s}' \coloneq \bm{W}_m \left[{\|\bm{W}_h\vec{\bm{v}}^o\|_2 \atop \bm{s}^o}\right] + \bm{b}
        
        And the vector nore features as:
        .. math:: \vec{\bm{v}}^n \leftarrow \sigma_g \left(\bm{W}_g\left(\sigma^+ \left(\bm{s}'\right)\right) + \bm{b}_g \right) \odot \bm{W}_\mu\bm{W}_h\vec{\bm{v}}^o

        Parameters
        ----------
        in_dims : Tuple[int, Optional[int]]
            Dimension of inputs
        out_dims : Tuple[int, Optional[int]]
            Dimension of outputs
        h_dim : Tuple[int, Optional[int]], optional
            Intermidiate number of vector channels, by default None
        activations : Tuple[ Optional[Callable], Optional[Callable] ], optional
            Scalar and vector activation functions (scalar_act, vector_act), by default (nn.functional.relu, torch.sigmoid)
        vector_gate : bool, optional
            Whether to use vector gating, by default True. The vector activation will be used as sigma^+ in vector gating if `True`
        """
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
        """Forward pass of GVP

        Parameters
        ----------
        x : Tuple[torch.Tensor, Optional[torch.Tensor]]
            Input scalar and vector node embeddings

        Returns
        -------
        Tuple[torch.Tensor, Optional[torch.Tensor]]
            Input scalar and vector node embeddings
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
        """Graph convolution / message passing with Geometric Vector Perceptrons.
        Takes in a graph with node and edge embeddings,
        and returns new node embeddings.

        This does NOT do residual updates and pointwise feedforward layers
        --- see `GVPConvLayer` instead.

        Parameters
        ----------
        in_dims :
            input node embedding dimensions (n_scalar, n_vector)
        out_dims :
            output node embedding dimensions (n_scalar, n_vector)
        edge_dims :
            input edge embedding dimensions (n_scalar, n_vector)
        n_layers : int, optional
            number of GVPs in the message function, by default 3
        aggr : str, optional
            Type of message aggregate function, by default 'mean'
        activations : tuple, optional
            activation functions (scalar_act, vector_act) to be used use in GVPs, by default (nn.functional.relu, torch.sigmoid)
        vector_gate : bool, optional
            Whether to use vector gating, by default True. The vector activation will be used as sigma^+ in vector gating if `True`
        cutoff : float, optional
            Radial cutoff, by default -1.0
        """
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
        """Forward pass of GVPConv

        Parameters
        ----------
        x : Tuple[torch.Tensor, torch.Tensor]
            Input scalar and vector node embeddings
        edge_index : torch.Tensor
            Index of edge sources and destinations
        edge_attr : Tuple[torch.Tensor, torch.Tensor]
            Edge attributes
        edge_lengths : torch.Tensor
            Edge lengths

        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor]
            Output scalar and vector node embeddings
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
    Geometric Vector Perceptrons. 
    Residually updates node embeddings with
    aggregated incoming messages, applies a pointwise feedforward
    network to node embeddings, and returns updated node embeddings.

    To only compute the aggregated messages, see `GVPConv`.
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
        """Full graph convolution / message passing layer with
        Geometric Vector Perceptrons. 
        Residually updates node embeddings with
        aggregated incoming messages, applies a pointwise feedforward
        network to node embeddings, and returns updated node embeddings.

        To only compute the aggregated messages see `GVPConv` instead.

        Parameters
        ----------
        node_dims : 
            node embedding dimensions (n_scalar, n_vector)
        edge_dims : 
            input edge embedding dimensions (n_scalar, n_vector)
        n_message : int, optional
            number of GVP layers to be used in message function, by default 3
        n_feedforward : int, optional
            number of GVPs to be used use in feedforward function, by default 2
        drop_rate : float, optional
            drop probability in all dropout layers, by default 0.1
        activations : tuple, optional
            activation functions (scalar_act, vector_act) to be used use in GVPs, by default (nn.functional.relu, torch.sigmoid)
        vector_gate : bool, optional
            whether to use vector gating, by default True. The vector activation will be used as sigma^+ in vector gating if `True`
        residual : bool, optional
            whether to perform the update residually, by default True
        cutoff : float, optional
            radial cutoff, by default -1.0
        """
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
        """Forward pass of GVPConvLayer
    
        Parameters
        ----------
        x : Tuple[torch.Tensor, torch.Tensor]
            Input scalar and vector node embeddings
        edge_index : torch.Tensor
            Index of edge sources and destinations
        edge_attr : Tuple[torch.Tensor, torch.Tensor]
            Edge attributes
        edge_lengths : torch.Tensor
            Edge lengths
        node_mask : Optional[torch.Tensor], optional
            Mask to restrict the node update to a subset. 
            It should be a tensor of type `bool` to index the first dim of node embeddings (s, V), by default None. 
            If not `None`, only the selected nodes will be updated.

        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor]
            Output scalar and vector node embeddings
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
        """Forward pass of LayerNorm

        Parameters
        ----------
        x : Tuple[torch.Tensor, Optional[torch.Tensor]]
            Input channels, if a single tensor is provided it assumes it to be the scalar channel

        Returns
        -------
        Tuple[torch.Tensor, Optional[torch.Tensor]]
            Normalized channels
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
        """Forward pass of Dropout

        Parameters
        ----------
        x : Tuple[torch.Tensor, Optional[torch.Tensor]]
            Input channels, if a single tensor is provided it assumes it to be the scalar channel

        Returns
        -------
        Tuple[torch.Tensor, Optional[torch.Tensor]]
            Dropped out channels
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

    def forward(self, x : torch.Tensor) -> torch.Tensor:
        """Forward pass of _VDropout

        Parameters
        ----------
        x : torch.Tensor
            Vector channel

        Returns
        -------
        torch.Tensor
            Dropped out vector channel
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


@torch.jit.script_if_tracing
def _tuple_cat(
    input_1: Tuple[torch.Tensor, torch.Tensor],
    input_2: Tuple[torch.Tensor, torch.Tensor],
    input_3: Tuple[torch.Tensor, torch.Tensor],
    dim: int = -1
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Concatenates any number of tuples (s, V) elementwise.

    Parameters
    ----------
    input_1 : Tuple[torch.Tensor, torch.Tensor]
        First input to concatenate
    input_2 : Tuple[torch.Tensor, torch.Tensor]
        Second input to concatenate
    input_3 : Tuple[torch.Tensor, torch.Tensor]
        Third input to concatenate
    dim : int, optional
        dimension along which to concatenate when viewed
        as the `dim` index for the scalar-channel tensors, by default -1.
        This means that `dim=-1` will be applied as
        `dim=-2` for the vector-channel tensors.

    Returns
    -------
    Tuple[torch.Tensor, torch.Tensor]
        Concatenated tuple
    """

    dim = int(dim % len(input_1[0].shape))
    s_args, v_args = list(zip(input_1, input_2, input_3))
    return torch.cat(s_args, dim=dim), torch.cat(v_args, dim=dim)


@torch.jit.script_if_tracing
def _tuple_index(
    x: Tuple[torch.Tensor, torch.Tensor], idx: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Indexes a tuple (s, V) along the first dimension at a given index.

    Parameters
    ----------
    x : Tuple[torch.Tensor, torch.Tensor]
        Tuple to be indexed
    idx : torch.Tensor
        any object which can be used to index a `torch.Tensor`

    Returns
    -------
    Tuple[torch.Tensor, torch.Tensor]
        Tuple with the element at the given index
    """
    return x[0][idx], x[1][idx]


@torch.jit.script_if_tracing
def _norm_no_nan(
    x: torch.Tensor,
    axis: int = -1,
    keepdims: bool = False,
    eps: float = 1e-8,
    sqrt: bool = True
) -> torch.Tensor:
    """L2 norm of tensor clamped above a minimum value `eps`.

    Parameters
    ----------
    x : torch.Tensor
        Input tensor
    axis : int, optional
        Axis along which to compute the norm, by default -1
    keepdims : bool, optional
        Whether to keep the original dimensions, by default False
    eps : float, optional
        Lowest threshold for clamping the norm, by default 1e-8
    sqrt : bool, optional
        Compute the sqaure root in L2 norm, by default True. 
        If `False`, returns the square of the L2 norm

    Returns
    -------
    torch.Tensor
        Normed tensor
    """
    out = torch.clamp(torch.sum(torch.square(x), axis, keepdims), min=eps)
    return torch.sqrt(out) if sqrt else out


@torch.jit.script_if_tracing
def _split(x: torch.Tensor, nv: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """Splits a merged representation of (s, V) back into a tuple.
    Should be used only with `_merge(s, V)` and only if the tuple
    representation cannot be used.


    Parameters
    ----------
    x : torch.Tensor
        the `torch.Tensor` returned from `_merge`
    nv : int
        the number of vector channels in the input to `_merge`

    Returns
    -------
    Tuple[torch.Tensor, torch.Tensor]
        split representation
    """
    s = x[..., :-3 * nv]
    v = x[..., -3 * nv:].contiguous().view(x.shape[0], nv, 3)
    return s, v


@torch.jit.script_if_tracing
def _merge(s: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Merges a tuple (s, V) into a single `torch.Tensor`, where the
    vector channels are flattened and appended to the scalar channels.
    Should be used only if the tuple representation cannot be used.
    Use `_split(x, nv)` to reverse.
    """
    v = v.contiguous().view(v.shape[0], v.shape[1] * 3)
    return torch.cat([s, v], -1)


def test_gvp() -> None:
    from mlcolvar.core.nn.graph.utils import _test_get_data
    from mlcolvar.data.graph.utils import create_graph_tracing_example

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
    ref_out = torch.tensor([[0.6100070244145421, -0.2559670171962067]] * 6)
    assert ( torch.allclose(model(data), ref_out) )
    
    traced_model = torch.jit.trace(model, example_inputs=create_graph_tracing_example(2))
    assert ( torch.allclose(traced_model(data), ref_out) )

    model = GVPModel(
        n_out=2,
        cutoff=0.1,
        atomic_numbers=[1, 8],
        n_bases=6,
        n_polynomials=6,
        n_layers=2,
        n_messages=2,
        n_feedforwards=2,
        n_scalars_node=16,
        n_vectors_node=8,
        n_scalars_edge=16,
        drop_rate=0,
        activation='SiLU',
    )

    data = _test_get_data()
    ref_out = torch.tensor([[-0.3065361946949377, 0.16624918721972567]] * 6)
    assert ( torch.allclose(model(data), ref_out) )
    
    traced_model = torch.jit.trace(model, example_inputs=create_graph_tracing_example(2))
    assert ( torch.allclose(traced_model(data), ref_out) )


    torch.set_default_dtype(torch.float32)

if __name__ == '__main__':
    test_gvp()
