import math
import torch
from torch import nn
from torch_geometric.nn import MessagePassing

from mlcolvar.data import DictDataset
from mlcolvar.core.nn.graph.gnn import BaseGNN

from typing import List, Dict, Optional, Tuple

"""
The PaiNN components. This module is directly taken from repo:
https://github.com/MaxH1996/PaiNN-in-PyG
"""

__all__ = ['PaiNNModel', 'MessagePassingPaiNN', 'UpdatePaiNN', 'AttentionGatePaiNN']

class PaiNNModel(BaseGNN):
    """
    The PaiNN model [1]. This implementation is taken from:
    https://github.com/MaxH1996/PaiNN-in-PyG/blob/main/PaiNN.py


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
    long_range_cutoff : float
        Cutoff radius for the long-range edges defined on subsystem atoms. 
        If negative, no long-range interactions are considered, by default -1.0
    pooling_operation : str
        Type of pooling operation to combine node-level features into graph-level features, either mean or sum, by default 'mean'
    n_bases : int, optional
        Size of the basis set used for the embedding, by default 6
    n_layers : int, optional
        Number of the graph convolution layers, by default 2
    n_hidden_channels : int, optional
        Size of hidden embeddings, by default 16
    aggr: str
        Type of the GNN aggr function.
    w_out_after_pool : bool, optional
        Whether to apply the last linear transformation form hidden to output channels after the pooling sum, by default True

    References
    ----------
    .. [1] Schütt, Kristof, Oliver Unke, and Michael Gastegger.
        "Equivariant message passing for the prediction of tensorial properties
        and molecular spectra." International conference on machine learning.
        PMLR, 2021.
    """

    def __init__(
        self,
        n_out: int,
        dataset_for_initialization: DictDataset = None,
        pooling_operation : str = 'mean',
        n_bases: int = 6,
        n_layers: int = 2,
        n_hidden_channels: int = 16,
        aggr: str = 'add',
        w_out_after_pool: bool = True,
        **kwargs
    ) -> None:

        super().__init__(
            n_out=n_out, 
            dataset_for_initialization=dataset_for_initialization,
            pooling_operation=pooling_operation, 
            n_bases=n_bases, 
            n_polynomials=0, 
            basis_type='gaussian',
            **kwargs
        )

        self.W_v = nn.Linear(
            len(self.atomic_numbers), n_hidden_channels, bias=False
        )

        # TODO: find out how to do attentional aggr properly.
        if aggr in ['attention', 'attentional']:
            raise NotImplementedError(
                'Attentional aggregation is not implementated for PaiNN.'
            )
        elif aggr in ['attention_separate', 'attentional_separate']:
            raise NotImplementedError(
                'Attentional aggregation is not implementated for PaiNN.'
            )
        else:
            aggr = [aggr] * n_layers

        self.layers_message = nn.ModuleList([
            MessagePassingPaiNN(
                n_hidden_channels, n_bases, self.cutoff, self.long_range_cutoff, aggr[i],
            ) for i in range(n_layers)
        ])
        self.layers_update = nn.ModuleList([
            UpdatePaiNN(n_hidden_channels)
            for i in range(n_layers)
        ])

        self.W_out = nn.ModuleList([
            nn.Linear(n_hidden_channels, n_hidden_channels // 2),
            nn.SiLU(),
            nn.Linear(n_hidden_channels // 2, n_out)
        ])

        self._w_out_after_pool = w_out_after_pool

        self.reset_parameters()

    def reset_parameters(self) -> None:
        """
        Resets all learnable parameters of the module.
        """
        self.W_v.reset_parameters()

        for layer in self.layers_message:
            layer.reset_parameters()
        for layer in self.layers_update:
            layer.reset_parameters()

        nn.init.xavier_uniform_(self.W_out[0].weight)
        self.W_out[0].bias.data.fill_(0)
        nn.init.xavier_uniform_(self.W_out[2].weight)
        self.W_out[2].bias.data.fill_(0)

    def forward(
        self, data: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """
        The forward pass.

        Parameters
        ----------
        data: Dict[str, torch.Tensor]
            The data dict. Usually came from the `to_dict` method of a
            `torch_geometric.data.Batch` object.
        """

        h_E = self.embed_edge(data)
        h_V_s = self.W_v(data['node_attrs'])
        h_V_v = torch.zeros_like(h_V_s).unsqueeze(-1).expand(-1, -1, 3)

        batch_id = data['batch']

        for message, update in zip(self.layers_message, self.layers_update):
            s_temp, v_temp = message(
                h_V_s,
                h_V_v,
                data['edge_index'],
                h_E[0],
                h_E[2],
                h_E[1],
                data.get('edge_masks_lr'),
            )
            h_V_s, h_V_v = s_temp + h_V_s, v_temp + h_V_v
            h_V_s, h_V_v = update(h_V_s, h_V_v)
            h_V_s, h_V_v = s_temp + h_V_s, v_temp + h_V_v

        if not self._w_out_after_pool:
            for w in self.W_out:
                h_V_s = w(h_V_s)
        out = h_V_s

        # pooling is controlled by `self.pooling_operation` (mean/sum/None)
        out = self.pooling(input=out, data=data)
        
        # in case the last linear transformation is performed AFTER pooling
        if self._w_out_after_pool:
            for w in self.W_out:
                out = w(out)

        return out

    def forward_node_feature(
        self, data: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """
        The forward pass without the readout function.

        Parameters
        ----------
        data: Dict[str, torch.Tensor]
            The data dict. Usually came from the `to_dict` method of a
            `torch_geometric.data.Batch` object.
        """

        h_E = self.embed_edge(data)
        h_V_s = self.W_v(data['node_attrs'])
        h_V_v = torch.zeros_like(h_V_s).unsqueeze(-1).expand(-1, -1, 3)

        for message, update in zip(self.layers_message, self.layers_update):
            s_temp, v_temp = message(
                h_V_s,
                h_V_v,
                data['edge_index'],
                h_E[0],
                h_E[2],
                h_E[1],
                data.get('edge_masks_lr'),
            )
            h_V_s, h_V_v = s_temp + h_V_s, v_temp + h_V_v
            h_V_s, h_V_v = update(h_V_s, h_V_v)
            h_V_s, h_V_v = s_temp + h_V_s, v_temp + h_V_v

        return h_V_s
    
class MessagePassingPaiNN(MessagePassing):

    propagate_type = {
        'x': torch.Tensor,
        'W': torch.Tensor,
        'C': torch.Tensor,
        'edge_lengths': torch.Tensor,
        'edge_vectors': torch.Tensor,
        'flat_shape_s': int,
        'flat_shape_v': int,
    }

    def __init__(
        self,
        n_hidden_channels: int,
        n_gaussians: int,
        cutoff: float,
        long_range_cutoff: float = -1.0,
        aggr: str = 'mean',
    ) -> None:
        super(MessagePassingPaiNN, self).__init__(aggr=aggr)

        self.cutoff = cutoff
        self.long_range_cutoff = long_range_cutoff
        self.n_hidden_channels = n_hidden_channels

        self.lin1 = nn.Linear(n_hidden_channels, n_hidden_channels)
        self.lin2 = nn.Linear(n_hidden_channels, 3 * n_hidden_channels)
        self.silu = nn.SiLU()

        self.lin_rbf = nn.Linear(n_gaussians, 3 * n_hidden_channels)
        if long_range_cutoff > 0:
            self.lin_rbf_l = nn.Linear(n_gaussians, 3 * n_hidden_channels)
        else:
            self.lin_rbf_l = None

    def reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.lin1.weight)
        self.lin1.bias.data.fill_(0)
        nn.init.xavier_uniform_(self.lin2.weight)
        self.lin2.bias.data.fill_(0)
        nn.init.xavier_uniform_(self.lin_rbf.weight)
        self.lin_rbf.bias.data.fill_(0)
        nn.init.xavier_uniform_(self.lin_rbf.weight)
        self.lin_rbf.bias.data.fill_(0)
        if self.lin_rbf_l is not None:
            nn.init.xavier_uniform_(self.lin_rbf_l.weight)
            self.lin_rbf_l.bias.data.fill_(0)
            nn.init.xavier_uniform_(self.lin_rbf_l.weight)
            self.lin_rbf_l.bias.data.fill_(0)

    def forward(
        self,
        s: torch.Tensor,
        v: torch.Tensor,
        edge_index: torch.Tensor,
        edge_lengths: torch.Tensor,
        edge_vectors: torch.Tensor,
        edge_attr: torch.Tensor,
        edge_masks_lr: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        s = s.flatten(-1)
        v = v.flatten(-2)

        flat_shape_v = v.shape[-1]
        flat_shape_s = s.shape[-1]

        W = self.lin_rbf(edge_attr)
        C = 0.5 * torch.cos(edge_lengths * math.pi / self.cutoff) + 0.5

        if edge_masks_lr is not None and self.lin_rbf_l is not None:
            assert self.long_range_cutoff > self.cutoff

            indices_l = edge_masks_lr.nonzero()[:, 0]
            lengths_l = edge_lengths[indices_l]
            edge_attr_l = edge_attr[indices_l]

            W_l = self.lin_rbf_l(edge_attr_l)
            W = W.index_copy_(0, indices_l, W_l)

            C_l = 0.5 * torch.cos(lengths_l * math.pi / self.long_range_cutoff) + 0.5
            C_l_1 = 0.5 - 0.5 * torch.cos(lengths_l * math.pi / self.cutoff)
            C_l = C_l * (
                C_l_1 * (lengths_l < self.cutoff)
                + 1.0 * (lengths_l > self.cutoff)
            )
            C = C.index_copy_(0, indices_l, C_l)

        x = torch.cat([s, v], dim=-1)

        x = self.propagate(
            edge_index,
            x=x,
            W=W,
            C=C,
            edge_lengths=edge_lengths,
            edge_vectors=edge_vectors,
            flat_shape_s=flat_shape_s,
            flat_shape_v=flat_shape_v,
        )

        return x

    def message(
        self,
        x_j: torch.Tensor,
        W: torch.Tensor,
        C: torch.Tensor,
        edge_lengths: torch.Tensor,
        edge_vectors: torch.Tensor,
        flat_shape_s: int,
        flat_shape_v: int,
    ) -> torch.Tensor:

        # Split Input into s_j and v_j
        s_j, v_j = torch.split(x_j, [flat_shape_s, flat_shape_v], dim=-1)

        # s_j channel
        phi = self.lin1(s_j)
        phi = self.silu(phi)
        phi = self.lin2(phi)

        # Split

        left, dsm, right = torch.split(
            phi * W * C.view(-1, 1), self.n_hidden_channels, dim=-1
        )

        # v_j channel
        v_j = v_j.reshape(-1, flat_shape_v // 3, 3)
        hadamard_right = torch.einsum('ij,ik->ijk', right, edge_vectors)
        hadamard_left = torch.einsum('ijk,ij->ijk', v_j, left)
        dvm = (hadamard_left + hadamard_right).flatten(-2)

        # Prepare vector for update
        x_j = torch.cat((dsm, dvm), dim=-1)

        return x_j

    def update(
        self,
        out_aggr: torch.Tensor,
        flat_shape_s: int,
        flat_shape_v: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        s_j, v_j = torch.split(out_aggr, [flat_shape_s, flat_shape_v], dim=-1)

        return s_j, v_j.reshape(-1, flat_shape_v // 3, 3)


class UpdatePaiNN(torch.nn.Module):

    def __init__(self, n_hidden_channels: int) -> None:
        super(UpdatePaiNN, self).__init__()

        self.n_hidden_channels = n_hidden_channels
        self.lin1 = nn.Linear(2 * n_hidden_channels, n_hidden_channels)
        self.lin2 = nn.Linear(n_hidden_channels, 3 * n_hidden_channels)
        self.linu = nn.Linear(
            n_hidden_channels, n_hidden_channels, bias=False
        )
        self.linv = nn.Linear(
            n_hidden_channels, n_hidden_channels, bias=False
        )
        self.silu = nn.SiLU()

    def reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.lin1.weight)
        self.lin1.bias.data.fill_(0)
        nn.init.xavier_uniform_(self.lin2.weight)
        self.lin2.bias.data.fill_(0)
        nn.init.xavier_uniform_(self.linv.weight)
        nn.init.xavier_uniform_(self.linu.weight)

    def forward(
        self, s: torch.Tensor, v: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        s = s.flatten(-1)
        v = v.flatten(-2)

        flat_shape_v = v.shape[-1]

        v_u = v.reshape(-1, flat_shape_v // 3, 3)
        v_ut = torch.transpose(
            v_u, 1, 2
        )  # need transpose to get lin.comb a long feature dimension
        U = torch.transpose(self.linu(v_ut), 1, 2)
        V = torch.transpose(self.linv(v_ut), 1, 2)

        # form the dot product
        UV = torch.einsum('ijk,ijk->ij', U, V)

        # s_j channel
        nV = V.pow(2).sum(dim=-1).sqrt()

        s_u = torch.cat([s, nV], dim=-1)
        s_u = self.lin1(s_u)
        s_u = self.silu(s_u)
        s_u = self.lin2(s_u)

        # final split
        top, middle, bottom = torch.split(s_u, self.n_hidden_channels, dim=-1)

        # outputs
        dvu = torch.einsum('ijk,ij->ijk', v_u, top)
        dsu = middle * UV + bottom

        return dsu, dvu.reshape(-1, flat_shape_v // 3, 3)


class AttentionGatePaiNN(nn.Module):

    def __init__(self, n_hidden_channels: int) -> None:
        super(AttentionGatePaiNN, self).__init__()

        self.n_hidden_channels = n_hidden_channels
        self.gate = nn.Sequential(
            nn.Linear(n_hidden_channels * 2, n_hidden_channels),
            nn.SiLU(),
            nn.Linear(n_hidden_channels, 1)
        )

    def reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.gate[0].weight)
        self.gate[0].bias.data.fill_(0)
        nn.init.xavier_uniform_(self.gate[2].weight)
        self.gate[2].bias.data.fill_(0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        s, v = torch.split(
            x, [self.n_hidden_channels, self.n_hidden_channels * 3], dim=-1
        )
        sv = torch.cat(
            [s, torch.norm(v.reshape(-1, self.n_hidden_channels, 3), dim=-1)],
            dim=1
        )
        return self.gate(sv)
    

from mlcolvar.data.graph.utils import create_graph_tracing_example, create_test_graph_input

def _create_test_data_list():
    batch = create_test_graph_input(
        output_type='batch',
        n_atoms=3,
        n_samples=6,
        n_states=1,
        add_noise=False,
    )
    return batch['data_list']

def test_painn() -> None:
    torch.manual_seed(0)
    torch.set_default_dtype(torch.float64)

    model = PaiNNModel(
        n_out=2,
        cutoff=0.1,
        atomic_numbers=[1, 8],
        n_bases=6,
        n_layers=2,
        n_hidden_channels=12,
        w_out_after_pool=True,
    )

    data = _create_test_data_list()
    ref_out = torch.tensor([[0.012601337298479546, -0.0032668391572678087]] * 5)
    assert ( torch.allclose(model(data), ref_out) )


    result = model.forward_node_feature(data)[:3, :].mean(dim=0, keepdim=True)
    for w in model.W_out:
        result = w(result)
    ref_out = torch.tensor([[0.012601337298479546, -0.0032668391572678087]])
    assert ( torch.allclose(model(data), ref_out) )

    traced_model = torch.jit.trace(model, example_inputs=create_graph_tracing_example(2))
    assert ( torch.allclose(traced_model(data), ref_out) )

def test_painn_2() -> None:
    torch.manual_seed(0)
    torch.set_default_dtype(torch.float64)

    model = PaiNNModel(
        n_out=2,
        cutoff=0.1,
        long_range_cutoff=0.2,
        atomic_numbers=[1, 8],
        n_bases=6,
        n_layers=2,
        n_hidden_channels=12,
        w_out_after_pool=True,
    )

    data = _create_test_data_list()
    data['edge_masks_le'] = torch.zeros(
        ((data['edge_index'].shape[1]), 1), dtype=bool
    )
    data['edge_masks_le'][:-6] = True
    torch.set_printoptions(precision=16)
    ref_out = torch.tensor([[0.0720856700379041, -0.0420276151917215]] * 5)
    assert ( torch.allclose(model(data), ref_out) )