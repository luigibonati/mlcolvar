import math
import torch
import torch_scatter # TODO check this is equivalent in torch scatter
from torch import nn
from torch_geometric.nn import MessagePassing

from mlcolvar.core.nn.graph.gnn import BaseGNN

from typing import List, Dict

"""
The SchNet components. This module is taken from the pgy package:
https://github.com/pyg-team/pytorch_geometric/blob/master/torch_geometric/nn/models/schnet.py
"""

__all__ = ["SchNetModel", "InteractionBlock", "ShiftedSoftplus"]

class SchNetModel(BaseGNN):
    """
    The SchNet [1] model. This implementation is taken from torch_geometric:
    https://github.com/pyg-team/pytorch_geometric/blob/master/torch_geometric/nn/models/schnet.py
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
    n_layers: int
        Number of the graph convolution layers.
    n_filters: int
        Number of filters.
    n_hidden_channels: int
        Size of hidden embeddings.
    aggr: str
        Type of the GNN aggr function.
    w_out_after_sum: bool
        If apply the readout MLP layer after the scatter sum.
    References
    ----------
    .. [1] Schütt, Kristof T., et al. "Schnet–a deep learning architecture for
        molecules and materials." The Journal of Chemical Physics 148.24
        (2018).
    """

    def __init__(
        self,
        n_out: int,
        cutoff: float,
        atomic_numbers: List[int],
        n_bases: int = 16,
        n_layers: int = 2,
        n_filters: int = 16,
        n_hidden_channels: int = 16,
        drop_rate: int = 0,
        aggr: str = 'mean',
        w_out_after_sum: bool = False
    ) -> None:

        super().__init__(
            n_out, cutoff, atomic_numbers, n_bases, 0, 'gaussian'
        )

        self.W_v = nn.Linear(
            len(atomic_numbers), n_hidden_channels, bias=False
        )

        self.layers = nn.ModuleList([
            InteractionBlock(
                n_hidden_channels, n_bases, n_filters, cutoff, aggr
            )
            for _ in range(n_layers)
        ])

        self.W_out = nn.ModuleList([
            nn.Linear(n_hidden_channels, n_hidden_channels // 2),
            ShiftedSoftplus(),
            nn.Linear(n_hidden_channels // 2, n_out)
        ])

        self._w_out_after_sum = w_out_after_sum

        self.reset_parameters()

    def reset_parameters(self) -> None:
        """
        Resets all learnable parameters of the module.
        """
        self.W_v.reset_parameters()

        for layer in self.layers:
            layer.reset_parameters()

        nn.init.xavier_uniform_(self.W_out[0].weight)
        self.W_out[0].bias.data.fill_(0)
        nn.init.xavier_uniform_(self.W_out[2].weight)
        self.W_out[2].bias.data.fill_(0)

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

        h_E = self.embed_edge(data)
        h_V = self.W_v(data['node_attrs'])

        batch_id = data['batch']

        for layer in self.layers:
            h_V = h_V + layer(h_V, data['edge_index'], h_E[0], h_E[1])

        if not self._w_out_after_sum:
            for w in self.W_out:
                h_V = w(h_V)
        out = h_V

        if scatter_mean:
            if 'system_masks' not in data.keys():
                # TODO check this is equivalent in torch scatter
                out = torch_scatter.scatter_mean(out, batch_id, dim=0)
            else:
                out = out * data['system_masks']
                # TODO check this is equivalent in torch scatter
                out = torch_scatter.scatter_sum(out, batch_id, dim=0)
                out = out / data['n_system']
        
        if self._w_out_after_sum:
            for w in self.W_out:
                out = w(out)

        return out

class InteractionBlock(nn.Module):
    def __init__(
        self,
        hidden_channels: int,
        num_gaussians: int,
        num_filters: int,
        cutoff: float
    ) -> None:
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(num_gaussians, num_filters),
            ShiftedSoftplus(),
            nn.Linear(num_filters, num_filters),
        )
        self.conv = CFConv(
            hidden_channels, hidden_channels, num_filters, self.mlp, cutoff
        )
        self.act = ShiftedSoftplus()
        self.lin = nn.Linear(hidden_channels, hidden_channels)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.mlp[0].weight)
        self.mlp[0].bias.data.fill_(0)
        nn.init.xavier_uniform_(self.mlp[2].weight)
        self.mlp[2].bias.data.fill_(0)
        self.conv.reset_parameters()
        nn.init.xavier_uniform_(self.lin.weight)
        self.lin.bias.data.fill_(0)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weight: torch.Tensor,
        edge_attr: torch.Tensor,
    ) -> torch.Tensor:
        x = self.conv(x, edge_index, edge_weight, edge_attr)
        x = self.act(x)
        x = self.lin(x)
        return x


class CFConv(MessagePassing):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_filters: int,
        network: nn.Sequential,
        cutoff: float,
    ) -> None:
        super().__init__(aggr='mean')
        self.lin1 = nn.Linear(in_channels, num_filters, bias=False)
        self.lin2 = nn.Linear(num_filters, out_channels)
        self.network = network
        self.cutoff = cutoff

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.lin1.weight)
        nn.init.xavier_uniform_(self.lin2.weight)
        self.lin2.bias.data.fill_(0)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weight: torch.Tensor,
        edge_attr: torch.Tensor,
    ) -> torch.Tensor:
        C = 0.5 * (torch.cos(edge_weight * math.pi / self.cutoff) + 1.0)
        W = self.network(edge_attr) * C.view(-1, 1)

        x = self.lin1(x)
        x = self.propagate(edge_index, x=x, W=W)
        x = self.lin2(x)
        return x

    def message(self, x_j: torch.Tensor, W: torch.Tensor) -> torch.Tensor:
        return x_j * W

# TODO maybe remove and use the common one
class ShiftedSoftplus(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.shift = torch.log(torch.tensor(2.0)).item()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return nn.functional.softplus(x) - self.shift