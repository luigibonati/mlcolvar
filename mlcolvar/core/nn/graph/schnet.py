import math
import torch
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
        Type of aggregation function for the GNN message passing.
    w_out_after_pool: bool
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
        pooling_operation : str = 'mean',
        n_bases: int = 16,
        n_layers: int = 2,
        n_filters: int = 16,
        n_hidden_channels: int = 16,
        aggr: str = 'mean',
        w_out_after_pool: bool = False,
    ) -> None:
        """The SchNet model. This implementation is taken from torch_geometric:
        https://github.com/pyg-team/pytorch_geometric/blob/master/torch_geometric/nn/models/schnet.py

        Parameters
        ----------
        n_out : int
            Size of the output node features.
        cutoff : float
            Cutoff radius of the basis functions. Should be the same as the cutoff
            radius used to build the graphs.
        atomic_numbers : List[int]
            The atomic numbers mapping.
        pooling_operation : str
            Type of pooling operation to combine node-level features into graph-level features, either mean or sum, by default 'mean'
        n_bases : int, optional
            Size of the basis set used for the embedding, by default 16
        n_layers : int, optional
            Number of the graph convolution layers, by default 2
        n_filters : int, optional
            Number of filters, by default 16
        n_hidden_channels : int, optional
            Size of hidden embeddings, by default 16
        aggr : str, optional
            Type of the GNN aggregation function, by default 'mean'
        w_out_after_pool : bool, optional
            Whether to apply the last linear transformation form hidden to output channels after the pooling sum, by default False
        """

        super().__init__(
            n_out=n_out, 
            cutoff=cutoff, 
            atomic_numbers=atomic_numbers, 
            pooling_operation=pooling_operation, 
            n_bases=n_bases, 
            n_polynomials=0, 
            basis_type='gaussian'
        )

        # transforms embedding into hidden channels
        self.W_v = nn.Linear(
            in_features=len(atomic_numbers), 
            out_features=n_hidden_channels, 
            bias=False
        )

        # initialize layers with interaction blocks
        self.layers = nn.ModuleList([
            InteractionBlock(
                n_hidden_channels, n_bases, n_filters, cutoff, aggr
            )
            for _ in range(n_layers)
        ])

        # transforms hidden channels into output channels
        self.W_out = nn.ModuleList([
            nn.Linear(n_hidden_channels, n_hidden_channels // 2),
            ShiftedSoftplus(),
            nn.Linear(n_hidden_channels // 2, n_out)
        ])

        self._w_out_after_pool = w_out_after_pool

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
        self, data: Dict[str, torch.Tensor], pool: bool = True
    ) -> torch.Tensor:
        """
        The forward pass.
        Parameters
        ----------
        data: Dict[str, torch.Tensor]
            The data dict. Usually came from the `to_dict` method of a
            `torch_geometric.data.Batch` object.
        pool: bool
            If to perform the pooling to the model output.
        """

        # embed edges and node attrs
        h_E = self.embed_edge(data)
        h_V = self.W_v(data['node_attrs'])

        # update through layers
        for layer in self.layers:
            h_V = h_V + layer(h_V, data['edge_index'], h_E[0], h_E[1])

        # in case the last linear transformation is performed BEFORE pooling
        if not self._w_out_after_pool:
            for w in self.W_out:
                h_V = w(h_V)
        out = h_V

        # perform pooling of the node-level ouptuts
        if pool:
            out = self.pooling(input=out, data=data)
        
        # in case the last linear transformation is performed AFTER pooling
        if self._w_out_after_pool:
            for w in self.W_out:
                out = w(out)

        return out

class InteractionBlock(nn.Module):
    def __init__(
        self,
        hidden_channels: int,
        num_gaussians: int,
        num_filters: int,
        cutoff: float,
        aggr: str = 'mean'
    ) -> None:
        """SchNet interaction block

        Parameters
        ----------
        hidden_channels : int
            Size of hidden embeddings
        num_gaussians : int
            Number of Gaussians for the embedding
        num_filters : int
            Number of filters
        cutoff : float
            Radial cutoff
        aggr : str, optional
            Aggregation function, by default 'mean'
        """
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(num_gaussians, num_filters),
            ShiftedSoftplus(),
            nn.Linear(num_filters, num_filters),
        )
        self.conv = CFConv(
            hidden_channels,
            hidden_channels,
            num_filters,
            self.mlp,
            cutoff,
            aggr
        )
        self.act = ShiftedSoftplus()
        self.lin = nn.Linear(hidden_channels, hidden_channels)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        """
        Resets all learnable parameters of the module.
        """
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
    """Continuos-filter convolution from SchNet"""
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_filters: int,
        network: nn.Sequential,
        cutoff: float,
        aggr: str = 'mean'
    ) -> None:
        """Applies a continuous-filter convolution

        Parameters
        ----------
        in_channels : int
            Number of input channels
        out_channels : int
            Number of output channels
        num_filters : int
            Number of filters
        network : nn.Sequential
            Neural network
        cutoff : float
            Radial cutoff
        aggr : str, optional
            Aggregation function, by default 'mean'
        """
        super().__init__(aggr=aggr)
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

def test_schnet_1() -> None:
    torch.manual_seed(0)
    torch.set_default_dtype(torch.float64)

    model = SchNetModel(
        n_out=2,
        cutoff=0.1,
        atomic_numbers=[1, 8],
        n_bases=6,
        n_layers=2,
        n_filters=16,
        n_hidden_channels=16
    )

    data = _create_test_data_list()
    ref_out = torch.tensor([[0.40384621527953063, -0.12575133651389694]] * 5)
    assert ( torch.allclose(model(data), ref_out) )

    model = SchNetModel(
        n_out=2,
        cutoff=0.1,
        atomic_numbers=[1, 8],
        n_bases=6,
        n_layers=2,
        n_filters=16,
        n_hidden_channels=16,
        pooling_operation='sum',
    )

    data = _create_test_data_list()
    ref_out = torch.tensor([[0.15911003978422333, 0.45333821159230125]] * 5)
    assert ( torch.allclose(model(data), ref_out) )

    traced_model = torch.jit.trace(model, example_inputs=create_graph_tracing_example(2))
    assert ( torch.allclose(traced_model(data), ref_out) )


def test_schnet_2() -> None:
    torch.manual_seed(0)
    torch.set_default_dtype(torch.float64)

    model = SchNetModel(
        n_out=2,
        cutoff=0.1,
        atomic_numbers=[1, 8],
        n_bases=6,
        n_layers=2,
        n_filters=16,
        n_hidden_channels=16,
        aggr='min',
        w_out_after_pool=True
    )

    data = _create_test_data_list()
    ref_out = torch.tensor([[0.3654537816221449, -0.0748265132499575]] * 5)
    assert ( torch.allclose(model(data), ref_out) )
    
    torch.set_default_dtype(torch.float32)

if __name__ == "__main__":
    test_schnet_1()
