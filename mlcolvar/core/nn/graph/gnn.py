import torch
from torch import nn
from typing import List, Dict, Tuple

from mlcolvar.core.nn.graph import radial

"""
GNN models.
"""

__all__ = ['BaseGNN']


class BaseGNN(nn.Module):
    """
    The commen GNN interface for mlcolvar.
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
    basis_type: str
        Type of the basis function.
    """

    def __init__(
        self,
        n_out: int,
        cutoff: float,
        atomic_numbers: List[int],
        n_bases: int = 6,
        n_polynomials: int = 6,
        basis_type: str = 'bessel'
    ) -> None:
        super().__init__()
        self._model_type = 'gnn'

        self._n_out = n_out
        self._radial_embedding = radial.RadialEmbeddingBlock(
            cutoff, n_bases, n_polynomials, basis_type
        )
        self.register_buffer(
            'n_out', torch.tensor(n_out, dtype=torch.int64)
        )
        self.register_buffer(
            'cutoff', torch.tensor(cutoff, dtype=torch.get_default_dtype())
        )
        self.register_buffer(
            'atomic_numbers', torch.tensor(atomic_numbers, dtype=torch.int64)
        )

    @property
    def out_features(self):
        return self._n_out
    
    def embed_edge(
        self, data: Dict[str, torch.Tensor], normalize: bool = True
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Perform the edge embedding.
        Parameters
        ----------
        data: Dict[str, torch.Tensor]
            The data dict. Usually came from the `to_dict` method of a
            `torch_geometric.data.Batch` object.
        normalize: bool
            If return the normalized distance vectors.
        Returns
        -------
        edge_lengths: torch.Tensor (shape: [n_edges, 1])
            The edge lengths.
        edge_length_embeddings: torch.Tensor (shape: [n_edges, n_bases])
            The edge length embeddings.
        edge_unit_vectors: torch.Tensor (shape: [n_edges, 3])
            The normalized edge vectors.
        """
        vectors, lengths = get_edge_vectors_and_lengths(
            positions=data['positions'],
            edge_index=data['edge_index'],
            shifts=data['shifts'],
            normalize=normalize,
        )
        return lengths, self._radial_embedding(lengths), vectors
    
def get_edge_vectors_and_lengths(
    positions: torch.Tensor,
    edge_index: torch.Tensor,
    shifts: torch.Tensor,
    normalize: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Calculate edge vectors and lengths by indices and shift vectors.
    Parameters
    ----------
    position: torch.Tensor (shape: [n_atoms, 3])
        The position vector.
    edge_index: torch.Tensor (shape: [2, n_edges])
        The edge indices.
    shifts: torch.Tensor (shape: [n_edges, 3])
        The shift vector.
    normalize: bool
        If return the normalized distance vectors.
    Returns
    -------
    vectors: torch.Tensor (shape: [n_edges, 3])
        The distance vectors.
    lengths: torch.Tensor (shape: [n_edges, 1])
        The edge lengths.
    """
    sender = edge_index[0]
    receiver = edge_index[1]
    vectors = positions[receiver] - positions[sender] + shifts  # [n_edges, 3]
    lengths = torch.linalg.norm(vectors, dim=-1, keepdim=True)  # [n_edges, 1]

    if normalize:
        vectors = torch.nan_to_num(torch.div(vectors, lengths))

    return vectors, lengths


# def test_get_data() -> tg.data.Batch:
#     # TODO: This is not a real test, but a helper function for other tests.
#     # Maybe should change its name.
#     torch.manual_seed(0)
#     torch_tools.set_default_dtype('float64')

#     numbers = [8, 1, 1]
#     positions = np.array(
#         [
#             [[0.0, 0.0, 0.0], [0.07, 0.07, 0.0], [0.07, -0.07, 0.0]],
#             [[0.0, 0.0, 0.0], [-0.07, 0.07, 0.0], [0.07, 0.07, 0.0]],
#             [[0.0, 0.0, 0.0], [0.07, -0.07, 0.0], [0.07, 0.07, 0.0]],
#             [[0.0, 0.0, 0.0], [0.0, -0.07, 0.07], [0.0, 0.07, 0.07]],
#             [[0.0, 0.0, 0.0], [0.07, 0.0, 0.07], [-0.07, 0.0, 0.07]],
#             [[0.1, 0.0, 1.1], [0.17, 0.07, 1.1], [0.17, -0.07, 1.1]],
#         ],
#         dtype=np.float64
#     )
#     cell = np.identity(3, dtype=float) * 0.2
#     graph_labels = np.array([[1]])
#     node_labels = np.array([[0], [1], [1]])
#     z_table = gdata.atomic.AtomicNumberTable.from_zs(numbers)

#     config = [
#         gdata.atomic.Configuration(
#             atomic_numbers=numbers,
#             positions=p,
#             cell=cell,
#             pbc=[True] * 3,
#             node_labels=node_labels,
#             graph_labels=graph_labels,
#         ) for p in positions
#     ]
#     dataset = gdata.create_dataset_from_configurations(
#         config, z_table, 0.1, show_progress=False
#     )

#     loader = gdata.GraphDataModule(
#         dataset,
#         lengths=(1.0,),
#         batch_size=10,
#         shuffle=False,
#     )
#     loader.setup()

#     return next(iter(loader.train_dataloader()))

