import torch
from torch import nn
from typing import List, Dict, Tuple

from mlcolvar.core.nn.graph import radial
from mlcolvar.utils import _code

"""
GNN models.
"""

__all__ = ['BaseGNN']


class BaseGNN(nn.Module):
    """
    Base class for Graph Neural Network (GNN) models
    """

    def __init__(
        self,
        n_out: int,
        cutoff: float,
        atomic_numbers: List[int],
        pooling_operation: str,
        n_bases: int = 6,
        n_polynomials: int = 6,
        basis_type: str = 'bessel',
    ) -> None:
        """Initializes the core of a GNN model, taking care of edge embeddings.

        Parameters
        ----------
        n_out : int
            Number of the output scalar node features.
        cutoff : float
            Cutoff radius of the basis functions. Should be the same as the cutoff
            radius used to build the graphs.
        atomic_numbers : List[int]
            The atomic numbers mapping.
        pooling_operation : str
            Type of pooling operation to combine node-level features into graph-level features, either mean or sum
        n_bases : int, optional
            Size of the basis set used for the embedding, by default 6
        n_polynomials : int, optional
            Order of the polynomials in the basis functions, by default 6
        basis_type : str, optional
            Type of the basis function, by default 'bessel'
        """
        super().__init__()

        self._radial_embedding = radial.RadialEmbeddingBlock(cutoff=cutoff, 
                                                             n_bases=n_bases, 
                                                             n_polynomials=n_polynomials, 
                                                             basis_type=basis_type
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
        self.pooling_operation = pooling_operation

    @property
    def out_features(self):
        return self.n_out
    
    @property
    def in_features(self):
        return None

    def embed_edge(
        self, data: Dict[str, torch.Tensor], normalize: bool = True
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Performs the model edge embedding form `torch_geometric.data.Batch` object.

        Parameters
        ----------
        data: Dict[str, torch.Tensor]
            The data dict. Usually from the `to_dict` method of a
            `torch_geometric.data.Batch` object.
        normalize: bool
            If to return the normalized distance vectors, by default True.
        
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
    
    def pooling(self,
                input : torch.Tensor,
                data : Dict[str, torch.Tensor]) -> torch.Tensor:
        """Performs pooling of the node-level outputs to obtain a graph-level output

        Parameters
        ----------
        input : torch.Tensor
            Nodel level features to be pooled
        data : Dict[str, torch.Tensor]
            Data batch containing the graph data informations

        Returns
        -------
        torch.Tensor
            Pooled output
        """
        if self.pooling_operation == 'mean':
            if 'system_masks' not in data.keys():
                out = _code.scatter_mean(input, data['batch'], dim=0)
            else:
                out = input * data['system_masks']
                out = _code.scatter_sum(out, data['batch'], dim=0)
                out = out / data['n_system']
        
        elif self.pooling_operation == 'sum':
            if 'system_masks' in data.keys():
                out = input * data['system_masks']
            else:
                out = _code.scatter_sum(input, data['batch'], dim=0)
        else:
            raise ValueError (f"Invalid pooling operation! Found {self.pooling_operation}")

        return out
    
def get_edge_vectors_and_lengths(
    positions: torch.Tensor,
    edge_index: torch.Tensor,
    shifts: torch.Tensor,
    normalize: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Calculates edge vectors and lengths by indices and shift vectors.

    Parameters
    ----------
    positions: torch.Tensor (shape: [n_atoms, 3])
        The positions tensor.
    edge_index: torch.Tensor (shape: [2, n_edges])
        The edge indices.
    shifts: torch.Tensor (shape: [n_edges, 3])
        The shifts vector.
    normalize: bool
        If to return the normalized distance vectors, by default True.
    
    Returns
    -------
    vectors: torch.Tensor (shape: [n_edges, 3])
        The distances vectors.
    lengths: torch.Tensor (shape: [n_edges, 1])
        The edges lengths.
    """
    sender = edge_index[0]
    receiver = edge_index[1]
    vectors = positions[receiver] - positions[sender] + shifts  # [n_edges, 3]
    lengths = torch.linalg.norm(vectors, dim=-1, keepdim=True)  # [n_edges, 1]

    if normalize:
        vectors = torch.nan_to_num(torch.div(vectors, lengths))

    return vectors, lengths


def test_get_edge_vectors_and_lengths() -> None:
    dtype = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)

    data = dict()
    data['positions'] = torch.tensor(
        [[0.0, 0.0, 0.0], [0.07, 0.07, 0.0], [0.07, -0.07, 0.0]],
        dtype=torch.float64
    )
    data['edge_index'] = torch.tensor(
        [[0, 0, 1, 1, 2, 2], [2, 1, 0, 2, 1, 0]]
    )
    data['shifts'] = torch.tensor([
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0],
        [0.0, 0.2, 0.0],
        [0.0, -0.2, 0.0],
        [0.0, 0.0, 0.0],
    ])

    vectors, distances = get_edge_vectors_and_lengths(**data, normalize=False)
    assert(torch.allclose(vectors, torch.tensor([[0.0700, -0.0700, 0.0000],
                                                 [0.0700,  0.0700, 0.0000],
                                                 [-0.070, -0.0700, 0.0000],
                                                 [0.0000,  0.0600, 0.0000],
                                                 [0.0000, -0.0600, 0.0000],
                                                 [-0.070,  0.0700, 0.0000]])
                        )
            )
    assert(torch.allclose(distances,torch.tensor([[0.09899494936611666],
                                                  [0.09899494936611666],
                                                  [0.09899494936611666],
                                                  [0.06000000000000000],
                                                  [0.06000000000000000],
                                                  [0.09899494936611666]]) 
                        )
            )
    
    vectors, distances = get_edge_vectors_and_lengths(**data, normalize=True)
    assert(torch.allclose(vectors, torch.tensor([[0.70710678118654757, -0.70710678118654757, 0.0],
                                                 [0.70710678118654757,  0.70710678118654757, 0.0],
                                                 [-0.7071067811865476, -0.70710678118654757, 0.0],
                                                 [0.00000000000000000,  1.00000000000000000, 0.0],
                                                 [0.00000000000000000, -1.00000000000000000, 0.0],
                                                 [-0.7071067811865476,  0.70710678118654757, 0.0]])
                        )  
        )

    assert(torch.allclose(distances, torch.tensor([[0.09899494936611666],
                                                   [0.09899494936611666],
                                                   [0.09899494936611666],
                                                   [0.06000000000000000],
                                                   [0.06000000000000000],
                                                   [0.09899494936611666]])
                        )
        )

    torch.set_default_dtype(dtype)

if __name__ == "__main__":
    test_get_edge_vectors_and_lengths()