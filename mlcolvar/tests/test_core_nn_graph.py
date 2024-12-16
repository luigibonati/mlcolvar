from mlcolvar.core.nn.graph.gnn import test_get_edge_vectors_and_lengths
from mlcolvar.core.nn.graph.radial import test_bessel_basis, test_gaussian_basis, test_polynomial_cutoff, test_radial_embedding_block

if __name__ == "__main__":
    test_get_edge_vectors_and_lengths()
    test_bessel_basis()
    test_gaussian_basis()
    test_polynomial_cutoff()
    test_radial_embedding_block()