import pytest

from mlcolvar.core.transform.utils import test_statistics, test_adjacency_matrix, test_applycutoff

if __name__ == "__main__":
    test_statistics()
    test_adjacency_matrix()
    test_applycutoff()