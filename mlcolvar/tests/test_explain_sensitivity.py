import pytest

from mlcolvar.explain.sensitivity import test_sensitivity_analysis
from mlcolvar.explain.graph_sensitivity import test_graph_sensitivity, test_get_cv_values_graph

if __name__ == "__main__":
    test_sensitivity_analysis()
    test_graph_sensitivity()
    test_get_cv_values_graph()
