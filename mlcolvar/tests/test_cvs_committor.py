from mlcolvar.cvs.committor.committor import test_committor_1, test_committor_2 , test_committor_with_derivatives
from mlcolvar.cvs.committor.utils import test_compute_committor_weights, test_Kolmogorov_bias


if __name__ == "__main__":
    test_committor_1()
    test_committor_2()
    test_committor_with_derivatives()
    test_Kolmogorov_bias()
    test_compute_committor_weights()