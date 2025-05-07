from mlcolvar.cvs.committor.committor import test_committor
from mlcolvar.cvs.committor.utils import test_compute_committor_weights, test_Kolmogorov_bias
from mlcolvar.core.loss.committor_loss import test_smart_derivatives


if __name__ == "__main__":
    test_committor()
    test_Kolmogorov_bias()
    test_smart_derivatives()
    test_compute_committor_weights()