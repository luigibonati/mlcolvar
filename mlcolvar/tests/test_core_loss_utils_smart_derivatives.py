from mlcolvar.core.loss.utils.smart_derivatives import test_smart_derivatives, test_batched_smart_derivatives, test_compute_descriptors_and_derivatives, test_train_with_smart_derivatives

if __name__ == "__main__":
    test_compute_descriptors_and_derivatives()
    test_smart_derivatives()
    test_batched_smart_derivatives()
    test_train_with_smart_derivatives()
