import pytest

from mlcolvar.utils.timelagged import test_create_timelagged_dataset, test_compute_koopman_weights

if __name__ == "__main__":
    test_create_timelagged_dataset()
    test_compute_koopman_weights()