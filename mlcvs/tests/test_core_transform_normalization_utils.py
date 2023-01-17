import pytest

from mlcvs.core.transform.normalization.utils import test_running_average, test_running_min_max

if __name__ == "__main__":
    test_running_average()
    test_running_min_max()