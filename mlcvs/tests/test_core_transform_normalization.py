import pytest

from mlcvs.core.transform.normalization import test_normalization,test_running_average,test_running_min_max

if __name__ == "__main__":
    test_running_average()
    test_running_min_max()
    test_normalization()