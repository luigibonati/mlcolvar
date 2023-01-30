import pytest

from mlcvs.cvs.supervised.deeptda_cv import test_deeptda_cv

if __name__ == "__main__":
    test_deeptda_cv(2,1)
    test_deeptda_cv(3,1)
    test_deeptda_cv(3,2) 