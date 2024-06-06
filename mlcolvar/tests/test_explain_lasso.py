import pytest

from mlcolvar.explain.lasso import test_lasso_classification, test_lasso_regression

if __name__ == "__main__":
    test_lasso_classification()
    test_lasso_regression()
