import pytest

from mlcolvar.cvs.supervised.deeplda import test_deeplda

if __name__ == "__main__":
    test_deeplda(n_states=2)
    test_deeplda(n_states=3) 