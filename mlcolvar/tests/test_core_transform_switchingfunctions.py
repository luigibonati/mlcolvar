import torch

from mlcolvar.core.transform.tools.switching_functions import SwitchingFunctions


def test_switchingfunctions():
    x = torch.Tensor([1., 2., 3.])
    cutoff = 2
    switch = SwitchingFunctions(in_features=len(x), name='Fermi', cutoff=cutoff)
    switch(x)

    switch = SwitchingFunctions(in_features=len(x), name='Fermi', cutoff=cutoff, options = {'q' : 0.5})
    switch(x)

    switch = SwitchingFunctions(in_features=len(x), name='Rational', cutoff=cutoff, options = {'n' : 6, 'm' : 12})
    switch(x)
