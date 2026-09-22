import torch

from mlcolvar.core.transform.tools.normalization import Normalization
from mlcolvar.core.transform.utils import Inverse, Statistics


def test_normalization():
    # create data
    torch.manual_seed(42)
    in_features = 2
    X = torch.randn((100, in_features)) * 10

    # get stats
    stats = Statistics(X).to_dict()
    norm = Normalization(in_features, mean=stats["mean"], range=stats["std"])

    y = norm(X)

    # test inverse
    z = norm.inverse(y)
    assert(torch.allclose(X.mean(0), z.mean(0)))
    assert(torch.allclose(X.std(0), z.std(0)))

    # test inverse class
    inverse = Inverse(norm)
    q = inverse(y)
    assert(torch.allclose(X.mean(0), q.mean(0)))
    assert(torch.allclose(X.std(0), q.std(0)))

    norm = Normalization(
        in_features, mean=stats["mean"], range=stats["std"], mode="min_max"
    )
