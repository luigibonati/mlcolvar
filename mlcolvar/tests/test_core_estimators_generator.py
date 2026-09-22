import torch

from mlcolvar.core.estimators.generator import Generator
from mlcolvar.data import DictDataset, DictModule


def test_generator():
    in_features = 2
    X = torch.rand(100, in_features) * 100

    w = torch.rand(len(X))

    # Compute generator
    generator = Generator(in_features, out_features=2)

    dataset = DictDataset({"data": X, "weights": w})
    datamodule = DictModule(dataset, lengths=[0.8, 0.2])
    datamodule.setup()

    generator.compute(
        datamodule.train_dataloader(),
        eta=0.1,
        friction=torch.tensor([1.0, 1.0]),
        tikhonov_reg=1e-4,
        n_dim=1,
        softmax_postproc=False,
    )

    s = generator(X)

    print(X.shape, "-->", s.shape)
    print("eigvals", generator.evals)