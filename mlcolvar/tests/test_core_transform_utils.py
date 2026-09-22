import lightning
import torch

from mlcolvar.core.transform import Transform
from mlcolvar.core.transform.descriptors import PairwiseDistances
from mlcolvar.core.transform.tools import SwitchingFunctions
from mlcolvar.core.transform.utils import Inverse, SequentialTransform, Statistics
from mlcolvar.cvs.committor import Committor
from mlcolvar.cvs.committor.utils import initialize_committor_masses
from mlcolvar.data import DictDataset, DictLoader, DictModule


def test_sequential_transform():
    # test with sequential PairwiseDistances and a SwitchingFunctions as to compute contacts
    compute_distances = PairwiseDistances(
        n_atoms=4,
        PBC=True,
        cell=[2, 2, 2],
        scaled_coords=True,
    )
    apply_switch = SwitchingFunctions(
        in_features=6,
        name="Rational",
        cutoff=1,
    )

    # mock positions
    pos = torch.rand((2, 4, 3))
    pos.requires_grad = True

    # create sequential transform
    sequential = SequentialTransform(compute_distances, apply_switch)

    # compute reference
    dist = compute_distances(pos)
    cont_ref = apply_switch(dist)

    # compute sequential
    cont_seq = sequential(pos)
    cont_seq.sum().backward()

    assert torch.allclose(cont_ref, cont_seq)
    assert sequential.in_features == compute_distances.in_features
    assert sequential.out_features == apply_switch.out_features

    # check the machinery in training
    masses = initialize_committor_masses(
        atom_types=[0, 0, 0, 0],
        masses=[1.008],
    )

    model = Committor(
        model=[6, 2, 1],
        atomic_masses=masses,
        alpha=1,
    )
    model.preprocessing = sequential

    pos = torch.rand((5, 4, 3))
    labels = torch.zeros(len(pos))
    labels[int(len(pos) / 2):] += 1
    weights = torch.ones(len(pos))

    dataset = DictDataset(
        {
            "data": pos,
            "labels": labels,
            "weights": weights,
        }
    )
    datamodule = DictModule(dataset, lengths=[1])

    trainer = lightning.Trainer(
        max_epochs=5,
        logger=None,
        enable_checkpointing=False,
        limit_val_batches=0,
        num_sanity_val_steps=0,
    )
    trainer.fit(model, datamodule)

    out = model(pos)
    out.sum().backward()


def test_inverse():
    class ForwardModel(Transform):
        def __init__(self, in_features=5, out_features=5):
            super().__init__(in_features=5, out_features=5)
            self.mean = 0

        def update_mean(self, x):
            self.mean = torch.mean(x)

        def forward(self, x):
            return x - self.mean

        def inverse(self, x):
            return x + self.mean

    forward_model = ForwardModel()
    inverse_model = Inverse(forward_model)

    input = torch.rand(5)
    forward_model.update_mean(input)
    out = forward_model(input)

    assert input.mean() == inverse_model(out).mean()


def test_statistics():
    X = torch.arange(0, 100)
    X = torch.stack([X + 0.0, X + 100.0, X - 1000.0], dim=1)
    y = X.square().sum(1)

    print("X", X.shape)
    print("y", y.shape)

    stats = Statistics()
    stats(X)
    print(stats)
    stats.to_dict()

    loader = DictLoader({"data": X, "target": y}, batch_size=20)

    key = "data"
    stats = Statistics()
    for batch in loader:
        stats.update(batch[key])

    print(stats)

    stats = {}
    for batch in loader:
        for key in loader.keys:
            if key not in stats:
                stats[key] = Statistics(batch[key])
            else:
                stats[key].update(batch[key])

    for key in loader.keys:
        print(key, stats[key])