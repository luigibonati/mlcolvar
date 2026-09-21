import pytest
import torch

from mlcolvar.core.nn.graph.schnet import SchNetModel
from mlcolvar.data.graph.utils import (
    create_graph_tracing_example,
    create_test_graph_input,
)


@pytest.fixture
def float64_default():
    old_dtype = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)
    yield
    torch.set_default_dtype(old_dtype)


def _data():
    return create_test_graph_input(
        output_type="batch",
        n_atoms=3,
        n_samples=6,
        n_states=1,
        add_noise=False,
    )["data_list"]


def _model(**kwargs):
    return SchNetModel(
        n_out=2,
        cutoff=0.1,
        atomic_numbers=[1, 8],
        n_bases=6,
        n_layers=2,
        n_filters=16,
        n_hidden_channels=16,
        **kwargs,
    )


def test_schnet_1(float64_default):
    torch.manual_seed(0)

    model = _model()
    data = _data()
    ref = torch.tensor(
        [[0.40384621527953063, -0.12575133651389694]] * 5
    )
    assert torch.allclose(model(data), ref)

    model = _model(pooling_operation="sum")
    ref = torch.tensor(
        [[0.15911003978422333, 0.45333821159230125]] * 5
    )
    assert torch.allclose(model(data), ref)

    traced = torch.jit.trace(
        model,
        example_inputs=create_graph_tracing_example(2),
    )
    assert torch.allclose(traced(data), ref)


def test_schnet_2(float64_default):
    torch.manual_seed(0)

    model = _model(
        aggr="min",
        w_out_after_pool=True,
    )

    ref = torch.tensor(
        [[0.3654537816221449, -0.0748265132499575]] * 5
    )
    assert torch.allclose(model(_data()), ref)


def test_schnet_from_dataset(float64_default):
    torch.manual_seed(0)

    for environment, expected in [
        (False, [0.36632594, -0.08193991]),
        (True, [0.13408032, -0.22392153]),
    ]:
        dataset = create_test_graph_input(
            output_type="dataset",
            n_atoms=3,
            n_samples=5,
            n_states=1,
            add_noise=False,
            environment=environment,
        )

        model = SchNetModel(
            n_out=2,
            dataset_for_initialization=dataset,
            n_bases=6,
            n_layers=2,
            n_filters=16,
            n_hidden_channels=16,
            aggr="max",
            w_out_after_pool=True,
        )

        assert model.cutoff == dataset.metadata["cutoff"]
        assert torch.allclose(
            model.atomic_numbers,
            torch.as_tensor(dataset.metadata["atomic_numbers"]),
        )
        assert torch.allclose(
            model.buffer,
            torch.as_tensor(dataset.metadata["buffer"]),
        )

        ref = torch.tensor([expected] * 5)
        assert torch.allclose(
            model(dataset.get_graph_inputs()),
            ref,
        )


def test_schnet_3(float64_default):
    torch.manual_seed(0)

    model = _model(aggr="attention")
    ref = torch.tensor(
        [[-0.3191231788534454, -0.0436194218681725]] * 5
    )
    assert torch.allclose(model(_data()), ref)

    model = _model(aggr="attention_separate")
    ref = torch.tensor(
        [[-0.1364561627454978, -0.1203537910489112]] * 5
    )
    assert torch.allclose(model(_data()), ref)


def test_schnet_4(float64_default):
    torch.manual_seed(0)

    model = _model(
        long_range_cutoff=0.2,
        aggr="min",
    )

    data = _data()
    data["edge_masks_lr"] = torch.zeros(
        (data["edge_index"].shape[1], 1),
        dtype=bool,
    )
    data["edge_masks_lr"][:-2] = True

    ref = torch.tensor(
        [
            [-0.1873424391457965, -0.0150953093265520],
            [-0.1873424391457965, -0.0150953093265520],
            [-0.1873424391457965, -0.0150953093265520],
            [-0.1873424391457965, -0.0150953093265520],
            [-0.1846414580701179, -0.0121660647548140],
        ]
    )

    assert torch.allclose(model(data), ref)