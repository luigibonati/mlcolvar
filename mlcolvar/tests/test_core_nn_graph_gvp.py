import pytest
import torch

from mlcolvar.core.nn.graph.gvp import GVPModel
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


def _create_test_data_list():
    return create_test_graph_input(
        output_type="batch",
        n_atoms=3,
        n_samples=6,
        n_states=1,
        add_noise=False,
    )["data_list"]


def test_gvp(float64_default):
    torch.manual_seed(0)

    model = GVPModel(
        n_out=2,
        cutoff=0.1,
        atomic_numbers=[1, 8],
        n_bases=6,
        n_polynomials=6,
        n_layers=2,
        n_messages=2,
        n_feedforwards=1,
        n_scalars_node=16,
        n_vectors_node=8,
        n_scalars_edge=16,
        drop_rate=0,
        activation="SiLU",
    )

    data = _create_test_data_list()
    ref_out = torch.tensor(
        [[0.6100070244145421, -0.2559670171962067]] * 5
    )

    assert torch.allclose(model(data), ref_out)

    traced = torch.jit.trace(
        model,
        example_inputs=create_graph_tracing_example(2),
    )
    assert torch.allclose(traced(data), ref_out)

    model = GVPModel(
        n_out=2,
        cutoff=0.1,
        atomic_numbers=[1, 8],
        n_bases=6,
        n_polynomials=6,
        n_layers=2,
        n_messages=2,
        n_feedforwards=2,
        n_scalars_node=16,
        n_vectors_node=8,
        n_scalars_edge=16,
        drop_rate=0,
        activation="SiLU",
    )

    ref_out = torch.tensor(
        [[0.5097288781305398, -0.032077559793064814]] * 5
    )

    assert torch.allclose(model(data), ref_out)

    traced = torch.jit.trace(
        model,
        example_inputs=create_graph_tracing_example(2),
    )
    assert torch.allclose(traced(data), ref_out)


def test_gvp_from_dataset(float64_default):
    torch.manual_seed(0)

    dataset = create_test_graph_input(
        output_type="dataset",
        n_atoms=3,
        n_samples=5,
        n_states=1,
        add_noise=False,
    )

    model = GVPModel(
        n_out=2,
        dataset_for_initialization=dataset,
        n_bases=6,
        n_polynomials=6,
        n_layers=2,
        n_messages=2,
        n_feedforwards=2,
        n_scalars_node=16,
        n_vectors_node=8,
        n_scalars_edge=16,
        drop_rate=0,
        activation="SiLU",
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

    ref_out = torch.tensor(
        [[-0.12551015, -0.5192468]] * 5
    )
    assert torch.allclose(
        model(dataset.get_graph_inputs()),
        ref_out,
    )

    dataset = create_test_graph_input(
        output_type="dataset",
        n_atoms=3,
        n_samples=5,
        n_states=1,
        add_noise=False,
        environment=True,
    )

    model = GVPModel(
        n_out=2,
        dataset_for_initialization=dataset,
        n_bases=6,
        n_polynomials=6,
        n_layers=2,
        n_messages=2,
        n_feedforwards=2,
        n_scalars_node=16,
        n_vectors_node=8,
        n_scalars_edge=16,
        drop_rate=0,
        activation="SiLU",
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

    ref_out = torch.tensor(
        [[0.33821704, 0.05638876]] * 5
    )
    assert torch.allclose(
        model(dataset.get_graph_inputs()),
        ref_out,
    )