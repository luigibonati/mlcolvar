import pytest
import torch

from mlcolvar.core.nn.graph.painn import PaiNNModel
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


def test_painn(float64_default):
    torch.manual_seed(0)

    model = PaiNNModel(
        n_out=2,
        cutoff=0.1,
        atomic_numbers=[1, 8],
        n_bases=6,
        n_layers=2,
        n_hidden_channels=12,
        w_out_after_pool=True,
    )

    data = _data()
    ref = torch.tensor(
        [[0.012601337298479546, -0.0032668391572678087]] * 5
    )

    assert torch.allclose(model(data), ref)

    result = model.forward_node_feature(data)[:3].mean(
        dim=0,
        keepdim=True,
    )
    for layer in model.W_out:
        result = layer(result)

    ref = torch.tensor(
        [[0.012601337298479546, -0.0032668391572678087]]
    )
    assert torch.allclose(result, ref)

    traced = torch.jit.trace(
        model,
        example_inputs=create_graph_tracing_example(2),
    )
    assert torch.allclose(traced(data), model(data))


def test_painn_2(float64_default):
    torch.manual_seed(0)

    model = PaiNNModel(
        n_out=2,
        cutoff=0.1,
        long_range_cutoff=0.2,
        atomic_numbers=[1, 8],
        n_bases=6,
        n_layers=2,
        n_hidden_channels=12,
        w_out_after_pool=True,
    )

    data = _data()
    data["edge_masks_lr"] = torch.zeros(
        (data["edge_index"].shape[1], 1),
        dtype=bool,
    )
    data["edge_masks_lr"][:-6] = True

    ref = torch.tensor(
        [[0.0580523212375041, -0.0220428793692266]] * 4
        + [[0.0720856700379041, -0.0420276151917215]]
    )

    assert torch.allclose(model(data), ref)