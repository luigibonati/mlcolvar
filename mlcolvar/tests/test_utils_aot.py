import torch

from mlcolvar.core.nn.graph import SchNetModel
from mlcolvar.data.graph.utils import create_test_graph_input
from mlcolvar.utils import aot


def test_aot_export_gnn(tmp_path) -> None:
    old_dtype = torch.get_default_dtype()
    torch.manual_seed(0)
    torch.set_default_dtype(torch.float32)

    try:
        model = SchNetModel(
            n_out=4,
            cutoff=0.1,
            atomic_numbers=[1, 8],
            n_bases=6,
            n_layers=2,
            n_filters=16,
            n_hidden_channels=16,
        )

        model.dtype = torch.float32
        model.device = "cpu"

        batch = create_test_graph_input(
            output_type="batch",
            n_atoms=3,
            n_samples=6,
            n_states=1,
            add_noise=False,
        )["data_list"]

        dataset = batch.to_data_list()[0]
        output_path = tmp_path / "model.pt2"

        result = aot.export(
            model,
            example_inputs=dataset,
            file_name=str(output_path),
            run_check=True,
        )

        assert output_path.exists()
        assert str(result).endswith(".pt2")

        compiled_model = aot.load(str(output_path))
        metadata = compiled_model.get_metadata()

        assert metadata["n_cvs"] == "4"
        assert metadata["n_outputs"] == "4"

    finally:
        torch.set_default_dtype(old_dtype)