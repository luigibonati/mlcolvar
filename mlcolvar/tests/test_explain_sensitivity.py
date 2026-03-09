import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pytest
import torch

from mlcolvar.data import DictDataset
from mlcolvar.explain.sensitivity import plot_sensitivity, sensitivity_analysis, test_sensitivity_analysis
from mlcolvar.explain.graph_sensitivity import test_graph_sensitivity, test_get_cv_values_graph

def _sensitivity_dataset(with_labels: bool = True) -> DictDataset:
    random_generator = np.random.default_rng(10)
    input_data = torch.tensor(random_generator.normal(size=(40, 3)), dtype=torch.float32)
    dataset_payload = {"data": input_data}
    if with_labels:
        dataset_payload["labels"] = torch.tensor([0] * 20 + [1] * 20)
    dataset = DictDataset(dataset_payload)
    dataset.feature_names = np.asarray(["a", "b", "c"])
    return dataset


def _sensitivity_model() -> torch.nn.Module:
    # Simple differentiable map with scalar output.
    model = torch.nn.Sequential(torch.nn.Linear(3, 1, bias=True))
    with torch.no_grad():
        model[0].weight[:] = torch.tensor([[1.0, -2.0, 0.5]])
        model[0].bias[:] = torch.tensor([0.1])
    return model


def test_sensitivity_analysis():
    dataset = _sensitivity_dataset(with_labels=True)
    model = _sensitivity_model()
    feature_standard_deviations = np.ones(3)

    # Case 1: run core analysis without plotting for all metric aliases.
    for sensitivity_metric in ("mean_abs_val", "MAV", "root_mean_square", "RMS", "mean"):
        sensitivity_results = sensitivity_analysis(
            model,
            dataset,
            std=feature_standard_deviations,
            feature_names=None,
            metric=sensitivity_metric,
            per_class=False,
            plot_mode=None,
        )
        assert "Dataset" in sensitivity_results["sensitivity"]
        assert sensitivity_results["gradients"]["Dataset"].shape[1] == 3

    # Case 2: per-class + plotting in all supported modes.
    for plot_mode in ("violin", "barh", "scatter"):
        figure, axis = plt.subplots()
        sensitivity_results = sensitivity_analysis(
            model,
            dataset,
            std=feature_standard_deviations,
            feature_names=["x", "y", "z"],
            metric="MAV",
            per_class=True,
            plot_mode=plot_mode,
            ax=axis,
        )
        assert "State 0" in sensitivity_results["sensitivity"]
        assert "State 1" in sensitivity_results["sensitivity"]
        plt.close(figure)

    # Case 3: invalid metric should raise.
    with pytest.raises(NotImplementedError):
        sensitivity_analysis(model, dataset, std=feature_standard_deviations, metric="invalid", plot_mode=None)

    # Case 4: per_class without labels should raise.
    dataset_without_labels = _sensitivity_dataset(with_labels=False)
    with pytest.raises(KeyError):
        sensitivity_analysis(model, dataset_without_labels, std=feature_standard_deviations, per_class=True, plot_mode=None)


def test_plot_sensitivity():
    dataset = _sensitivity_dataset(with_labels=True)
    model = _sensitivity_model()
    feature_standard_deviations = np.ones(3)
    sensitivity_results = sensitivity_analysis(model, dataset, std=feature_standard_deviations, per_class=True, plot_mode=None)

    # Case 1: explicit plotting modes + max_features cut.
    for plot_mode in ("violin", "barh", "scatter"):
        figure, axis = plt.subplots()
        plot_sensitivity(sensitivity_results, mode=plot_mode, per_class=True, max_features=2, ax=axis)
        assert axis.get_xlabel() == "Sensitivity"
        plt.close(figure)

    # Case 2: invalid plot mode should raise.
    with pytest.raises(NotImplementedError):
        plot_sensitivity(sensitivity_results, mode="invalid", per_class=True)

    # Case 3: invalid per_class type should raise.
    with pytest.raises(TypeError):
        plot_sensitivity(sensitivity_results, mode="barh", per_class="yes")

    # Case 4: requesting per_class from dataset-only results should raise.
    dataset_only_results = sensitivity_analysis(
        model,
        dataset,
        std=feature_standard_deviations,
        per_class=False,
        plot_mode=None,
    )
    with pytest.raises(KeyError):
        plot_sensitivity(dataset_only_results, mode="barh", per_class=True)

if __name__ == "__main__":
    test_sensitivity_analysis()
    test_graph_sensitivity()
    test_get_cv_values_graph()
