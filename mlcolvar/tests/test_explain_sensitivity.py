import lightning
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pytest
import torch

from mlcolvar.core.nn.graph import SchNetModel
from mlcolvar.cvs import DeepLDA, DeepTDA
from mlcolvar.data import DictDataset, DictModule
from mlcolvar.data.graph.utils import create_test_graph_input
from mlcolvar.explain.graph_sensitivity import (
    get_dataset_cv_values,
    graph_node_sensitivity,
)
from mlcolvar.explain.sensitivity import (
    plot_sensitivity,
    sensitivity_analysis,
)


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


def test_sensitivity_analysis_deeplda():
    n_states = 2
    in_features, out_features = 2, n_states - 1
    layers = [in_features, 5, 5, out_features]

    # create dataset
    samples = 10
    X = torch.randn((samples * n_states, 2))

    # create labels
    y = torch.zeros(X.shape[0])
    for i in range(1, n_states):
        y[samples * i :] += 1

    dataset = DictDataset({"data": X, "labels": y})

    # define CV
    opts = {
        "nn": {"activation": "shifted_softplus"},
    }
    model = DeepLDA(layers, n_states, options=opts)

    # feature importances
    for per_class in [True, False, None]:
        for names in [None, ["x", "y"], np.asarray(["x", "y"])]:
            results = sensitivity_analysis(
                model, dataset, feature_names=names, per_class=per_class, plot_mode=None
            )


def test_get_cv_values_graph():
    # create data, we need the dataset for sensitivity analysis later
    dataset = create_test_graph_input(output_type='dataset', n_samples=50, n_states=2, n_atoms=3)
    datamodule = DictModule(dataset=dataset, lengths=[0.8, 0.2], shuffle=[1, 0])

    # create model
    gnn_model = SchNetModel(n_out=1, cutoff=0.1, atomic_numbers=[8, 1])
    model = DeepTDA(
        n_states=2,
        n_cvs=1,
        target_centers=[-5, 5],
        target_sigmas=[0.2, 0.2],
        model=gnn_model
    )

    # train model
    trainer = lightning.Trainer(
        accelerator="cpu", max_epochs=2, logger=False, enable_checkpointing=False, enable_model_summary=False
    )
    trainer.fit(model, datamodule)

    # do analysis
    cv_values = get_dataset_cv_values(model=model, dataset=dataset, batch_size=0)

    # print results
    print(cv_values)

    assert (torch.allclose(model(dataset.get_graph_inputs()), torch.Tensor(cv_values)))


def test_graph_sensitivity():
    for environment in [False, True]:
        # create data, we need the dataset for sensitivity analysis later
        dataset = create_test_graph_input(output_type='dataset', n_samples=100, n_states=2, n_atoms=3, environment=environment)
        datamodule = DictModule(dataset=dataset, lengths=[0.8, 0.2], shuffle=[1, 0])

        # create model
        gnn_model = SchNetModel(n_out=1, cutoff=0.1, atomic_numbers=[8, 1])
        model = DeepTDA(
            n_states=2,
            n_cvs=1,
            target_centers=[-5, 5],
            target_sigmas=[0.2, 0.2],
            model=gnn_model
        )

        # train model
        trainer = lightning.Trainer(
            accelerator="cpu", max_epochs=2, logger=False, enable_checkpointing=False, enable_model_summary=False
        )
        trainer.fit(model, datamodule)

        # do analysis
        test_sensitivity = graph_node_sensitivity(model=model,
                                        dataset=dataset,
                                        batch_size=0)

        # print results
        print(test_sensitivity)
