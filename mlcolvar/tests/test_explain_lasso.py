import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pytest
import torch

from mlcolvar.data import DictDataset
from mlcolvar.explain.lasso import (
    SparsityScoring,
    test_lasso_classification,
    test_lasso_regression,
    lasso_classification,
    lasso_regression,
    plot_lasso_classification,
    plot_lasso_regression,
)


def _classification_dataset(number_of_states: int = 2, number_of_samples: int = 60) -> DictDataset:
    # Build a linearly separable toy dataset for stable sparse classification.
    random_generator = np.random.default_rng(0)
    input_data = random_generator.normal(size=(number_of_samples, 3))
    class_labels = np.zeros(number_of_samples, dtype=int)
    samples_per_state = number_of_samples // number_of_states
    for state_index in range(number_of_states):
        class_labels[state_index * samples_per_state : (state_index + 1) * samples_per_state] = state_index
        input_data[state_index * samples_per_state : (state_index + 1) * samples_per_state, state_index % 3] += 3.0
    dataset = DictDataset({"data": torch.tensor(input_data, dtype=torch.float32), "labels": torch.tensor(class_labels)})
    dataset.feature_names = np.asarray(["f1", "f2", "f3"])
    return dataset


def _regression_dataset(number_of_samples: int = 80) -> DictDataset:
    # Target depends on a sparse linear combination of 3 features.
    random_generator = np.random.default_rng(1)
    input_data = random_generator.normal(size=(number_of_samples, 3))
    target_values = 2.0 * input_data[:, 0] - 1.5 * input_data[:, 2] + 0.1 * random_generator.normal(size=number_of_samples)
    dataset = DictDataset({"data": torch.tensor(input_data, dtype=torch.float32), "target": torch.tensor(target_values, dtype=torch.float32)})
    dataset.feature_names = np.asarray(["f1", "f2", "f3"])
    return dataset


def test_sparsity_scoring():
    # Case 1: direct score call on a fitted sklearn estimator.
    dataset = _classification_dataset(number_of_states=2, number_of_samples=50)
    classifier, _, _ = lasso_classification(dataset, Cs=[1.0], min_features=1, print_info=False, plot=False)
    scorer = SparsityScoring(min_features=1)
    sparsity_score = scorer(classifier, dataset["data"].numpy(), dataset["labels"].numpy().astype(int))
    assert np.isfinite(sparsity_score)

    # Case 2: recover accuracy from score for a known configuration.
    recovered_accuracy = scorer.accuracy_from_score(score=-2.0, num_features=1)
    assert np.isclose(recovered_accuracy, 0.98)


def test_lasso_classification_2():
    # Case 1: binary and multiclass runs return populated feature/coeff dictionaries.
    for number_of_states in (2, 3):
        dataset = _classification_dataset(number_of_states=number_of_states, number_of_samples=60)
        classifier, selected_features, selected_coefficients = lasso_classification(
            dataset,
            min_features=1,
            Cs=np.logspace(-2, 1, 6),
            scale_inputs=True,
            print_info=False,
            plot=False,
        )
        assert len(classifier.C_) >= 1
        assert len(selected_features) == len(selected_coefficients) >= 1

        # Case 2: plotting path with provided axes.
        number_of_models = len(classifier.C_)
        figure, axes = plt.subplots(3, number_of_models if number_of_models > 1 else 1, squeeze=False)
        axes_for_plot = axes[:, 0] if number_of_models == 1 else axes
        plot_lasso_classification(
            classifier,
            feats=selected_features,
            coeffs=selected_coefficients,
            draw_labels=False,
            axs=axes_for_plot,
        )
        plt.close(figure)

    # Case 3: single-C classifier triggers early-return plot branch.
    dataset_single_regularization = _classification_dataset(number_of_states=2, number_of_samples=40)
    single_regularization_classifier, single_regularization_features, single_regularization_coefficients = lasso_classification(
        dataset_single_regularization, Cs=[0.1], min_features=0, print_info=False, plot=False
    )
    plot_lasso_classification(
        single_regularization_classifier,
        single_regularization_features,
        single_regularization_coefficients,
    )

    # Case 4: missing feature names must raise.
    dataset_without_feature_names = _classification_dataset(number_of_states=2, number_of_samples=40)
    dataset_without_feature_names.feature_names = None
    with pytest.raises(ValueError):
        lasso_classification(dataset_without_feature_names, print_info=False, plot=False)


def test_lasso_regression_2():
    # Case 1: nominal regression with plotting disabled.
    dataset = _regression_dataset()
    regressor, selected_features, selected_coefficients = lasso_regression(
        dataset,
        alphas=np.logspace(-3, -1, 6),
        scale_inputs=True,
        print_info=False,
        plot=False,
    )
    assert regressor.alpha_ > 0
    assert len(selected_features) == len(selected_coefficients)

    # Case 2: regression plotting path with provided axes.
    figure, axes = plt.subplots(3, 1)
    plot_lasso_regression(regressor, feats=selected_features, coeffs=selected_coefficients, draw_labels=False, axs=axes)
    plt.close(figure)

    # Case 3: single-alpha branch early-return in plotting.
    single_regularization_regressor, single_regularization_features, single_regularization_coefficients = lasso_regression(
        dataset, alphas=[0.1], scale_inputs=True, print_info=False, plot=False
    )
    plot_lasso_regression(
        single_regularization_regressor,
        single_regularization_features,
        single_regularization_coefficients,
    )

    # Case 4: invalid multi-target shape should raise.
    random_generator = np.random.default_rng(3)
    input_data = random_generator.normal(size=(40, 3)).astype(np.float32)
    invalid_target_values = random_generator.normal(size=(40, 2)).astype(np.float32)
    dataset_with_invalid_target_shape = DictDataset({"data": torch.tensor(input_data), "target": torch.tensor(invalid_target_values)})
    dataset_with_invalid_target_shape.feature_names = np.asarray(["f1", "f2", "f3"])
    with pytest.raises(ValueError):
        lasso_regression(dataset_with_invalid_target_shape, print_info=False, plot=False)


def test_lasso_print(capsys):
    classification_dataset = _classification_dataset(number_of_states=2, number_of_samples=50)
    lasso_classification(
        classification_dataset,
        min_features=1,
        Cs=np.logspace(-2, 1, 6),
        print_info=True,
        plot=False,
    )
    classification_stdout = capsys.readouterr().out
    assert "LASSO results" in classification_stdout
    assert "Features:" in classification_stdout

    regression_dataset = _regression_dataset(number_of_samples=60)
    lasso_regression(
        regression_dataset,
        alphas=np.logspace(-3, -1, 6),
        print_info=True,
        plot=False,
    )
    regression_stdout = capsys.readouterr().out
    assert "LASSO results" in regression_stdout
    assert "Relevant features" in regression_stdout

    single_c_classifier, single_c_features, single_c_coefficients = lasso_classification(
        classification_dataset, Cs=[0.1], min_features=0, print_info=False, plot=False
    )
    plot_lasso_classification(single_c_classifier, single_c_features, single_c_coefficients)
    assert "Plotting is not available" in capsys.readouterr().out

    single_alpha_regressor, single_alpha_features, single_alpha_coefficients = lasso_regression(
        regression_dataset, alphas=[0.1], print_info=False, plot=False
    )
    plot_lasso_regression(single_alpha_regressor, single_alpha_features, single_alpha_coefficients)
    assert "Plotting is not available" in capsys.readouterr().out

if __name__ == "__main__":
    test_lasso_classification()
    test_lasso_regression()
