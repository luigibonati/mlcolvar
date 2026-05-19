import numpy as np
import pytest
from mlcolvar.cli.delta_g import main


def test_delta_g_cli_prints_yaml_template(capsys, tmp_path):
    assert main(["--yaml-template"]) == 0

    text = capsys.readouterr().out
    assert text.startswith("# Input/output options\ninput: null")
    assert "# DeltaG options" in text
    assert "output: deltaG" in text
    assert "Output prefix" in text
    assert "State A bounds as a YAML list of floats" in text
    assert "state_a_bounds:" in text
    assert "\n  - 0\n  - 1\nstate_b_bounds:" in text
    assert "state_a_bounds: null" not in text
    assert "state_b_bounds:" in text
    assert "\n  - 1\n  - 2\nintervals:" in text
    assert "state_b_bounds: null" not in text
    assert "plot: null" not in text
    assert "yaml_template" not in text

    template = tmp_path / "template_delta_g.yaml"
    assert main(["--yaml-template", str(template)]) == 0
    assert capsys.readouterr().out == ""
    assert template.exists()
    assert template.read_text() == text


def test_delta_g_cli_writes_outputs(tmp_path, monkeypatch):
    # Use tmp_path so the CLI can read and write real files without touching the repository.
    colvar = tmp_path / "COLVAR"
    colvar.write_text(
        "#! FIELDS time cv opes.bias wall.bias\n"
        "0 -1.0 2.0 0.5\n"
        "1 1.0 3.0 0.5\n"
    )

    # Replace the numerical routine with a deterministic function so the test
    # focuses on CLI parsing, COLVAR loading, bias selection and output writing.
    def fake_compute_deltaG(**kwargs):
        # A single selected CV should be passed as the 1D array expected by compute_deltaG.
        assert kwargs["X"].shape == (2,)
        assert kwargs["X"].tolist() == [-1.0, 1.0]

        # Both data points are inside their corresponding state bounds.
        assert kwargs["stateA_bounds"] == (-1.5, -0.5)
        assert kwargs["stateB_bounds"] == (0.5, 1.5)

        # Because --bias is omitted, all COLVAR fields containing "bias" should be summed.
        np.testing.assert_allclose(kwargs["bias"], np.array([2.5, 3.5]))
        assert kwargs["intervals"] == 4
        assert kwargs["time"] is None
        assert kwargs["plot"] is True
        assert kwargs["plot_color"] == "fessa6"

        grid = np.array([0, 1])
        delta_g = np.array([0.2, 0.0])
        return grid, delta_g

    monkeypatch.setattr("mlcolvar.cli.delta_g.compute_deltaG", fake_compute_deltaG)

    output = tmp_path / "deltaG"
    config = tmp_path / "config.yaml"
    config.write_text(
        f"cvs: cv\n"
        f"kbt: 1.0\n"
        f"state-a-bounds: [-1.5, -0.5]\n"
        f"state-b-bounds: [0.5, 1.5]\n"
        f"intervals: 4\n"
        f"output: {output}\n"
    )

    # This test uses YAML aliases such as cvs and state-a-bounds to exercise
    # the config-file path. The input file remains positional and is the only
    # value allowed on the command line together with --config.
    main(["--config", str(config), str(colvar)])

    # The binary NumPy archive is the machine-readable output.
    assert output.with_suffix(".npz").exists()
    assert (tmp_path / "deltaG.png").exists()
    assert (tmp_path / "deltaG.yaml").exists()

    # The CLI also writes a COLVAR-like text file next to the .npz output by default.
    colvar_output = tmp_path / "deltaG.dat"
    assert colvar_output.exists()

    text = colvar_output.read_text()
    assert text.startswith("#! FIELDS frame deltaG")

    yaml_text = (tmp_path / "deltaG.yaml").read_text()
    assert f"output: {output}" in yaml_text
    assert "cvs:" in yaml_text
    assert "state_a_bounds:" in yaml_text
    assert "\nplot:" not in yaml_text


def test_delta_g_cli_warns_periodic_cv_from_colvar_header(tmp_path, monkeypatch):
    colvar = tmp_path / "COLVAR"
    colvar.write_text(
        "#! FIELDS time phi opes.bias\n"
        "#! SET min_phi -pi\n"
        "#! SET max_phi pi\n"
        "0 -0.5 0.5\n"
        "1 0.5 0.5\n"
    )

    def fake_compute_deltaG(**_kwargs):
        return np.array([0, 1]), np.array([0.0, 0.1])

    monkeypatch.setattr("mlcolvar.cli.delta_g.compute_deltaG", fake_compute_deltaG)

    output = tmp_path / "deltaG_periodic"

    with pytest.warns(UserWarning, match="periodic.*phi"):
        assert main([
            str(colvar),
            "--cvs", "phi",
            "--kbt", "1.0",
            "--state-a-bounds", "-1.0", "0.0",
            "--state-b-bounds", "0.0", "1.0",
            "--output", str(output),
            "--no-plot",
        ]) == 0

    assert output.with_suffix(".npz").exists()


def test_delta_g_cli_writes_2d_outputs_and_plot(tmp_path, monkeypatch):
    # Two CV columns let us check 2D data selection and one min/max pair per CV.
    # Both points are inside the state A/B bounds used below.
    colvar = tmp_path / "COLVAR"
    colvar.write_text(
        "#! FIELDS time cv1 cv2 opes.bias wall.bias\n"
        "0 1.0 0.5 2.0 0.5\n"
        "10 2.0 1.5 3.0 0.5\n"
    )

    def fake_compute_deltaG(**kwargs):
        # The CLI should preserve the requested CV ordering.
        np.testing.assert_allclose(kwargs["X"], np.array([[1.0, 0.5], [2.0, 1.5]]))

        # For 2D, bounds are parsed as one (min, max) tuple per selected CV.
        assert kwargs["stateA_bounds"] == [(0.0, 1.5), (0.0, 1.0)]
        assert kwargs["stateB_bounds"] == [(1.5, 2.5), (1.0, 2.0)]
        np.testing.assert_allclose(kwargs["bias"], np.array([2.5, 3.5]))

        # The time and plotting options should be forwarded exactly as parsed by the CLI.
        assert kwargs["time"].tolist() == [0, 10]
        assert kwargs["plot"] is True
        assert kwargs["plot_color"] == "C0"
        assert kwargs["reverse"] is True
        assert kwargs["eps"] == 1e-6

        grid = np.array([0, 10])
        delta_g = np.array([0.4, 0.0])
        return grid, delta_g

    monkeypatch.setattr("mlcolvar.cli.delta_g.compute_deltaG", fake_compute_deltaG)

    output = tmp_path / "deltaG_2d"

    main([
        str(colvar),
        "--cvs", "cv1", "cv2",
        "--kbt", "1.0",
        "--state-a-bounds", "0.0", "1.5", "0.0", "1.0",
        "--state-b-bounds", "1.5", "2.5", "1.0", "2.0",
        "--time-field", "time",
        "--reverse",
        "--eps", "1e-6",
        "--output", str(output),
        "--plot-color", "C0",
    ])

    assert output.with_suffix(".npz").exists()
    assert (tmp_path / "deltaG_2d.png").exists()
    assert (tmp_path / "deltaG_2d.yaml").exists()

    colvar_output = tmp_path / "deltaG_2d.dat"
    assert colvar_output.exists()

    text = colvar_output.read_text()
    assert text.startswith("#! FIELDS time deltaG")
