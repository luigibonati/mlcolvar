import numpy as np
from mlcolvar.cli.fes import main


def test_fes_cli_writes_outputs(tmp_path, monkeypatch):
    # Use tmp_path so the CLI can read and write real files without touching the repository.
    colvar = tmp_path / "COLVAR"
    colvar.write_text(
        "#! FIELDS time cv opes.bias wall.bias\n"
        "0 1.0 2.0 0.5\n"
        "1 2.0 3.0 0.5\n"
    )

    # Replace the expensive numerical routine with a small deterministic function.
    # This keeps the test focused on the CLI: argument parsing, COLVAR loading,
    # automatic bias selection, and output writing.
    def fake_compute_fes(**kwargs):
        # The CLI should select only the requested CV column.
        assert kwargs["X"].shape == (2, 1)
        assert kwargs["X"].ravel().tolist() == [1.0, 2.0]

        # The test data points are inside these bounds, so this checks the CLI
        # parses and forwards a valid 1D bounds specification.
        assert kwargs["bounds"] == (0.0, 3.0)

        # Because --bias is omitted, all COLVAR fields containing "bias" should be summed.
        assert kwargs["bias"].tolist() == [2.5, 3.5]

        # Return a simple 1D FES without block errors so the text output has no error column.
        grid = np.array([0.0, 1.0])
        fes = np.array([1.2, 0.0])
        return fes, grid, [(0.0, 1.0)], None

    monkeypatch.setattr("mlcolvar.cli.fes.compute_fes", fake_compute_fes)

    output = tmp_path / "fes.npz"

    main([
        str(colvar),
        "--columns", "cv",
        "--kbt", "1.0",
        "--bounds", "0.0", "3.0",
        "--output", str(output),
    ])

    # The binary NumPy archive is the machine-readable output.
    assert output.exists()

    # The CLI also writes a COLVAR-like text file next to the .npz output by default.
    colvar_output = tmp_path / "fes.dat"
    assert colvar_output.exists()

    # Since fake_compute_fes returned error=None, the text header should not expose an error field.
    text = colvar_output.read_text()
    assert text.startswith("#! FIELDS cv fes")
    assert "error" not in text.splitlines()[0]


def test_fes_cli_writes_2d_outputs_with_error(tmp_path, monkeypatch):
    # Two CV columns let us check the 2D data selection and the min/max pair
    # required for each dimension. Both points are inside the bounds below.
    colvar = tmp_path / "COLVAR"
    colvar.write_text(
        "#! FIELDS time cv1 cv2 opes.bias wall.bias\n"
        "0 1.0 0.5 2.0 0.5\n"
        "1 2.0 1.5 3.0 0.5\n"
    )

    def fake_compute_fes(**kwargs):
        # The CLI should preserve the requested CV ordering.
        np.testing.assert_allclose(kwargs["X"], np.array([[1.0, 0.5], [2.0, 1.5]]))

        # For 2D, bounds are parsed as one (min, max) tuple per selected CV.
        assert kwargs["bounds"] == [(0.0, 3.0), (0.0, 2.0)]
        assert kwargs["bias"].tolist() == [2.5, 3.5]

        # The plotting options should be forwarded to compute_fes exactly as parsed by the CLI.
        assert kwargs["plot"] is True
        assert kwargs["plot_max_fes"] == 2.0
        assert kwargs["plot_levels"] == 5

        x_grid, y_grid = np.meshgrid(np.array([0.0, 3.0]), np.array([0.0, 2.0]))
        fes = np.array([[1.2, 0.8], [0.4, 0.0]])
        error = np.array([[0.1, 0.2], [0.3, 0.4]])
        return fes, [x_grid, y_grid], [(0.0, 3.0), (0.0, 2.0)], error

    monkeypatch.setattr("mlcolvar.cli.fes.compute_fes", fake_compute_fes)

    output = tmp_path / "fes_2d.npz"
    plot = tmp_path / "fes_2d.png"

    main([
        str(colvar),
        "--columns", "cv1", "cv2",
        "--kbt", "1.0",
        "--bounds", "0.0", "3.0", "0.0", "2.0",
        "--output", str(output),
        "--plot", str(plot),
        "--plot-max-fes", "2.0",
        "--plot-levels", "5",
    ])

    assert output.exists()
    assert plot.exists()

    colvar_output = tmp_path / "fes_2d.dat"
    assert colvar_output.exists()

    # With block errors available, the COLVAR-like text output should expose an error column.
    text = colvar_output.read_text()
    assert text.startswith("#! FIELDS cv1 cv2 fes error")
