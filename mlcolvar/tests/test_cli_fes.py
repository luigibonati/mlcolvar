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

if __name__ == "__main__":
    test_fes_cli_writes_outputs()
