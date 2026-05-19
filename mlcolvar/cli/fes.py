"""Command-line example for computing a free energy surface.

Example
-------
After installing the package in editable mode::

    pip install -e .
    mlcolvar-fes COLVAR --cvs phi psi --kbt 2.494 --bandwidth 0.05

The command writes a NumPy ``.npz`` archive and a COLVAR-like text file
containing the grid coordinates, ``fes`` and ``error``.
"""

from __future__ import annotations

import argparse
import textwrap
from pathlib import Path
from typing import Sequence

import matplotlib

# Use a non-interactive backend so the CLI can run on machines without a display.
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from mlcolvar.cli.utils import (YAML_CONFIG_ALIASES,
                                YAML_TEMPLATE_ALIASES,
                                flatten_min_max_bounds,
                                get_output_prefix,
                                get_output_path_with_suffix,
                                load_colvar_data,
                                parse_args_with_yaml_config,
                                parse_min_max_bounds,
                                save_colvar_table,
                                save_yaml_config,
                                validate_common_args,
                                write_yaml_template)
from mlcolvar.utils import plot as _plot_utils  # noqa: F401 - registers fessa colormap
from mlcolvar.utils.fes import compute_fes


def _parse_bounds(values: Sequence[float] | None, dimensions: int):
    # compute_fes expects one min/max pair for each selected CV dimension.
    if values is None:
        return None

    return parse_min_max_bounds(values, dimensions, "bounds")


def _save_output(path: Path,
                 fes,
                 grid,
                 bounds,
                 error,
                 fields: Sequence[str],
                 bias_fields: Sequence[str] | None):
    # Store both numerical results and enough metadata to identify the input fields.
    arrays = {"fes": np.asarray(fes),
              "bounds": np.asarray(bounds),
              "error": np.asarray(error) if error is not None else np.array([]),
              "fields": np.asarray(fields),
              "bias_fields": np.asarray(bias_fields if bias_fields is not None else []),
             }

    # In 1D compute_fes returns one grid array; in higher dimensions it returns one grid per axis.
    if isinstance(grid, list):
        arrays.update({f"grid_{i}": np.asarray(axis) for i, axis in enumerate(grid)})
    else:
        arrays["grid"] = np.asarray(grid)

    np.savez(path, **arrays)


def _save_colvar_output(path: Path, fes, grid, error, fields: Sequence[str]):
    # Write a PLUMED-like text table that can be inspected with standard tools.
    grid_columns = ([np.asarray(axis).ravel() for axis in grid]
                    if isinstance(grid, list) else [np.asarray(grid).ravel()])
    fes_column = np.asarray(fes).ravel()
    columns = [*grid_columns, fes_column]
    output_fields = [*fields, "fes"]

    if error is not None:
        columns.append(np.asarray(error).ravel())
        output_fields.append("error")

    save_colvar_table(path, columns, output_fields)


def build_parser() -> argparse.ArgumentParser:
    # Keep argparse setup separate from main so tests can inspect the CLI without running it.
    parser = argparse.ArgumentParser(description=textwrap.dedent("""\
                                    Compute a free energy surface with mlcolvar.utils.fes.compute_fes.                            
                                    It can be used in two ways:
                                        1. Writing the input as a YAML file and passed as --config. A template YAML file with all the options can be generated with --yaml-template [PATH].
                                        2. Passing the options as keywords, which can be listed with --help.
                                    Whatever the mode, the used options are saved as a YAML file.
                                    The function returns the grid coordinates, fes and error as a COLVAR-like text file (.dat), a plot, and a NumPy archive (.npz).
                                    """),
                                    formatter_class=argparse.RawDescriptionHelpFormatter)

    # Input/output options.
    input_output = parser.add_argument_group("Input/output options")
    input_output.add_argument("input", nargs="*", help="PLUMED COLVAR file(s).")
    input_output.add_argument("--config", type=Path,
                              help="YAML file with CLI options; exclusive with other options. Default: none.")
    input_output.add_argument("--yaml-template", nargs="?", const="-", metavar="PATH",
                              help=("Print the YAML configuration template, or write it to PATH, and exit. "
                                    "Default: false."))
    input_output.add_argument("-o", "--output", type=Path, default=Path("fes"),
                              help=("Output prefix used for the .npz, .dat, .yaml and default .png files. "
                                    "Default: fes."))
    input_output.add_argument("--cvs", "--cv", dest="fields", nargs="+",
                              help="COLVAR field names to use as collective variables.")
    input_output.add_argument("--bias", dest="bias_fields", nargs="+",
                              help=("COLVAR bias field(s). If more than one is provided, values are summed. "
                                    "Default: all fields containing 'bias'."))

    # Row slicing is delegated directly to load_dataframe.
    row_slicing = parser.add_argument_group("Row slicing options")
    row_slicing.add_argument("--start", type=int, default=0,
                             help="Read COLVAR rows starting from this index. Default: 0.")
    row_slicing.add_argument("--stop", type=int, help="Read COLVAR rows until this index. Default: end of file.")
    row_slicing.add_argument("--stride", type=int, default=1, help="Read every Nth COLVAR row. Default: 1.")

    # compute_fes requires exactly one thermal-energy specification.
    thermal_options = parser.add_argument_group("Thermal energy options")
    thermal = thermal_options.add_mutually_exclusive_group()
    thermal.add_argument("--kbt", type=float, help="Thermal energy in the desired FES units.")
    thermal.add_argument("--temp", type=float, help="Temperature in Kelvin.")
    thermal_options.add_argument("--units", choices=("kJ/mol", "kcal/mol", "eV"), default="kJ/mol",
                                 help="Free-energy units when using --temp. Default: kJ/mol.")

    # Parameters passed through to compute_fes.
    fes_options = parser.add_argument_group("FES options")
    fes_options.add_argument("--num-samples", type=int, default=200,
                             help="Grid points per dimension. Default: 200.")
    fes_options.add_argument("--bounds", nargs="+", type=float,
                             help="Bounds as 'min max' for 1D or 'x_min x_max y_min y_max ...' "
                                  "for higher dimensions. Default: data range.")
    fes_options.add_argument("--bandwidth", "--bw", type=float, default=0.01, help="KDE bandwidth. Default: 0.01.")
    fes_options.add_argument("--kernel", default="gaussian", help="KDE kernel. Default: gaussian.")
    fes_options.add_argument("--scale-by", choices=("std", "range"),
                             help="Scale input variables before KDE. Default: none.")
    fes_options.add_argument("--blocks", type=int, default=1,
                             help="Number of blocks for uncertainty estimates. Default: 1.")
    fes_options.add_argument("--backend", choices=("KDEpy", "sklearn"), help="KDE backend. Default: best available.")
    fes_options.add_argument("--eps", type=float,
                             help="Regularization added before taking the logarithm. Default: auto-tuned.")

    # Plotting controls.
    plotting = parser.add_argument_group("Plotting options")
    plotting.add_argument("--no-plot", action="store_true", help="Do not write a FES plot. Default: false.")
    plotting.add_argument("--plot-color", default="fessa6", help="Line color for the FES plot. Default: fessa6.")
    plotting.add_argument("--plot-max-fes", type=float,
                          help="Mask plot values above this FES. Default: no masking.")
    plotting.add_argument("--plot-levels", type=int,
                          help="Contour levels for 2D plots. Default: matplotlib default.")

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    try:
        args = parse_args_with_yaml_config(parser, argv, aliases=YAML_CONFIG_ALIASES)
    except ValueError as exc:
        parser.error(str(exc))

    if args.yaml_template:
        write_yaml_template(parser, args.yaml_template, aliases=YAML_TEMPLATE_ALIASES)
        return 0

    validate_common_args(parser, args)

    try:
        # Load and validate COLVAR fields before calling the numerical routine.
        data, bias, _time, fields, bias_fields = load_colvar_data(args.input,
                                                                  args.fields,
                                                                  args.bias_fields,
                                                                  args.start,
                                                                  args.stop,
                                                                  args.stride)
        dimensions = 1 if data.ndim == 1 else data.shape[1]
        bounds = _parse_bounds(args.bounds, dimensions)
    except ValueError as exc:
        parser.error(str(exc))

    output_prefix = get_output_prefix(args.output)
    data_output = get_output_path_with_suffix(output_prefix, suffix='npz')
    colvar_output = get_output_path_with_suffix(output_prefix, suffix='dat')
    yaml_output = get_output_path_with_suffix(output_prefix, suffix='yaml')
    plot_output = None if args.no_plot else get_output_path_with_suffix(output_prefix, suffix='png')

    # This is the only scientific computation done by the CLI wrapper.
    fes, grid, used_bounds, error = compute_fes(X=data,
                                                temp=args.temp,
                                                units=args.units,
                                                kbt=args.kbt,
                                                num_samples=args.num_samples,
                                                bounds=bounds,
                                                bandwidth=args.bandwidth,
                                                kernel=args.kernel,
                                                bias=bias,
                                                scale_by=args.scale_by,
                                                blocks=args.blocks,
                                                plot=plot_output is not None,
                                                plot_color=args.plot_color,
                                                plot_max_fes=args.plot_max_fes,
                                                plot_levels=args.plot_levels,
                                                backend=args.backend,
                                                eps=args.eps)

    # Always save the raw result arrays and a COLVAR-like text table; plotting is optional.
    _save_output(data_output, fes, grid, used_bounds, error, fields, bias_fields)
    _save_colvar_output(colvar_output, fes, grid, error, fields)
    # Record the effective options after defaults, YAML, and automatic bias-field detection.
    save_yaml_config(yaml_output, {"config": args.config,
                                   "input": args.input,
                                   "output": output_prefix,
                                   "cvs": fields,
                                   "bias": bias_fields,
                                   "start": args.start,
                                   "stop": args.stop,
                                   "stride": args.stride,
                                   "kbt": args.kbt,
                                   "temp": args.temp,
                                   "units": args.units,
                                   "num_samples": args.num_samples,
                                   "bounds": flatten_min_max_bounds(bounds),
                                   "bandwidth": args.bandwidth,
                                   "kernel": args.kernel,
                                   "scale_by": args.scale_by,
                                   "blocks": args.blocks,
                                   "backend": args.backend,
                                   "eps": args.eps,
                                   "no_plot": args.no_plot,
                                   "plot_color": args.plot_color,
                                   "plot_max_fes": args.plot_max_fes,
                                   "plot_levels": args.plot_levels})

    if plot_output is not None:
        plt.tight_layout()
        plt.savefig(plot_output, dpi=200)

    print(f"Used COLVAR fields: {', '.join(fields)}")
    if bias_fields is not None:
        print(f"Used bias fields: {', '.join(bias_fields)} using {args.units} as units")
    print(f"Saved FES data to {data_output}")
    print(f"Saved FES COLVAR data to {colvar_output}")
    print(f"Saved FES YAML keywords to {yaml_output}")
    if plot_output is not None:
        print(f"Saved FES plot to {plot_output}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
