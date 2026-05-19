"""Command-line example for computing a free-energy difference.

Example
-------
After installing the package in editable mode::

    pip install -e .
    mlcolvar-deltag COLVAR --cvs cv --state-a-bounds -2 -1 --state-b-bounds 1 2 --kbt 2.494

The command writes a NumPy ``.npz`` archive, a COLVAR-like text file
containing the interval coordinate and ``deltaG``, and a plot image.
"""

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
from mlcolvar.utils.fes import compute_deltaG


def _save_output(path: Path,
                 grid,
                 delta_g,
                 fields: Sequence[str],
                 state_a_bounds,
                 state_b_bounds,
                 bias_fields: Sequence[str] | None,
                 time_field: str | None):
    # Store both numerical results and enough metadata to identify the input fields.
    arrays = {"grid": np.asarray(grid),
              "deltaG": np.asarray(delta_g),
              "fields": np.asarray(fields),
              "state_a_bounds": np.asarray(state_a_bounds),
              "state_b_bounds": np.asarray(state_b_bounds),
              "bias_fields": np.asarray(bias_fields if bias_fields is not None else []),
              "time_field": np.asarray(time_field if time_field is not None else ""),
             }

    np.savez(path, **arrays)


def build_parser() -> argparse.ArgumentParser:
    # Keep argparse setup separate from main so tests can inspect the CLI without running it.
    parser = argparse.ArgumentParser(description=textwrap.dedent("""\
                                Compute a free-energy difference with mlcolvar.utils.fes.compute_deltaG.                            
                                It can be used in two ways:
                                    1. Writing the input as a YAML file and passed as --config. A template YAML file with all the options can be generated with --yaml-template [PATH].
                                    2. Passing the options as keywords, which can be listed with --help.
                                Whatever the mode, the used options are saved as a YAML file.
                                The function returns the interval coordinates and deltaG  as a COLVAR-like text file (.dat), a plot (.png), and a NumPy archive (.npz).
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
    input_output.add_argument("-o", "--output", type=Path, default=Path("deltaG"),
                              help=("Output prefix used for the .npz, .dat, .yaml and default .png files. "
                                    "Default: deltaG."))
    input_output.add_argument("--cvs", "--cv", dest="fields", nargs="+",
                              help="COLVAR field names to use as collective variables.")
    input_output.add_argument("--bias", dest="bias_fields", nargs="+",
                              help=("COLVAR bias field(s). If more than one is provided, values are summed. "
                                    "Default: all fields containing 'bias'."))
    input_output.add_argument("--time-field", "--time", dest="time_field",
                              help="COLVAR time field to use for the output grid. Default: frame index.")

    # Row slicing is delegated directly to load_dataframe.
    row_slicing = parser.add_argument_group("Row slicing options")
    row_slicing.add_argument("--start", type=int, default=0,
                             help="Read COLVAR rows starting from this index. Default: 0.")
    row_slicing.add_argument("--stop", type=int, help="Read COLVAR rows until this index. Default: end of file.")
    row_slicing.add_argument("--stride", type=int, default=1, help="Read every Nth COLVAR row. Default: 1.")

    # compute_deltaG requires exactly one thermal-energy specification.
    thermal_options = parser.add_argument_group("Thermal energy options")
    thermal = thermal_options.add_mutually_exclusive_group()
    thermal.add_argument("--kbt", type=float, help="Thermal energy in the desired deltaG units.")
    thermal.add_argument("--temp", type=float, help="Temperature in Kelvin.")
    thermal_options.add_argument("--units", choices=("kJ/mol", "kcal/mol", "eV"), default="kJ/mol",
                                 help="Free-energy units when using --temp. Default: kJ/mol.")

    # Parameters passed through to compute_deltaG.
    delta_g_options = parser.add_argument_group("DeltaG options")
    state_a_bounds = delta_g_options.add_argument("--state-a-bounds", "--bounds-a",
                                                  dest="state_a_bounds",
                                                  nargs="+",
                                                  type=float,
                                                  metavar="BOUND",
                                                  help=("State A bounds. Use 2 values for 1D: MIN MAX, "
                                                        "or 4 values for 2D: X_MIN X_MAX Y_MIN Y_MAX."))
    state_a_bounds.yaml_example = [0.0, 1.0]
    state_a_bounds.yaml_help = ("State A bounds as a YAML list of floats. Use [min, max] for 1D or "
                                "[x_min, x_max, y_min, y_max] for 2D.")

    state_b_bounds = delta_g_options.add_argument("--state-b-bounds", "--bounds-b",
                                                  dest="state_b_bounds",
                                                  nargs="+",
                                                  type=float,
                                                  metavar="BOUND",
                                                  help=("State B bounds. Use 2 values for 1D: MIN MAX, "
                                                        "or 4 values for 2D: X_MIN X_MAX Y_MIN Y_MAX."))
    state_b_bounds.yaml_example = [1.0, 2.0]
    state_b_bounds.yaml_help = ("State B bounds as a YAML list of floats. Use [min, max] for 1D or "
                                "[x_min, x_max, y_min, y_max] for 2D.")
    delta_g_options.add_argument("--intervals", "--ints", "--n-ints", dest="intervals", type=int, default=10,
                                 help="Number of intervals for progressive deltaG estimates. Default: 10.")
    delta_g_options.add_argument("--reverse", action="store_true", help="Reverse the input data. Default: false.")
    delta_g_options.add_argument("--eps", type=float, default=1e-8,
                                 help="Regularization for empty state counts. Default: 1e-8.")

    # Plotting controls.
    plotting = parser.add_argument_group("Plotting options")
    plotting.add_argument("--no-plot", action="store_true", help="Do not write a deltaG plot. Default: false.")
    plotting.add_argument("--plot-color", default="fessa6", help="Line color for the deltaG plot. Default: fessa6.")

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
    if args.state_a_bounds is None:
        parser.error("--state-a-bounds is required, either as command-line argument or in --config.")
    if args.state_b_bounds is None:
        parser.error("--state-b-bounds is required, either as command-line argument or in --config.")

    try:
        # Load and validate COLVAR fields before calling the numerical routine.
        data, bias, time, fields, bias_fields = load_colvar_data(args.input,
                                                                 args.fields,
                                                                 args.bias_fields,
                                                                 args.start,
                                                                 args.stop,
                                                                 args.stride,
                                                                 time_field=args.time_field,
                                                                 flatten_single_cv=True)
        dimensions = 1 if np.ndim(data) == 1 else data.shape[1]
        state_a_bounds = parse_min_max_bounds(args.state_a_bounds, dimensions, "state A bounds")
        state_b_bounds = parse_min_max_bounds(args.state_b_bounds, dimensions, "state B bounds")
    except ValueError as exc:
        parser.error(str(exc))

    output_prefix = get_output_prefix(args.output)
    data_output = get_output_path_with_suffix(output_prefix, suffix='npz')
    colvar_output = get_output_path_with_suffix(output_prefix, suffix='dat')
    yaml_output = get_output_path_with_suffix(output_prefix, suffix='yaml')
    plot_output = None if args.no_plot else get_output_path_with_suffix(output_prefix, suffix='png')

    # This is the only scientific computation done by the CLI wrapper.
    grid, delta_g = compute_deltaG(X=data,
                                   stateA_bounds=state_a_bounds,
                                   stateB_bounds=state_b_bounds,
                                   temp=args.temp,
                                   units=args.units,
                                   kbt=args.kbt,
                                   intervals=args.intervals,
                                   bias=bias,
                                   reverse=args.reverse,
                                   time=time,
                                   plot=plot_output is not None,
                                   plot_color=args.plot_color,
                                   eps=args.eps)

    # Always save the raw result arrays and a COLVAR-like text table; plotting is enabled by default.
    _save_output(data_output, grid, delta_g, fields, state_a_bounds, state_b_bounds, bias_fields, args.time_field)
    
    # colvar output
    coordinate_field = args.time_field if args.time_field is not None else "frame"
    save_colvar_table(colvar_output, [grid, delta_g], [coordinate_field, "deltaG"])
    
    # Record the effective options after defaults, YAML, and automatic bias-field detection.
    save_yaml_config(yaml_output, {"config": args.config,
                                   "input": args.input,
                                   "output": output_prefix,
                                   "cvs": fields,
                                   "bias": bias_fields,
                                   "time_field": args.time_field,
                                   "start": args.start,
                                   "stop": args.stop,
                                   "stride": args.stride,
                                   "kbt": args.kbt,
                                   "temp": args.temp,
                                   "units": args.units,
                                   "state_a_bounds": None if state_a_bounds is None else np.asarray(state_a_bounds).ravel().tolist(),
                                   "state_b_bounds": None if state_b_bounds is None else np.asarray(state_b_bounds).ravel().tolist(),
                                   "intervals": args.intervals,
                                   "reverse": args.reverse,
                                   "eps": args.eps,
                                   "no_plot": args.no_plot,
                                   "plot_color": args.plot_color})

    if plot_output is not None:
        plt.tight_layout()
        plt.savefig(plot_output, dpi=200)

    print(f"Used COLVAR fields: {', '.join(fields)}")
    if bias_fields is not None:
        print(f"Used bias fields: {', '.join(bias_fields)} using {args.units} as units")
    if args.time_field is not None:
        print(f"Used time field: {args.time_field}")
    print(f"Saved deltaG data to {data_output}")
    print(f"Saved deltaG COLVAR data to {colvar_output}")
    print(f"Saved deltaG YAML keywords to {yaml_output}")
    if plot_output is not None:
        print(f"Saved deltaG plot to {plot_output}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
