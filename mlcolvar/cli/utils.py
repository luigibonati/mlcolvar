"""Shared helpers for mlcolvar command-line interfaces."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import yaml

from mlcolvar.utils.io.colvar import load_dataframe


_BOUND_KEYS = {"bounds", "state_a_bounds", "state_b_bounds"}
# These option families need light type normalization when they are read from YAML.
# argparse normally performs this conversion for command-line values.
_LIST_KEYS = {"input", "fields", "bias_fields", *_BOUND_KEYS}
_PATH_KEYS = {"config", "output", "output_colvar", "output_yaml", "plot"}


def parse_min_max_bounds(values: Sequence[float], dimensions: int, name: str):
    # CLI bounds are passed as min/max pairs, one pair for each selected CV dimension.
    expected = 2 * dimensions
    if len(values) != expected:
        raise ValueError(f"Expected {expected} values for {name} with {dimensions}D data "
                         f"(min max for each dimension), got {len(values)}.")

    if dimensions == 1:
        return (values[0], values[1])

    return [(values[i], values[i + 1]) for i in range(0, expected, 2)]


def load_colvar_data(file_names: Sequence[str],
                     fields: Sequence[str],
                     bias_fields: Sequence[str] | None,
                     start: int,
                     stop: int | None,
                     stride: int,
                     time_field: str | None = None,
                     flatten_single_cv: bool = False):
    # load_dataframe understands PLUMED COLVAR headers and returns named columns.
    dataframe = load_dataframe(file_names=file_names[0] if len(file_names) == 1 else list(file_names),
                               start=start,
                               stop=stop,
                               stride=stride)

    # Fail early with the available field names when a requested field is missing.
    missing = []
    missing.extend(field for field in fields if field not in dataframe.columns)
    if bias_fields is not None:
        missing.extend(field for field in bias_fields if field not in dataframe.columns)
    if time_field is not None and time_field not in dataframe.columns:
        missing.append(time_field)
    if missing:
        available = ", ".join(dataframe.columns)
        raise ValueError(f"Field(s) not found in COLVAR data: {', '.join(missing)}. Available fields: {available}.")

    if bias_fields is None:
        bias_fields = [column for column in dataframe.columns if "bias" in column.lower()]

    # Numerical routines work on NumPy arrays, so the dataframe is only used for loading/selection.
    data = dataframe.loc[:, fields].to_numpy()
    if flatten_single_cv and len(fields) == 1:
        data = data.ravel()
    bias = dataframe.loc[:, bias_fields].to_numpy().sum(axis=1) if bias_fields else None
    time = dataframe.loc[:, time_field].to_numpy() if time_field is not None else None

    return data, bias, time, list(fields), list(bias_fields) if bias_fields else None


def get_colvar_output_path(output: Path, output_colvar: Path | None) -> Path:
    return output_colvar if output_colvar is not None else output.with_suffix(".dat")


def get_yaml_output_path(output: Path, output_yaml: Path | None) -> Path:
    return output_yaml if output_yaml is not None else output.with_suffix(".yaml")


def _yaml_template_key(action: argparse.Action, aliases: Mapping[str, str]) -> str:
    return aliases.get(action.dest, action.dest)


def _yaml_template_value(value: Any) -> str:
    if value is None or value is argparse.SUPPRESS:
        return "null"
    if isinstance(value, bool):
        return str(value).lower()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float):
        return np.format_float_positional(value, trim="-")
    if isinstance(value, list):
        return f"[{', '.join(_yaml_template_value(item) for item in value)}]"

    return str(value)


def _yaml_template_comment(help_text: str | None) -> str:
    return "" if help_text in (None, argparse.SUPPRESS) else " ".join(help_text.split())


def yaml_template_from_parser(parser: argparse.ArgumentParser,
                              aliases: Mapping[str, str] | None = None,
                              skip: Sequence[str] = ("help", "config", "yaml_template")) -> str:
    aliases = aliases or {}
    rows = []
    for action in parser._actions:
        if action.dest in skip or action.dest == argparse.SUPPRESS:
            continue
        key = _yaml_template_key(action, aliases)
        value = _yaml_template_value(action.default)
        rows.append((f"{key}: {value}", _yaml_template_comment(action.help)))

    comment_column = max(max(len(prefix) for prefix, _comment in rows) + 2, 22)
    lines = [f"{prefix.ljust(comment_column)}# {comment}".rstrip() for prefix, comment in rows]
    return "\n".join(lines) + "\n"


def write_yaml_template(parser: argparse.ArgumentParser,
                        output: str | Path | None = None,
                        aliases: Mapping[str, str] | None = None):
    text = yaml_template_from_parser(parser, aliases=aliases)
    if output in (None, "-"):
        sys.stdout.write(text)
        return

    Path(output).write_text(text)


def flatten_min_max_bounds(bounds):
    # Keep YAML bounds in the same flat format accepted by argparse: min max [min max ...].
    return None if bounds is None else np.asarray(bounds).ravel().tolist()


def save_colvar_table(path: Path, columns: Sequence[np.ndarray], fields: Sequence[str]):
    # Write a PLUMED-like text table that can be inspected with standard tools.
    table = np.column_stack([np.asarray(column).ravel() for column in columns])
    header = f"#! FIELDS {' '.join(fields)}"
    np.savetxt(path, table, header=header, comments="", fmt=" %.10g")


def _normalize_config_key(key: str, aliases: Mapping[str, str]) -> str:
    # YAML can use command-line spelling, such as state-a-bounds, or user-facing aliases like cvs.
    normalized = key.replace("-", "_")
    return aliases.get(normalized, normalized)


def _normalize_config_value(key: str, value: Any):
    # YAML scalars are convenient for one-value CLI options; argparse stores these as lists.
    if key in _LIST_KEYS and value is not None and not isinstance(value, list):
        value = [value]

    if key in _BOUND_KEYS and value is not None:
        # Bounds may be written as flat lists or nested pairs; compute functions receive flat CLI-style values.
        return flatten_min_max_bounds(value)

    if key in _PATH_KEYS and value is not None:
        return Path(value)

    return value


def _load_yaml_config(path: Path, aliases: Mapping[str, str]) -> dict[str, Any]:
    # The YAML file is expected to mirror parser keyword names, not positional argparse syntax.
    with path.open() as file:
        data = yaml.safe_load(file) or {}

    if not isinstance(data, dict):
        raise ValueError("YAML configuration must contain a mapping of option names to values, "
                         f"got {type(data).__name__}.")

    normalized = {}
    for key, value in data.items():
        if not isinstance(key, str):
            raise ValueError(f"YAML configuration keys must be strings, got {key!r}.")
        # Convert YAML keys to argparse destination names before injecting them as defaults.
        normalized_key = _normalize_config_key(key, aliases)
        normalized[normalized_key] = _normalize_config_value(normalized_key, value)

    return normalized


def _valid_parser_dests(parser: argparse.ArgumentParser) -> set[str]:
    # argparse stores values by destination name; YAML keys must map to one of these destinations.
    return {action.dest for action in parser._actions if action.dest != argparse.SUPPRESS}


def _option_strings(parser: argparse.ArgumentParser) -> set[str]:
    # Collect every registered option spelling, including aliases like --bw and --bandwidth.
    return {option for action in parser._actions for option in action.option_strings}


def _disallowed_options_with_config(parser: argparse.ArgumentParser, argv: Sequence[str]) -> list[str]:
    # When --config is used, all keyword options should come from YAML.
    # Positional input files are still allowed because they do not start with "-".
    option_strings = _option_strings(parser)
    long_options = {option for option in option_strings if option.startswith("--")}
    short_options = {option for option in option_strings if option.startswith("-") and not option.startswith("--")}
    disallowed = []
    skip_next = False
    positionals_only = False

    for token in argv:
        # Everything after "--" is positional by argparse convention, so it can be input data.
        if positionals_only:
            continue
        if skip_next:
            skip_next = False
            continue
        if token == "--":
            positionals_only = True
            continue
        if token == "--config":
            # The path following --config belongs to --config itself, not to the disallowed options.
            skip_next = True
            continue
        if token.startswith("--config=") or token in ("-h", "--help"):
            continue

        if token.startswith("--"):
            option = token.split("=", 1)[0]
            # Accept argparse abbreviations as disallowed too, e.g. --band for --bandwidth.
            if option in long_options or any(item.startswith(option) for item in long_options):
                disallowed.append(option)
        elif token.startswith("-") and token[:2] in short_options:
            disallowed.append(token)

    return list(dict.fromkeys(disallowed))


def parse_args_with_yaml_config(parser: argparse.ArgumentParser,
                                argv: Sequence[str] | None,
                                aliases: Mapping[str, str] | None = None):
    aliases = aliases or {}
    argv = sys.argv[1:] if argv is None else list(argv)

    # First parse only --config so YAML defaults can be installed before the real parse.
    config_parser = argparse.ArgumentParser(add_help=False)
    config_parser.add_argument("--config", type=Path)
    config_args, _unknown = config_parser.parse_known_args(argv)

    if config_args.config is not None:
        # YAML mode is exclusive with keyword options; only positional input files may remain.
        disallowed = _disallowed_options_with_config(parser, argv)
        if disallowed:
            raise ValueError("--config cannot be combined with command-line option(s): "
                             f"{', '.join(disallowed)}. Put those keywords in the YAML file.")

        config = _load_yaml_config(config_args.config, aliases)
        invalid = sorted(set(config) - _valid_parser_dests(parser))
        if invalid:
            raise ValueError(f"Unknown option(s) in YAML configuration: {', '.join(invalid)}.")
        # Setting defaults lets normal argparse validation and type expectations still run afterward.
        parser.set_defaults(**config)

    return parser.parse_args(argv)


def _to_yaml_value(value: Any):
    # Convert parser/numpy objects to plain Python values that yaml.safe_dump can serialize.
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return _to_yaml_value(value.tolist())
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, tuple):
        return [_to_yaml_value(item) for item in value]
    if isinstance(value, list):
        return [_to_yaml_value(item) for item in value]
    if isinstance(value, dict):
        return {key: _to_yaml_value(item) for key, item in value.items()}

    return value


def save_yaml_config(path: Path, data: Mapping[str, Any]):
    # Save the fully resolved CLI keywords so the command can be reproduced later.
    with path.open("w") as file:
        yaml.safe_dump(_to_yaml_value(dict(data)), file, sort_keys=False)
