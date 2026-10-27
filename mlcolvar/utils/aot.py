import json
import os
import tempfile
import warnings
import zipfile
from contextlib import contextmanager
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch._inductor.package
import torch_geometric
from torch.fx.experimental.proxy_tensor import make_fx

from mlcolvar.core.nn import BaseGNN
from mlcolvar.utils import _code

__all__ = ["GraphAdapter", "export", "load"]


# Maximum optimization settings for exporting models with AOTInductor.
# Activated only when MLCOLVAR_EXPORT_MAXIMUM_OPT=1.
if os.environ.get("MLCOLVAR_EXPORT_MAXIMUM_OPT") == "1":
    torch._inductor.config.freezing = True
    torch._inductor.config.max_autotune = True
    torch._inductor.config.max_autotune_gemm = True

    if hasattr(torch._inductor.config, "cuda"):
        torch._inductor.config.cuda.compile_opt_level = "-O3"

    if hasattr(torch._inductor.config.aot_inductor, "compile_wrapper_opt_level"):
        torch._inductor.config.aot_inductor.compile_wrapper_opt_level = "O3"

    os.environ["MLCOLVAR_EXPORT_FLOAT_TOL"] = "1E-4"


# Graph serialization schema used by AOT-compiled GNN models.
_AOT_FORMAT_VERSION = 2

_GRAPH_FIELDS = (
    "edge_index", "shifts", "unit_shifts", "positions", "node_attrs", "batch",
    "weight", "graph_labels", "cell", "ptr", "n_system",
)

_OPTIONAL_GRAPH_FIELDS = ("system_masks", "subsystem_masks", "edge_masks_lr")

_EXCLUDED_AGGR_MODULES = ("MedianAggregation", "MinAggregation", "MaxAggregation")


# Static fallback implementations of scatter operations used during export.
#
# These fallback functions are only used inside the export context. They are not
# general replacements for scatter_sum/scatter_mean, because they intentionally
# ignore index/out/dim_size. They should only be used when the exported graph
# operations have already been reduced to fixed-shape reductions for AOT tracing.
def _scatter_sum_static(
    src: torch.Tensor, index: torch.Tensor, dim: int = -1,
    out: Optional[torch.Tensor] = None, dim_size: Optional[int] = None,
) -> torch.Tensor:
    return torch.sum(src, dim=dim, keepdim=True)


def _scatter_mean_static(
    src: torch.Tensor, index: torch.Tensor, dim: int = -1,
    out: Optional[torch.Tensor] = None, dim_size: Optional[int] = None,
) -> torch.Tensor:
    return torch.mean(src, dim=dim, keepdim=True)


def _get_graph_model(model: torch.nn.Module) -> torch.nn.Module:
    if isinstance(model, BaseGNN):
        return model

    preprocessing = getattr(model, "preprocessing", None)
    if getattr(preprocessing, "input_kind", None) == "graph":
        return preprocessing

    nn = getattr(model, "nn", None)
    if isinstance(nn, BaseGNN):
        return nn

    raise TypeError("AOT export requires a graph-based model or preprocessing.")


class _AOTWrapper(torch.nn.Module):
    """
    Adapter that makes a GNN model compatible with the AOTInductor interface.

    The compiled model always returns four tensors for compatibility with the
    PLUMED AOT interface.

    Normal CV mode:
        returns (CV, dCV/dx, 0, 0)

    Kolmogorov-bias mode:
        returns ([z, q], [dz/dx, dq/dx], V_K, dV_K/dx)

    In normal CV mode, the third and fourth outputs are scalar zero placeholders.
    They are only meaningful when Kolmogorov-bias mode is explicitly enabled
    through ``k_bias_options``.

    The Kolmogorov-bias mode assumes that the model has:
        - model.forward_nn(...)
        - model.sigmoid(...)
    """

    def __init__(
        self, model, calculate_gradients: bool = True,
        calculate_k_bias: bool = False, epsilon: float = 1e-14,
        lambd: float = -1.0, beta: float = 1.0,
    ):
        super().__init__()
        self.model = model
        self.calculate_gradients = calculate_gradients
        self.calculate_k_bias = calculate_k_bias

        dtype = torch.get_default_dtype()
        self.register_buffer("epsilon", torch.tensor(epsilon, dtype=dtype))
        self.register_buffer("lambd", torch.tensor(lambd, dtype=dtype))
        self.register_buffer("beta", torch.tensor(beta, dtype=dtype))

        if calculate_k_bias:
            if not hasattr(model, "forward_nn"):
                raise RuntimeError(
                    "k_bias_options was provided, so the model is treated as a "
                    "committor model, but it does not have forward_nn()."
                )
            if not hasattr(model, "sigmoid"):
                raise RuntimeError(
                    "k_bias_options was provided, so the model is treated as a "
                    "committor model, but it does not have sigmoid."
                )
            self.register_buffer(
                "kb_sigmoid_p", torch.tensor(model.sigmoid.p, dtype=dtype)
            )

        if calculate_k_bias and not calculate_gradients:
            raise RuntimeError("Can not calculate k_bias without gradients")

    # The token argument is kept for compatibility with the compiled AOT interface.
    def forward(self, inputs, token: bool = False):
        return self._forward_kbias(inputs) if self.calculate_k_bias else self._forward_cv(inputs)

    def _forward_cv(self, inputs):
        data = GraphAdapter.tuple_to_dict(inputs)
        x = data["positions"].requires_grad_(True)
        data["positions"] = x

        outputs = self.model(data)
        zero = torch.tensor(0, device=outputs.device, dtype=outputs.dtype)
        gradients = (
            self._compute_cv_gradients(outputs, x, data)
            if self.calculate_gradients else zero
        )

        return outputs, gradients, zero, zero

    def _compute_cv_gradients(self, outputs, x, data):
        # Multi-output CV: compute full Jacobian.
        if outputs.shape[1] > 1:
            def wrapper(pos):
                data["positions"] = pos
                return self.model(data)

            return torch.autograd.functional.jacobian(
                wrapper, x, create_graph=False, strict=False, vectorize=False
            )[0]

        # Single-output CV: ordinary gradient.
        gradients = torch.autograd.grad(
            outputs.sum(), x, retain_graph=True, create_graph=False
        )[0]
        return gradients.unsqueeze(0)

    def _forward_kbias(self, inputs):
        data = GraphAdapter.tuple_to_dict(inputs)
        x = data["positions"].requires_grad_(True)
        data["positions"] = x

        outputs_raw = self.model.forward_nn(data)
        dtype, device = outputs_raw.dtype, outputs_raw.device

        epsilon = self.epsilon.to(device=device, dtype=dtype)
        lambd = self.lambd.to(device=device, dtype=dtype)
        beta = self.beta.to(device=device, dtype=dtype)
        sigmoid_p = self.kb_sigmoid_p.to(device=device, dtype=dtype)

        z = outputs_raw[:, 0]
        q = self.model.sigmoid(z)

        # outputs[0]: [batch, 2] = [z, q]
        outputs = torch.stack([z, q], dim=1)

        # Need create_graph=True because grad_kbias requires second derivatives.
        gradients_z = torch.autograd.grad(
            z.sum(), x, retain_graph=True, create_graph=True
        )[0]

        sigmoid_prime = sigmoid_p * q * (1.0 - q)
        gradients_q = gradients_z * sigmoid_prime.view(-1, 1)

        # outputs[1]: [2, n_atoms, 3] for GNN.
        gradients = torch.stack([gradients_z, gradients_q], dim=0)

        gradients_z_sum = torch.sum(gradients_z.pow(2))
        log_grad_sq = (
            torch.log(gradients_z_sum + epsilon)
            - 4.0 * torch.log(1.0 + torch.exp(-sigmoid_p * z))
            - 2.0 * sigmoid_p * z
        )

        k_bias_value = -(lambd / beta) * (log_grad_sq - torch.log(epsilon))
        gradients_b = torch.autograd.grad(
            k_bias_value.sum(), x, retain_graph=False, create_graph=False
        )[0]

        return outputs, gradients, k_bias_value, gradients_b.unsqueeze(0)


class GraphAdapter:
    """
    Utility class for converting between PyG graph objects, dictionaries and
    tensor tuples used by the exported GNN model.
    """

    @staticmethod
    def data_to_tuple(
        data: Union[torch_geometric.data.Data, Dict[str, Any], List[Any]],
        device: Union[str, torch.device] = "cpu",
    ) -> Tuple[torch.Tensor, ...]:
        if isinstance(data, dict) and "data_list" in data:
            data = data["data_list"]

        if isinstance(data, list):
            if len(data) != 1:
                raise ValueError(
                    "AOT export expects exactly one example graph, "
                    f"but received {len(data)} graphs."
                )
            data = data[0]

        loader = torch_geometric.loader.DataLoader([data], batch_size=1, shuffle=False)
        inputs = next(iter(loader)).to(device).to_dict()
        inputs["positions"].requires_grad_(True)

        return GraphAdapter.dict_to_tuple(inputs)

    @staticmethod
    def dict_to_tuple(inputs: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, ...]:
        dtype, device = inputs["positions"].dtype, inputs["positions"].device
        tensors = [inputs[key] for key in _GRAPH_FIELDS]
        tensors.extend(
            inputs[key] if key in inputs else torch.zeros((), device=device, dtype=dtype)
            for key in _OPTIONAL_GRAPH_FIELDS
        )
        return tuple(tensors)

    @staticmethod
    def tuple_to_dict(inputs: Tuple[torch.Tensor, ...]) -> Dict[str, torch.Tensor]:
        outputs = {key: inputs[i] for i, key in enumerate(_GRAPH_FIELDS)}
        offset = len(_GRAPH_FIELDS)

        for i, key in enumerate(_OPTIONAL_GRAPH_FIELDS):
            tensor = inputs[offset + i]
            if tensor.ndim != 0:
                outputs[key] = tensor

        return outputs


class _AOTExporter:
    """
    AOTInductor compiler for mlcolvar GNN models.

    This class manages:
      - graph input normalization
      - metadata generation
      - symbolic tracing
      - AOTInductor compilation and packaging
      - optional numerical validation
    """

    def __init__(
        self, model: torch.nn.Module,
        example_inputs: Union[torch_geometric.data.Data, Dict[str, Any], List[Any]],
        file_name: str = "model.pt2", calculate_gradients: bool = True,
        k_bias_options: Optional[Dict[str, Any]] = None,
        model_summary_level: int = 3, run_check: bool = False,
    ):
        self.model = model
        self.graph_model = _get_graph_model(model)
        self.example_inputs = example_inputs
        self.file_name = file_name
        self.calculate_gradients = calculate_gradients
        self.model_summary_level = model_summary_level
        self.run_check = run_check
        self.k_bias_options = self._normalize_k_bias_options(model, k_bias_options)
        self.calculate_k_bias = k_bias_options is not None

    @staticmethod
    def _normalize_k_bias_options(
        model, k_bias_options: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        try:
            dtype = next(model.parameters()).dtype
        except StopIteration:
            dtype = torch.get_default_dtype()

        options = {
            "epsilon": 1e-14 if dtype == torch.float64 else 1e-7,
            "lambd": 1.0,
            "beta": 1.0,
        }

        if k_bias_options:
            unknown = set(k_bias_options) - set(options)
            if unknown:
                raise ValueError(f"Unknown k_bias_options key: {sorted(unknown)[0]}")
            options.update({key: float(value) for key, value in k_bias_options.items()})

        if options["beta"] <= 0:
            raise ValueError("k_bias_options['beta'] must be positive.")

        return options

    def _build_model_summary(
        self, model_name: str, module: torch.nn.Module,
        level_max: int, level: int,
    ) -> str:
        indent = "  " * (level + 1)
        model_type = module.__class__.__name__
        result = f"{indent}({model_name}): " + (
            str(module) if model_type in ("Linear", "TICA") else model_type
        )

        children = list(module.named_children())
        if not children:
            return result + "\n"
        if level > level_max:
            return result + " { ... }\n"

        result += " {\n"
        for name, child in children:
            result += self._build_model_summary(name, child, level_max, level + 1)

        return result + indent + "}\n"

    def _build_model_metadata(self) -> Dict[str, str]:
        graph_model = self.graph_model
        n_cvs = (
            int(self.model.n_out)
            if isinstance(self.model, BaseGNN)
            else int(self.model.n_cvs)
        )

        if self.calculate_k_bias and n_cvs != 1:
            raise ValueError("Kolmogorov-bias export requires a single-CV model.")

        n_outputs = 2 if self.calculate_k_bias else n_cvs
        metadata = {
            "aot_format_version": str(_AOT_FORMAT_VERSION),
            "n_cvs": str(n_cvs),
            "n_outputs": str(n_outputs),
            "cutoff": str(graph_model.cutoff.item()),
            "buffer": str(graph_model.buffer.item()),
            "long_range_cutoff": str(graph_model.long_range_cutoff.item()),
            "n_atom_types": str(len(graph_model.atomic_numbers)),
            "float_dtype": str(self.model.dtype)[-2:],
            "calculate_gradients": str(self.calculate_gradients),
            "calculate_k_bias": str(self.calculate_k_bias),
            "model_type": "gnn",
        }

        for i, atomic_number in enumerate(graph_model.atomic_numbers):
            metadata[f"atomic_number_{i}"] = str(atomic_number.item())

        metadata["model_summary"] = self._build_model_summary(
            "CV", self.model, self.model_summary_level, 0
        )
        metadata["n_parameters"] = str(
            sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        )
        metadata.update({key: str(value) for key, value in self.k_bias_options.items()})

        return metadata

    @staticmethod
    def _update_package_metadata(file_name: str, data: Dict[str, str]) -> None:
        file_path = os.path.abspath(file_name)
        directory = os.path.dirname(file_path)
        fd, tmp_path = tempfile.mkstemp(dir=directory, suffix=".pt2")
        os.close(fd)

        try:
            with (
                zipfile.ZipFile(file_path, "r") as fin,
                zipfile.ZipFile(tmp_path, "w") as fout,
            ):
                for item in fin.infolist():
                    content = fin.read(item.filename)

                    if "metadata" in item.filename:
                        metadata = json.loads(content)
                        metadata.update(data)
                        content = json.dumps(metadata)

                    fout.writestr(item, content)

            os.replace(tmp_path, file_path)

        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

    def _check_aggr_modules(self) -> None:
        model_summary = self._build_model_summary("", self.model, 100, 0)

        for name in _EXCLUDED_AGGR_MODULES:
            if name in model_summary:
                raise RuntimeError(
                    f"Aggregation modules {_EXCLUDED_AGGR_MODULES} cannot be "
                    f"correctly exported on some machines, and your input model "
                    f"contains the {name} module!"
                )

    @staticmethod
    def _check_exported_model_outputs(
        file_name: str, model: torch.nn.Module,
        example_inputs: Tuple[torch.Tensor, ...],
    ) -> None:
        print("Export precision check:")

        def check_max_abs_error(x: float, dtype: str, prefix: str) -> None:
            if dtype == "32":
                tol = float(os.environ.get("MLCOLVAR_EXPORT_FLOAT_TOL", "1E-6"))
            elif dtype == "64":
                tol = float(os.environ.get("MLCOLVAR_EXPORT_FLOAT_TOL", "1E-12"))
            else:
                raise RuntimeError(f"Unknown dtype {dtype}")

            if x > tol:
                raise RuntimeError(
                    f"Maximum absolute error ({x:e}) of {prefix} is larger than "
                    f"{tol:e} for a float{dtype} model!"
                )
            print(f"  Maximum absolute error of {prefix}: {x:e}")

        aot_model = torch._inductor.aoti_load_package(file_name)
        metadata = aot_model.get_metadata()

        float_dtype = metadata["float_dtype"]
        calculate_gradients = metadata["calculate_gradients"] in ("True", "1", True)
        calculate_k_bias = metadata.get("calculate_k_bias", "False") in (
            "True", "1", True
        )
        n_outputs = int(metadata["n_outputs"])

        model_outputs = model(example_inputs)
        aot_model_outputs = aot_model(example_inputs)

        delta = model_outputs[0] - aot_model_outputs[0]
        check_max_abs_error(delta.abs().max().item(), float_dtype, "CV values")

        if calculate_gradients:
            for i in range(n_outputs):
                delta = model_outputs[1][i] - aot_model_outputs[1][i]
                check_max_abs_error(
                    delta.abs().max().item(), float_dtype, f"CV gradients {i}"
                )

        if calculate_k_bias:
            delta = model_outputs[2] - aot_model_outputs[2]
            check_max_abs_error(delta.abs().max().item(), float_dtype, "KBias")

            delta = model_outputs[3][0] - aot_model_outputs[3][0]
            check_max_abs_error(
                delta.abs().max().item(), float_dtype, "KBias gradients"
            )

    @contextmanager
    def _patched_graph_ops(self):
        scatter_sum, scatter_mean = _code.scatter_sum, _code.scatter_mean
        _code.scatter_sum, _code.scatter_mean = _scatter_sum_static, _scatter_mean_static

        try:
            yield
        finally:
            _code.scatter_sum, _code.scatter_mean = scatter_sum, scatter_mean

    @contextmanager
    def _exporting_flag(self):
        had_attr = hasattr(self.model, "_exporting")
        old_value = getattr(self.model, "_exporting", False)
        self.model._exporting = True

        try:
            yield
        finally:
            if had_attr:
                self.model._exporting = old_value
            else:
                delattr(self.model, "_exporting")

    def _compile_and_package(
        self, wrapped_model: _AOTWrapper,
        inputs: Tuple[torch.Tensor, ...], metadata: Dict[str, str],
    ) -> str:
        # Taken from: https://depyf.readthedocs.io/en/latest/walk_through.html
        def forward_and_backward(_inputs, _kwargs=None):
            return wrapped_model(_inputs, False)

        wrapped = make_fx(
            forward_and_backward, tracing_mode="symbolic", _allow_non_fake_inputs=True
        )
        graph = wrapped(inputs, {})

        aot_files = torch._inductor.aot_compile(
            graph, inputs, options={"aot_inductor.package": True}
        )

        file_name = self.file_name
        if not file_name.endswith(".pt2"):
            tmp = os.path.splitext(file_name)[0] + ".pt2"
            warnings.warn(f'renamed file name "{file_name}" to "{tmp}"!')
            file_name = tmp

        output_path = torch._inductor.package.package_aoti(file_name, aot_files)
        self._update_package_metadata(file_name, metadata)

        if self.run_check:
            self._check_exported_model_outputs(file_name, wrapped_model, inputs)

        return output_path

    def export(self) -> str:
        self._check_aggr_modules()
        torch._dynamo.allow_in_graph(torch.autograd.grad)
        torch._dynamo.allow_in_graph(torch.autograd.functional.jacobian)

        inputs = GraphAdapter.data_to_tuple(self.example_inputs, self.model.device)
        metadata = self._build_model_metadata()
        wrapped_model = _AOTWrapper(
            self.model,
            calculate_gradients=self.calculate_gradients,
            calculate_k_bias=self.calculate_k_bias,
            **self.k_bias_options,
        )

        with self._exporting_flag(), self._patched_graph_ops():
            return self._compile_and_package(wrapped_model, inputs, metadata)


def export(
    model,
    example_inputs,
    file_name: str = "model.pt2",
    calculate_gradients: bool = True,
    k_bias_options: Optional[Dict[str, Any]] = None,
    model_summary_level: int = 3,
    run_check: bool = False,
) -> str:
    """
    Ahead-of-Time compile a GNN CV model with AOTInductor.

    Parameters
    ----------
    model : torch.nn.Module
        Graph-based model to compile. Graph inputs may be handled directly by
        a ``BaseGNN`` model, by a ``BaseGNN`` stored in ``model.nn``, or by a
        graph representation used as preprocessing.
    example_inputs : torch_geometric.data.Data or dict or list
        Example graph input used to trace and compile the model.
    file_name : str, optional
        Name of the compiled model package. The filename should use the
        ``.pt2`` extension. By default ``"model.pt2"``.
    calculate_gradients : bool, optional
        Whether to include gradients of the CVs with respect to atomic
        positions in the compiled model. By default ``True``.
    k_bias_options : dict[str, Any], optional
        Options for enabling the Kolmogorov bias for a committor model.
        Providing this dictionary automatically enables Kolmogorov-bias
        evaluation.

        Supported options are:

        ``epsilon``
            Numerical regularization parameter used in the Kolmogorov bias.

        ``lambd``
            Scaling factor of the Kolmogorov bias.

        ``beta``
            Inverse-temperature-like scaling parameter.

        When Kolmogorov-bias mode is enabled, the first returned tensor
        contains two components, ``[z, q]``, where ``z`` is the raw
        committor coordinate and ``q`` is the sigmoid-transformed committor.
        The third and fourth returned tensors contain the Kolmogorov bias
        and its gradient, respectively.

        By default ``None``.
    model_summary_level : int, optional
        Maximum depth of the model summary stored in the exported metadata.
        By default 3.
    run_check : bool, optional
        Whether to numerically compare eager and AOT-compiled outputs after
        compilation. By default ``False``.

    Returns
    -------
    str
        Path to the generated ``.pt2`` AOTInductor package.

    Notes
    -----
    The dtype and device of the model are fixed at compile time. Move the
    model to the desired device and dtype before calling this function.

    The compiled model always returns four tensors for compatibility with
    the PLUMED AOT interface.

    In normal CV mode, the tensors correspond to:

    1. CV values.
    2. CV gradients.
    3. A scalar zero placeholder.
    4. A scalar zero placeholder.

    In Kolmogorov-bias mode, they correspond to:

    1. The ``[z, q]`` outputs.
    2. Their gradients.
    3. The Kolmogorov bias.
    4. The gradient of the Kolmogorov bias.

    Examples
    --------
    Export a GNN-based CV model:

    .. code-block:: python

        from mlcolvar.utils.aot import export

        export(
            model=model,
            example_inputs=dataset[0],
            file_name="model.pt2",
            run_check=True,
        )

    Export a committor model with the Kolmogorov bias:

    .. code-block:: python

        export(
            model=model,
            example_inputs=dataset[0],
            file_name="model.pt2",
            calculate_gradients=True,
            k_bias_options={
                "beta": 0.5,
                "lambd": 1.0,
            },
        )
    """
    return _AOTExporter(
        model=model,
        example_inputs=example_inputs,
        file_name=file_name,
        calculate_gradients=calculate_gradients,
        k_bias_options=k_bias_options,
        model_summary_level=model_summary_level,
        run_check=run_check,
    ).export()


def load(
    file_name: str,
) -> torch._inductor.package.package.AOTICompiledModel:
    """
    Load an AOT-compiled GNN CV model.

    Parameters
    ----------
    file_name : str
        Path to the ``.pt2`` AOTInductor package.

    Returns
    -------
    torch._inductor.package.package.AOTICompiledModel
        Loaded AOTInductor model.

    Examples
    --------
    .. code-block:: python

        from mlcolvar.utils.aot import load

        model = load("model.pt2")
    """
    return torch._inductor.aoti_load_package(file_name)