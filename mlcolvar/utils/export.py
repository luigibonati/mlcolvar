import os
import json
import uuid
import zipfile
import warnings

from typing import Dict, Tuple, Optional, Any, List, Union
from dataclasses import dataclass
from contextlib import contextmanager

import torch
import torch._inductor.package
import torch_geometric

from lightning import LightningModule
from torch.fx.experimental.proxy_tensor import make_fx

from mlcolvar.core.nn import FeedForward, BaseGNN
from mlcolvar.utils import _code

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

# Graph serialization schema used by the exported GNN models.
GRAPH_FIELDS = [
    "edge_index",
    "shifts",
    "unit_shifts",
    "positions",
    "node_attrs",
    "batch",
    "weight",
    "graph_labels",
    "cell",
    "ptr",
    "n_system",
]

OPTIONAL_GRAPH_FIELDS = [
    "system_masks",
    "subsystem_masks",
    "edge_masks_lr",
]

EXCLUDED_AGGR_MODULES = [
    "MedianAggregation",
    "MinAggregation",
    "MaxAggregation",
]

# Static fallback implementations of scatter operations used during export.
def _scatter_sum_static(
    src: torch.Tensor,
    index: torch.Tensor,
    dim: int = -1,
    out: Optional[torch.Tensor] = None,
    dim_size: Optional[int] = None,
) -> torch.Tensor:
    return torch.sum(src, dim=dim, keepdim=True)


def _scatter_mean_static(
    src: torch.Tensor,
    index: torch.Tensor,
    dim: int = -1,
    out: Optional[torch.Tensor] = None,
    dim_size: Optional[int] = None,
) -> torch.Tensor:
    return torch.mean(src, dim=dim, keepdim=True)


class ExportWrapper(torch.nn.Module):
    """
    Wrapper used during model export.

    Normal CV mode:
        returns (CV, ∇CV, 0, 0)

    Kolmogorov-bias mode:
        returns ([z, q], [∇z, ∇q], V_K, ∇V_K)

    The Kolmogorov-bias mode is enabled by passing k_bias_options to export().
    It assumes that the model has:
        - model.forward_nn(...)
        - model.sigmoid(...)
    """
    def __init__(
        self,
        model,
        calculate_gradients: bool = True,
        calculate_k_bias: bool = False,
        epsilon: float = 1e-14,
        lambd: float = -1.0,
        is_gnn: bool = False,
    ):
        super().__init__()

        self.model = model
        self.calculate_gradients = calculate_gradients
        self.calculate_k_bias = calculate_k_bias

        self.is_gnn = is_gnn

        self.epsilon = torch.tensor(
            epsilon, dtype=torch.get_default_dtype()
        )
        self.lambd = torch.tensor(
            lambd, dtype=torch.get_default_dtype()
        )

        if self.calculate_k_bias:
            if not hasattr(self.model, "forward_nn"):
                raise RuntimeError(
                    "k_bias_options was provided, so the model is treated as a "
                    "committor model, but it does not have forward_nn()."
                )

            if not hasattr(self.model, "sigmoid"):
                raise RuntimeError(
                    "k_bias_options was provided, so the model is treated as a "
                    "committor model, but it does not have sigmoid."
                )

            self.kb_sigmoid_p = torch.tensor(
                self.model.sigmoid.p,
                dtype=torch.get_default_dtype(),
            )

        if self.calculate_k_bias and not self.calculate_gradients:
            raise RuntimeError("Can not calculate k_bias without gradients")

    def forward(self, inputs, token: bool = False):
        if self.calculate_k_bias:
            return self._forward_kbias(inputs)

        return self._forward_cv(inputs)
    
    def _compute_cv_outputs(self, inputs):
        if self.is_gnn:
            data = GraphAdapter.tuple_to_dict(inputs)

            x = data["positions"].requires_grad_(True)
            data["positions"] = x

            outputs = self.model(data)

        else:
            data = None

            x = inputs[0].requires_grad_(True)
            outputs = self.model(x)

        return outputs, x, data
    
    def _forward_cv(self, inputs):
        outputs, x, data = self._compute_cv_outputs(inputs)

        zero = torch.tensor(
            0, device=outputs.device, dtype=outputs.dtype
        )

        if self.calculate_gradients:
            gradients = self._compute_cv_gradients(outputs, x, data)
        else:
            gradients = zero

        return outputs, gradients, zero, zero

    def _compute_cv_gradients(self, outputs, x, data):
        # Multi-output CV: compute full Jacobian
        if outputs.shape[1] > 1:
            if self.is_gnn:

                def wrapper(pos):
                    data["positions"] = pos
                    return self.model(data)

                gradients = torch.autograd.functional.jacobian(
                    wrapper,
                    x,
                    create_graph=False,
                    strict=False,
                    vectorize=False,
                )[0]

            else:

                def wrapper(inp):
                    return self.model(inp)

                gradients = torch.autograd.functional.jacobian(
                    wrapper,
                    x,
                    create_graph=False,
                    strict=False,
                    vectorize=False,
                )[0]

        # Single-output CV: ordinary gradient
        else:
            gradients = torch.autograd.grad(
                outputs.sum(),
                x,
                retain_graph=True,
                create_graph=False,
            )[0]
            gradients = gradients.unsqueeze(0)

        return gradients

    def _forward_kbias(self, inputs):
        outputs, gradients, k_bias_value, gradients_b = self._compute_kbias_outputs(
            inputs
        )

        return outputs, gradients, k_bias_value, gradients_b

    def _compute_kbias_outputs(self, inputs):
        if self.is_gnn:
            data = GraphAdapter.tuple_to_dict(inputs)

            x = data["positions"].requires_grad_(True)
            data["positions"] = x

            outputs_raw = self.model.forward_nn(data)

        else:
            data = None

            x = inputs[0].requires_grad_(True)
            outputs_raw = self.model.forward_nn(x)

        dtype = outputs_raw.dtype
        device = outputs_raw.device

        epsilon = self.epsilon.to(device=device, dtype=dtype)
        lambd = self.lambd.to(device=device, dtype=dtype)
        sigmoid_p = self.kb_sigmoid_p.to(device=device, dtype=dtype)

        z = outputs_raw[:, 0]
        q = self.model.sigmoid(z)

        # outputs[0]: [batch, 2] = [z, q]
        outputs = torch.stack([z, q], dim=1)

        # Need create_graph=True because grad_kbias requires second derivatives
        gradients_z = torch.autograd.grad(
            z.sum(),
            x,
            retain_graph=True,
            create_graph=True,
        )[0]

        sigmoid_prime = sigmoid_p * q * (1.0 - q)

        if self.is_gnn:
            gradients_q = gradients_z * sigmoid_prime.reshape(-1, 1)
        else:
            gradients_q = gradients_z * sigmoid_prime.reshape(-1, 1)

        # outputs[1]: [2, n_atoms, 3] for GNN, or [2, n_features] for FFNN
        gradients = torch.stack([gradients_z, gradients_q], dim=0)

        gradients_z_sum = torch.sum(gradients_z.pow(2))

        k_bias_value = lambd * (
            torch.log(gradients_z_sum + epsilon)
            - 4.0 * torch.log(1.0 + torch.exp(-sigmoid_p * z))
            - 2.0 * sigmoid_p * z
            - torch.log(epsilon)
        )

        gradients_b = torch.autograd.grad(
            k_bias_value.sum(),
            x,
            retain_graph=False,
            create_graph=False,
        )[0]

        gradients_b = gradients_b.unsqueeze(0)

        return outputs, gradients, k_bias_value, gradients_b


@dataclass
class ExportConfig:
    file_name: str = "model.pt2"
    calculate_gradients: bool = True
    k_bias_options: Optional[Dict[str, Any]] = None
    model_summary_level: int = 3
    run_check: bool = False
    is_gnn: bool = False


class GraphAdapter:
    """
    Utility class for converting between PyG graph objects,
    dictionaries and tensor tuples used by the exported model.
    """

    @staticmethod
    def data_to_tuple(
        data: Union[torch_geometric.data.Data, Dict[str, Any], List[Any]],
        device: str = "cpu",
    ) -> Tuple[torch.Tensor, ...]:

        if isinstance(data, dict) and "data_list" in data:
            data = data["data_list"]

        if isinstance(data, list):
            data = data[0]

        loader = torch_geometric.loader.DataLoader([data], batch_size=1, shuffle=False)
        batch = next(iter(loader)).to(device)
        dd = batch.to_dict()

        dd["positions"].requires_grad_(True)

        return GraphAdapter.dict_to_tuple(dd)

    @staticmethod
    def dict_to_tuple(inputs: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, ...]:

        dtype = inputs["positions"].dtype
        device = inputs["positions"].device

        tensors = [inputs[k] for k in GRAPH_FIELDS]

        for k in OPTIONAL_GRAPH_FIELDS:
            if k in inputs:
                tensors.append(inputs[k])
            else:
                tensors.append(torch.zeros((), device=device, dtype=dtype))

        return tuple(tensors)

    @staticmethod
    def tuple_to_dict(inputs: Tuple[torch.Tensor, ...]) -> Dict[str, torch.Tensor]:

        outputs: Dict[str, torch.Tensor] = {}

        for i, k in enumerate(GRAPH_FIELDS):
            outputs[k] = inputs[i]

        offset = len(GRAPH_FIELDS)

        for i, k in enumerate(OPTIONAL_GRAPH_FIELDS):
            tensor = inputs[offset + i]
            if tensor.ndim != 0:
                outputs[k] = tensor

        return outputs


class ModelExporter:
    """
    Exporter for FFNN and GNN models.
    This class manages:
      - input normalization
      - metadata generation
      - symbolic tracing + AOT compile
      - packaging
      - optional output precision checking
    """

    def __init__(
        self,
        model: LightningModule,
        example_inputs: Union[torch.Tensor, torch_geometric.data.Data, Dict[str, Any], List[Any]],
        config: ExportConfig,
    ):
        self.model = model
        self.example_inputs = example_inputs
        self.config = config

        self.calculate_k_bias = config.k_bias_options is not None

        self.k_bias_options = self._normalize_k_bias_options(
            model,
            config.k_bias_options,
        )

    # -------------------------------------------------------------------------
    # static helpers
    # -------------------------------------------------------------------------

    @staticmethod
    def _normalize_k_bias_options(
        model,
        k_bias_options: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:

        dtype = next(model.parameters()).dtype

        results = {
            "calculate_k_bias": k_bias_options is not None,
            "epsilon": 1e-14 if dtype == torch.float64 else 1e-7,
            "lambd": -1.0,
        }

        if not k_bias_options:
            return results

        for k, v in k_bias_options.items():
            if k == "epsilon":
                results["epsilon"] = float(v)
            elif k in ["lambd", "lambda"]:
                results["lambd"] = float(v)
            elif k == "calculate_k_bias":
                results["calculate_k_bias"] = bool(v)
            else:
                raise ValueError(f"Unknown k_bias_options key: {k}")

        return results

    # -------------------------------------------------------------------------
    # input preparation
    # -------------------------------------------------------------------------

    def _prepare_example_inputs(self) -> Tuple[torch.Tensor, ...]:
        if self.config.is_gnn:
            return GraphAdapter.data_to_tuple(
                self.example_inputs,
                self.model.device,
            )

        x = self.example_inputs.to(self.model.device)
        if x.dim() == 1:
            x = x.unsqueeze(0)
        return (x,)

    # -------------------------------------------------------------------------
    # model summary / metadata
    # -------------------------------------------------------------------------

    def _build_model_summary(
        self,
        model_name: str,
        module: torch.nn.Module,
        level_max: int,
        level: int,
    ) -> str:
        result = "  " * (level + 1) + "(" + model_name + "): "

        model_type = str(module.__class__.__name__)
        if model_type in ["Linear", "TICA"]:
            result = result + str(module)
        else:
            result = result + model_type

        if len(list(module.named_children())) != 0:
            if level <= level_max:
                result = result + " {\n"
                for s in module.named_children():
                    result = result + self._build_model_summary(
                        s[0], s[1], level_max, level + 1
                    )
                result = result + "  " * (level + 1) + "}\n"
            else:
                result = result + " { ... }\n"
        else:
            result = result + "\n"

        return result

    def _build_model_metadata(self) -> Dict[str, str]:
        if self.config.is_gnn:

            n_cvs = 2 if self.calculate_k_bias else int(self.model.n_cvs.item())

            metadata = {
                "n_cvs": str(n_cvs),
                "cutoff": str(self.model.cutoff.item()),
                "buffer": str(self.model.buffer.item()),
                "long_range_cutoff": str(self.model.long_range_cutoff.item()),
                "n_atom_types": str(len(self.model.atomic_numbers)),
                "float_dtype": str(self.model.dtype)[-2:],
                "calculate_gradients": str(self.config.calculate_gradients),
                "calculate_k_bias": str(self.calculate_k_bias),
                "model_type": "gnn",
            }

            for i in range(len(self.model.atomic_numbers)):
                metadata[f"atomic_number_{i:d}"] = str(
                    self.model.atomic_numbers[i].item()
                )

            metadata["model_summary"] = self._build_model_summary(
                "CV", self.model, self.config.model_summary_level, 0
            )

            metadata["n_parameters"] = str(
                sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            )

            for k, v in self.k_bias_options.items():
                metadata[k] = str(v)

            return metadata
        else:
            n_cvs = 2 if self.calculate_k_bias else int(self.model.n_cvs)

            metadata = {
                "n_cvs": str(n_cvs),
                "float_dtype": str(next(self.model.parameters()).dtype)[-2:],
                "calculate_gradients": str(self.config.calculate_gradients),
                "calculate_k_bias": str(self.calculate_k_bias),
                "model_type": "ffnn",
            }

            for k, v in self.k_bias_options.items():
                metadata[k] = str(v)

            return metadata

    # -------------------------------------------------------------------------
    # package metadata update
    # -------------------------------------------------------------------------

    @staticmethod
    def _update_package_metadata(file_name: str, data: Dict[str, str]) -> None:
        tmp = str(uuid.uuid4())

        with (
            zipfile.ZipFile(file_name, "r") as fin,
            zipfile.ZipFile(tmp, "w") as fout,
        ):
            for item in fin.infolist():
                if "metadata" in item.filename:
                    metadata = json.loads(fin.read(item.filename))
                    metadata.update(data)
                    fout.writestr(item.filename, json.dumps(metadata))
                else:
                    fout.writestr(item.filename, fin.read(item.filename))

        os.remove(file_name)
        os.rename(tmp, file_name)

    # -------------------------------------------------------------------------
    # checks / patches
    # -------------------------------------------------------------------------

    def _check_aggr_modules(self) -> None:
        model_summary = self._build_model_summary("", self.model, 100, 0)
        for name in EXCLUDED_AGGR_MODULES:
            if name in model_summary:
                message = (
                    "Aggregation modules {} can not be correctly exported on some "
                    + "machines, and your input model contains the {} module!"
                )
                raise RuntimeError(message.format(EXCLUDED_AGGR_MODULES, name))

    @staticmethod
    def _check_exported_model_outputs(
        file_name: str,
        model: torch.nn.Module,
        example_inputs: Tuple[torch.Tensor, ...],
    ) -> None:
        print("Export precision check:")

        def check_mae(x: float, dtype: str, prefix: str) -> bool:
            if dtype == "32":
                tol = float(os.environ.get("MLCOLVAR_EXPORT_FLOAT_TOL", "1E-6"))
            elif dtype == "64":
                tol = float(os.environ.get("MLCOLVAR_EXPORT_FLOAT_TOL", "1E-12"))
            else:
                raise RuntimeError("Unknown dtype " + dtype)

            if x > tol:
                raise RuntimeError(
                    "MAE ({:e}) of {:s} is larger than ".format(x, prefix)
                    + "{:e} for a float{:s} model!".format(tol, dtype)
                )
            else:
                print("  MAE of {:s}: {:e}".format(prefix, x))

        aot_model = torch._inductor.aoti_load_package(file_name)
        metadata = aot_model.get_metadata()

        float_dtype = metadata["float_dtype"]
        calculate_gradients = metadata["calculate_gradients"] in ("True", "1", True)
        calculate_k_bias = metadata.get("calculate_k_bias", "False") in ("True", "1", True)
        n_cvs = int(metadata["n_cvs"])

        model_outputs = model(example_inputs)
        aot_model_outputs = aot_model(example_inputs)

        delta = model_outputs[0] - aot_model_outputs[0]
        mae = delta.abs().max().item()
        check_mae(mae, float_dtype, "CV values")

        if calculate_gradients:
            for i in range(n_cvs):
                delta_i = model_outputs[1][i] - aot_model_outputs[1][i]
                mae = delta_i.abs().max().item()
                check_mae(mae, float_dtype, "CV gradients {:d}".format(i))

        if calculate_k_bias:
            delta_k = model_outputs[2] - aot_model_outputs[2]
            mae = delta_k.abs().max().item()
            check_mae(mae, float_dtype, "KBias")

            delta_k_grad = model_outputs[3][0] - aot_model_outputs[3][0]
            mae = delta_k_grad.abs().max().item()
            check_mae(mae, float_dtype, "KBias gradients")

    @contextmanager
    def _patched_graph_ops(self):
        scatter_sum = _code.scatter_sum
        scatter_mean = _code.scatter_mean

        _code.scatter_sum = _scatter_sum_static
        _code.scatter_mean = _scatter_mean_static

        try:
            yield
        finally:
            _code.scatter_sum = scatter_sum
            _code.scatter_mean = scatter_mean

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

    # -------------------------------------------------------------------------
    # compile / package
    # -------------------------------------------------------------------------

    def _wrap_model_for_export(self) -> ExportWrapper:
        return ExportWrapper(
            self.model,
            calculate_gradients=self.config.calculate_gradients,
            is_gnn=self.config.is_gnn,
            **self.k_bias_options,
        )

    def _compile_and_export(
        self,
        exportable_model: ExportWrapper,
        inputs: Tuple[torch.Tensor, ...],
        metadata: Dict[str, str],
    ) -> str:
        # taken from: https://depyf.readthedocs.io/en/latest/walk_through.html
        def forward_and_backward(_inputs, kwargs={}):
            return exportable_model(_inputs, False)

        wrapped = make_fx(
            forward_and_backward,
            tracing_mode="symbolic",
            _allow_non_fake_inputs=True,
        )
        graph = wrapped(inputs, {})

        aot_files = torch._inductor.aot_compile(
            graph,
            inputs,
            options={"aot_inductor.package": True},
        )

        file_name = self.config.file_name
        if not file_name.endswith(".pt2"):
            tmp = os.path.splitext(file_name)[0] + ".pt2"
            warnings.warn(f'renamed file name "{file_name}" to "{tmp}"!')
            file_name = tmp

        output_path = torch._inductor.package.package_aoti(
            file_name,
            aot_files,
        )

        self._update_package_metadata(file_name, metadata)

        if self.config.run_check:
            self._check_exported_model_outputs(
                file_name=file_name,
                model=exportable_model,
                example_inputs=inputs,
            )

        return output_path

    # -------------------------------------------------------------------------
    # public
    # -------------------------------------------------------------------------

    def export(self) -> str:
        if self.config.is_gnn:
            self._check_aggr_modules()
            torch._dynamo.allow_in_graph(torch.autograd.grad)
            torch._dynamo.allow_in_graph(torch.autograd.functional.jacobian)

        inputs = self._prepare_example_inputs()
        metadata = self._build_model_metadata()
        exportable = self._wrap_model_for_export()

        with self._exporting_flag():
            if self.config.is_gnn:
                with self._patched_graph_ops():
                    return self._compile_and_export(
                        exportable_model=exportable,
                        inputs=inputs,
                        metadata=metadata,
                    )
            return self._compile_and_export(
                exportable_model=exportable,
                inputs=inputs,
                metadata=metadata,
            )


def export(
    model,
    example_inputs,
    file_name: str = "model.pt2",
    calculate_gradients: bool = True,
    k_bias_options: Optional[Dict[str, Any]] = None,
    model_summary_level: int = 3,
    run_check: bool = False,
):
    """
    Export a CV model using symbolic tracing and Ahead-Of-Time (AOT)
    compilation.

    Models exported with this method are typically significantly faster
    than models compiled with JIT methods such as ``torch.jit.script``.

    Parameters
    ----------
    model : lightning.LightningModule
        The CV model to export.

    example_inputs : Any
        Example inputs used to trace the model. This can be a tensor
        (for FFNN models) or a ``torch_geometric.data.Data`` object
        (for GNN models).

    file_name : str, optional
        Name of the exported model file. The filename should include
        the ``.pt2`` extension.

    calculate_gradients : bool, optional
        Whether gradient calculations should be included in the exported
        model. This option should normally remain enabled.

    k_bias_options : dict[str, Any], optional
        Options for enabling the Kolmogorov bias :math:`V_K` for committor
        models. If this dictionary is provided, the Kolmogorov bias is
        automatically enabled in the exported model.

        Supported fields include:

        - ``epsilon`` : float  
            Numerical regularization parameter used in the Kolmogorov bias.

        - ``lambd`` : float  
            Scaling factor of the Kolmogorov bias.

    model_summary_level : int, optional
        Depth of the model summary stored in the exported metadata.

    run_check : bool, optional
        If ``True``, a precision check is performed by comparing the outputs
        of the original model and the exported model.

    Notes
    -----
    **1. Fixed dtype and device**

    The dtype and device of the model are fixed after export. Therefore,
    move the model to the desired device and dtype before exporting:

    ```python
    model = mlcolvar.cvs.DeepTICA(...)
    model = model.to(torch.float64).to("cuda")

    dataset = mlcolvar.utils.io.create_dataset_from_files(...) 
    mlcolvar.graph.utils.export.export(model, example_inputs=dataset[0])
    ```

    **2. GPU performance**

    Exported models usually run much faster on GPUs. It is therefore
    recommended to compile the PLUMED interface with a CUDA-enabled
    version of LibTorch.

    **3. Static gradient graph**

    In exported models, the gradient computation graph is statically
    compiled. As a result, Kolmogorov bias parameters cannot be changed
    at runtime. If these parameters need to be modified, the model must
    be re-exported.

    Example:

    ```python
    export(
        model,
        example_inputs=dataset[0],
        file_name="model_kbias_lambda_1.0.pt2",
        k_bias_options={"lambd": 1.0},
    )
    ```

    **4. CUDA toolkit requirement**

    When exporting CUDA models, the CUDA toolkit must be visible to
    PyTorch. For example:

    ```bash
    export CUDA_HOME=/usr/local/cuda-12.9
    export PATH=$PATH:/usr/local/cuda-12.9/bin
    export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda-12.9/lib64
    export C_INCLUDE_PATH=$C_INCLUDE_PATH:/usr/local/cuda-12.9/include
    ```

    **5. Portability**

    Exported models are not guaranteed to be portable across machines.
    They may only run correctly on the system where they were exported.

    **6. Unsupported PyG aggregation modules**

    Some ``torch_geometric`` aggregation modules (e.g.
    ``MedianAggregation``, ``MinAggregation``, and ``MaxAggregation``)
    may not export correctly and can lead to incorrect gradient
    computations. Avoid using these modules when exporting models.

    **7. PyTorch version**

    PyTorch v2.9 or newer is recommended for reliable export support.
    """

    # detect model type
    is_gnn = isinstance(model, BaseGNN) or (
        hasattr(model, "nn") and isinstance(model.nn, BaseGNN)
    )

    exporter = ModelExporter(
        model=model,
        example_inputs=example_inputs,
        config=ExportConfig(
            file_name=file_name,
            calculate_gradients=calculate_gradients,
            k_bias_options=k_bias_options,
            model_summary_level=model_summary_level,
            is_gnn=is_gnn,
            run_check=run_check,
        ),
    )

    return exporter.export()


def load_exported(
    file_name: str,
) -> torch._inductor.package.package.AOTICompiledModel:
    """
    Load an exported CV model.

    Parameters
    ----------
    file_name: str
        Name of the `.pt2` file.
    """

    model = torch._inductor.aoti_load_package(file_name)

    return model



def test_export_1():
    torch.manual_seed(0)
    torch.set_default_dtype(torch.float32)

    model = FeedForward(layers=[5, 16, 16, 2])
    model.n_cvs = 2
    model.device = "cpu"

    x = torch.randn(1,5)

    export(
        model,
        example_inputs=x,
        file_name="test.pt2",
        run_check=True
    )

    os.remove("test.pt2")


def test_export_2():

    torch.manual_seed(0)
    torch.set_default_dtype(torch.float32)

    from mlcolvar.cvs import Committor
    from mlcolvar.cvs.committor.utils import initialize_committor_masses

    atomic_masses = initialize_committor_masses(
        atom_types=[0,1],
        masses=[15.999, 1.008]
    )

    x = torch.randn((1,5))

    model = Committor(
        model=[5,4,2,1],
        atomic_masses=atomic_masses,
        alpha=1e-1,
        delta_f=0
    )

    k_bias_options = dict(
        epsilon=1e-6,
        lambd=1,
    )

    export(
        model,
        example_inputs=x,
        file_name="test.pt2",
        k_bias_options=k_bias_options,
        run_check=True
    )

    os.remove("test.pt2")


def test_export_3() -> None:
    torch.manual_seed(0)
    torch.set_default_dtype(torch.float32)

    model = __import__("mlcolvar").core.nn.graph.SchNetModel(
        n_out=2,
        cutoff=0.1,
        atomic_numbers=[1, 8],
        n_bases=6,
        n_layers=2,
        n_filters=16,
        n_hidden_channels=16,
    )

    model.n_cvs = model.n_out
    model.dtype = torch.float32
    model.device = "cpu"

    batch = __import__("mlcolvar").data.graph.utils.create_test_graph_input(
        output_type="batch",
        n_atoms=3,
        n_samples=6,
        n_states=1,
        add_noise=False,
    )["data_list"]

    dataset = batch.to_data_list()[0]

    export(
        model,
        example_inputs=dataset,
        file_name="model.pt2",
        run_check=True
    )

    os.remove("model.pt2")