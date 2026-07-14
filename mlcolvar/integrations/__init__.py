from . import pet_jit_fix as _pet_jit_fix  # noqa: F401

from .atomistic import (
    AtomisticConcatModel,
    AtomisticFeaturizer,
    AtomisticNodewiseModel,
    AtomisticPooledModel,
    BaseAtomisticBackbone,
)

from .backbones import (
    DeepMDBackbone,
    MACEBackbone,
    PETBackbone,
)

from .utils import align_node_attrs


__all__ = [
    "align_node_attrs",
    "BaseAtomisticBackbone",
    "AtomisticFeaturizer",
    "AtomisticPooledModel",
    "AtomisticNodewiseModel",
    "AtomisticConcatModel",
    "MACEBackbone",
    "PETBackbone",
    "DeepMDBackbone",
]


# Optional Metatomic export
try:
    from .metatomic import (
        MetatomicCVWrapper,
        create_metatomic_model,
        export_metatomic_model,
    )

    __all__.extend(
        [
            "MetatomicCVWrapper",
            "create_metatomic_model",
            "export_metatomic_model",
        ]
    )

except ImportError:
    pass