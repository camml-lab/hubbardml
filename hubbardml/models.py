import collections
import math
import uuid
from typing import Dict, Tuple, Union

from e3nn import o3
import e3psi
import numpy as np
import pandas as pd
import torch

from . import graphs

__all__ = "UModel", "VModel"


class UModel(e3psi.models.OnsiteModel):
    """Hubbard +U model

    Consist of a single node that carries information about the atomic specie and occupation matrix.
    """

    TYPE_ID = uuid.UUID("947373b7-6cfc-42f6-bbcb-279f88a02db2")

    def __init__(
        self,
        graph: graphs.UGraph,
        feature_irreps="2x0e + 2x1e + 2x2e + 2x3e + 2x4e",
        hidden_layers=2,
        **kwargs,
    ) -> None:
        if feature_irreps is None:
            feature_irreps = self._compress_non_scalars(
                o3.ReducedTensorProducts("ij=ji", i=e3psi.irreps(graph.site)).irreps_out, factor=1.0
            )

        super().__init__(
            graph,
            feature_irreps=feature_irreps,
            irreps_out="0e",
            hidden_layers=hidden_layers,
            **kwargs,
        )

        self._species = graph.species

    @property
    def species(self) -> Tuple[str]:
        """Get the supported species"""
        return self._species

    def _compress_non_scalars(self, irreps: o3.Irreps, factor=3.0) -> o3.Irreps:
        return o3.Irreps(
            o3._irreps._MulIr(int(math.ceil(float(mul_ir.mul) / factor)), mul_ir.ir)
            if mul_ir.ir.l != 0
            else mul_ir
            for mul_ir in irreps
        )


class VModel(e3psi.IntersiteModel):
    """Hubbard +V model"""

    TYPE_ID = uuid.UUID("4a647f9f-0f90-474a-928e-19691b576597")
    # Columns to store the current p-block and d-block elements
    P_ELEMENT = "p_element"
    D_ELEMENT = "d_element"

    def __init__(
        self,
        graph: graphs.VGraph,
        feature_irreps="4x0e + 4x1e + 4x2e + 4x3e",
        **kwargs,
    ):
        super().__init__(
            graph,
            feature_irreps=feature_irreps,
            irreps_out="0e",
            **kwargs,
        )
        self._species = tuple(graph.species)

    @property
    def species(self) -> Tuple:
        """Get the supported species"""
        return self._species


class Rescaler(e3psi.models.Module):
    def __init__(self, shift=0.0, scale=1.0):
        super().__init__()
        self.shift = shift
        self.scale = scale

    def __repr__(self) -> str:
        return str(self)

    def __str__(self) -> str:
        return f"Rescaler(shift={self.shift},scale={self.scale})"

    def forward(self, data):
        # y = mx + c
        return self.scale * data + self.shift

    def inverse(self, data):
        # x = (y - c) / m
        return (data - self.shift) / self.scale

    @classmethod
    def from_data(cls, dataset: Union[np.ndarray, torch.Tensor], method="mean") -> "Rescaler":
        if method == "mean":
            shift = dataset.mean()
            scale = dataset.std()
            return Rescaler(shift, scale)
        elif method == "minmax":
            dmin = dataset.min()
            shift = dmin
            scale = dataset.max() - dmin
            return Rescaler(shift, scale)
        else:
            raise ValueError(f"Unknown method: {method}")


def create_model_inputs(
    graph: graphs.ModelGraph, frame: pd.DataFrame, dtype=None, device=None
) -> Dict[str, torch.Tensor]:
    """Given a graph this method will create a dictionary with stacked tensors for
    each of the inputs in the given dataframe"""
    inputs = collections.defaultdict(list)

    for _, row in frame.iterrows():
        inp = graph.create_inputs(row, dtype=dtype, device=device)
        # Append to all inputs
        for key, val in inp.items():
            inputs[key].append(val)

    # Stack everything into torch tensors
    for key, val in inputs.items():
        inputs[key] = torch.vstack(val)

    return inputs


MODELS = {
    graphs.U: UModel,
    graphs.V: VModel,
}

HISTORIAN_TYPES = VModel, UModel
