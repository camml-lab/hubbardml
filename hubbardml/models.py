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

__all__ = "UModel", "VModel", "UVModel"


class UModel(e3psi.models.OnsiteModel):
    """Hubbard +U model

    Consist of a single node that carries information about the atomic specie and occupation matrix.
    """

    TYPE_ID = uuid.UUID("947373b7-6cfc-42f6-bbcb-279f88a02db2")

    def __init__(
        self,
        graph: graphs.UGraph,
        feature_irreps="23x0e + 4x2e + 1x3e + 4x4e + 1x5e + 2x6e + 1x8e",
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
        node_features="10x0e+4x1e+6x2e",
        **kwargs,
    ):
        super().__init__(
            graph,
            node_features=node_features,
            irreps_out="0e",
            **kwargs,
        )
        self._species = tuple(graph.species)

    @property
    def species(self) -> Tuple:
        """Get the supported species"""
        return self._species


class UVModel(e3psi.IntersiteModel):
    """Hubbard +U+V model"""

    # Columns to store the current p-block and d-block elements
    P_ELEMENT = "p_element"
    D_ELEMENT = "d_element"

    def __init__(
        self,
        species,
        node_features="106x0e+4x1e+6x2e+8x3e+4x4e+2x5e+2x6e",
    ):
        super().__init__(
            graphs.UVGraph(species),
            node_features=node_features,
            irreps_out="0e",
        )
        self._species = tuple(species)

    @property
    def species(self) -> Tuple:
        """Get the supported species"""
        return self._species


class Rescaler(e3psi.models.Module):
    def __init__(self, shift=0.0, scale=1.0):
        super().__init__()
        self.shift = shift
        self.scale = scale

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


def make_predictions(model: e3psi.Model, frame: pd.DataFrame, dtype=None, device=None):
    inputs = create_model_inputs(model.graph, frame, dtype=dtype, device=device)

    return model(inputs).detach().cpu().numpy().reshape(-1)


MODELS = {
    graphs.U: UModel,
    graphs.V: VModel,
    graphs.UV: UVModel,
}

HISTORIAN_TYPES = VModel, UModel
