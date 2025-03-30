import os
from abc import abstractmethod
from collections import OrderedDict
from typing import Dict, List, Optional, Union

import torch
import yaml
from e3nn.o3 import Irrep, Irreps
from numpy import sum

from trajcast.data._keys import (
    ATOM_TYPE_EMBEDDING_KEY,
    DISPLACEMENTS_KEY,
    EDGE_LENGTHS_EMBEDDING_KEY,
    EDGE_VECTORS_KEY,
    NODE_FEATURES_KEY,
    SPHERICAL_HARMONIC_KEY,
    UPDATE_VELOCITIES_KEY,
    VELOCITIES_KEY,
)
from trajcast.data._types import FIELD_IRREPS
from trajcast.data.atomic_graph import AtomicGraph
from trajcast.nn._encoding import (
    EdgeLengthEncoding,
    ElementBasedNormEncoding,
    OneHotAtomTypeEncoding,
    SphericalHarmonicProjection,
    TensorNormEncoding,
    TimestepEncoding,
)
from trajcast.nn._graph_module_irreps import GraphModuleIrreps
from trajcast.nn._message_passing import (
    ConditionedMessagePassingLayer,
    MessagePassingLayer,
    ResidualConditionedMessagePassingLayer,
)
from trajcast.nn._normalization import NormalizationLayer
from trajcast.nn._tensor_self_interactions import LinearTensorMixer
from trajcast.nn._wrapper_ops import CuEquivarianceConfig
from trajcast.nn.modules import (
    ConservationLayer,
    ForecastHorizonConditioning,
)
from trajcast.utils.misc import convert_irreps_to_string


class AbstractModel(GraphModuleIrreps, torch.nn.Module):
    def __init__(
        self,
        config: Dict,
        predicted_fields: Optional[List] = [DISPLACEMENTS_KEY, UPDATE_VELOCITIES_KEY],
    ) -> None:
        super().__init__()
        self.config = config
        self._o3_backend = config.get("o3_backend", "e3nn")
        self._predicted_fields = predicted_fields
        target_irreps = sum([FIELD_IRREPS[i] for i in predicted_fields])
        if isinstance(target_irreps, Irrep):
            self._target_start_index = {predicted_fields[0]: 0}
        else:
            self._target_start_index = {
                predicted_fields[count]: target_irreps.slices()[count].start
                for count, field in enumerate(target_irreps)
            }
            self._target_irreps = target_irreps.simplify()

    @property
    def o3_backend(self):
        return self._o3_backend

    @o3_backend.setter
    def o3_backend(self, value: str):
        self._o3_backend = value
        self.config["o3_backend"] = self._o3_backend
        self._build_model()

    @abstractmethod
    def _build_model(self) -> torch.nn.Sequential:
        raise NotImplementedError

    @abstractmethod
    def forward(self, data: AtomicGraph) -> AtomicGraph:
        raise NotImplementedError

    @classmethod
    def build_from_yaml(cls, filename: str, *args, **kwargs):
        if not os.path.exists(filename):
            raise FileNotFoundError(
                f"Could not find the file under the path {filename}"
            )

        with open(filename, "r") as file:
            dictionary = yaml.load(file, Loader=yaml.FullLoader)
        try:
            return cls(config=dictionary["model"], *args, **kwargs)
        except KeyError:
            return cls(config=dictionary, *args, **kwargs)

    def dump_config_to_yaml(
        self, filename: Optional[str] = "config.yaml", prefix: Optional[str] = "model"
    ):
        new_dict = {prefix: self.config}
        convert_irreps_to_string(new_dict)

        with open(filename, "w") as file:
            yaml.dump(new_dict, file, sort_keys=False)


class EfficientTrajCastModel(AbstractModel):
    def __init__(
        self,
        rms_targets: Optional[Union[torch.Tensor, float, List]] = 1.0,
        mean_targets: Optional[Union[torch.Tensor, float, List]] = 0.0,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        if isinstance(rms_targets, float):
            rms_targets = torch.tensor(rms_targets).repeat(len(self._predicted_fields))
        else:
            if len(rms_targets) != len(self._predicted_fields):
                raise TypeError(
                    "Root mean squares of targets need to be same length as predicted field."
                )

            if isinstance(rms_targets, List):
                rms_targets = torch.tensor(rms_targets)

        if isinstance(mean_targets, float):
            mean_targets = torch.tensor(mean_targets).repeat(
                len(self._predicted_fields)
            )
        else:
            if len(mean_targets) != len(self._predicted_fields):
                raise TypeError(
                    "Means of targets need to be same length as predicted field."
                )

            if isinstance(mean_targets, List):
                mean_targets = torch.tensor(mean_targets)

        self.register_buffer("rms_targets", rms_targets)
        self.register_buffer("mean_targets", mean_targets)

        self.register_buffer(
            "lmax", torch.tensor(self.config.get("max_rotation_order"))
        )

        self.register_buffer(
            "edge_cutoff",
            torch.tensor(self.config.get("edge_cutoff")),
        )

        self.register_buffer(
            "num_hidden_channels", torch.tensor(self.config.get("num_hidden_channels"))
        )

        self.register_buffer(
            "avg_num_neighbors",
            torch.tensor(self.config.get("avg_num_neighbors", 10.0)),
        )

        self.register_buffer(
            "num_mp_layers", torch.tensor(self.config.get("num_mp_layers"))
        )

        self._build_model()

    def forward(self, data: AtomicGraph) -> AtomicGraph:
        data.compute_edge_vectors()
        # perform encoding
        data = self._encoding(data)

        # now we create the initial features by concatenating the following:
        # atom_type_embedding (Irreps: num_chem_elementsx0e)
        # velocities_norm_embedding (Irreps: num_vel_rbfx0e)
        # sh_embedding_velocities (Irreps: SH up to lmax but without 0e)
        data[NODE_FEATURES_KEY] = torch.cat(
            (
                data[ATOM_TYPE_EMBEDDING_KEY],
                data[f"{VELOCITIES_KEY}_norm_embedding"],
                data[f"{SPHERICAL_HARMONIC_KEY}_{VELOCITIES_KEY}"][:, 1:],
            ),
            dim=1,
        )
        # now we can do the rest of the forward, i.e. interaction + readout
        data = self._int_out(data)

        if self.o3_backend == "cueq":
            data.target = self._transpose_output(data.target)

        data = self._conservation(data)
        return data

    def _build_model(self) -> torch.nn.Sequential:
        cueq_config = None

        if self.o3_backend == "cueq":
            cueq_config = CuEquivarianceConfig(
                enabled=True,
                layout="ir_mul",
                group="O3_e3nn",
                optimize_all=True,
            )
            from cuequivariance import Irreps as cue_Irreps
            from cuequivariance import ir_mul, mul_ir
            from cuequivariance_torch import TransposeIrrepsLayout

            self._transpose_output = TransposeIrrepsLayout(
                cue_Irreps(cueq_config.group, "2x1o"),
                source=ir_mul,
                target=mul_ir,
            )

        # construct normalisation dicts
        rms_dict = {
            self._predicted_fields[i]: self.rms_targets[i]
            for i in range(len(self._predicted_fields))
        }
        mean_dict = {
            self._predicted_fields[i]: self.mean_targets[i]
            for i in range(len(self._predicted_fields))
        }

        # normalisation layer
        norm = NormalizationLayer(
            input_fields=self._predicted_fields,
            output_fields=self._predicted_fields,
            means=mean_dict,
            stds=rms_dict,
        )

        # Embedding
        atom_type_embedding = OneHotAtomTypeEncoding(
            number_of_species=self.config.get("num_chem_elements"),
            output_field=ATOM_TYPE_EMBEDDING_KEY,
        )
        edge_length_embedding = EdgeLengthEncoding(
            radial_basis="BesselBasisTrainable",
            cutoff_function="PolynomialCutoff",
            basis_kwargs={
                "rmax": self.edge_cutoff,
                "basis_size": self.config.get("num_edge_rbf"),
            },
            cutoff_kwargs={
                "rmax": self.edge_cutoff,
                "p": self.config.get("num_edge_poly_cutoff"),
            },
            output_field=EDGE_LENGTHS_EMBEDDING_KEY,
            irreps_in=atom_type_embedding.irreps_out,
        )

        velocity_norm_embedding = ElementBasedNormEncoding(
            input_field=VELOCITIES_KEY,
            atom_type_embedding_field=ATOM_TYPE_EMBEDDING_KEY,
            basis="FixedBasis",
            cutoff_kwargs={},
            basis_kwargs={
                "rmax": self.config.get("vel_max"),
                "rmin": 0,
                "basis_size": self.config.get("num_vel_rbf"),
                "basis_function": "gaussian",
            },
            irreps_in=edge_length_embedding.irreps_out,
            output_field=f"{VELOCITIES_KEY}_norm_embedding",
            cueq_config=cueq_config,
        )
        edge_sh = SphericalHarmonicProjection(
            max_rotation_order=self.lmax,
            project_on_unit_sphere=True,
            normalization_projection="component",
            input_field=EDGE_VECTORS_KEY,
            output_field=f"{SPHERICAL_HARMONIC_KEY}_{EDGE_VECTORS_KEY}",
            irreps_in=velocity_norm_embedding.irreps_out,
        )
        vel_sh = SphericalHarmonicProjection(
            max_rotation_order=self.lmax,
            project_on_unit_sphere=True,
            normalization_projection="component",
            input_field=VELOCITIES_KEY,
            output_field=f"{SPHERICAL_HARMONIC_KEY}_{VELOCITIES_KEY}",
            irreps_in=edge_sh.irreps_out,
        )

        encoding_layers = {
            "Normalization": norm,
            "AtomTypeEncoding": atom_type_embedding,
            "EdgeLengthEncoding": edge_length_embedding,
            "VelocityNormEncoding": velocity_norm_embedding,
            "SHEdgeVectors": edge_sh,
            "SHVelVectors": vel_sh,
        }

        self._encoding = torch.nn.Sequential(OrderedDict(encoding_layers))

        # produce initial features
        # get irreps after concatenation:
        vel_sh.irreps_out[NODE_FEATURES_KEY] = (
            vel_sh.irreps_out[ATOM_TYPE_EMBEDDING_KEY]
            + vel_sh.irreps_out[f"{VELOCITIES_KEY}_norm_embedding"]
            + vel_sh.irreps_out[f"{SPHERICAL_HARMONIC_KEY}_{VELOCITIES_KEY}"][1:]
        ).simplify()

        # irreps after linear
        initial_feature_irreps = (
            (Irreps.spherical_harmonics(self.lmax) * self.num_hidden_channels)
            .sort()[0]
            .simplify()
        )

        linear_init_features = LinearTensorMixer(
            input_field=NODE_FEATURES_KEY,
            output_field=NODE_FEATURES_KEY,
            irreps_in=vel_sh.irreps_out,
            irreps_out=initial_feature_irreps,
            cueq_config=cueq_config,
        )

        # define feature dimension
        irreps_node_features = Irreps(
            ("+").join(
                f"{self.num_hidden_channels}x{i}o+{self.num_hidden_channels}x{i}e"
                for i in range(self.lmax + 1)
            )
        )

        # save to module list and subsequently add additional layers
        mp_layers = torch.nn.ModuleList([])

        previous_irreps_out = linear_init_features.irreps_out
        for lay_index in range(self.num_mp_layers):
            mp_layer = ResidualConditionedMessagePassingLayer(
                max_rotation_order=self.lmax,
                edge_attributes_field=f"{SPHERICAL_HARMONIC_KEY}_{EDGE_VECTORS_KEY}",
                output_field=NODE_FEATURES_KEY,
                avg_num_neighbors=self.avg_num_neighbors,
                irreps_in=previous_irreps_out,
                irreps_out=irreps_node_features,
                edge_mlp_kwargs=self.config.get("edge_mlp_kwargs"),
                vel_emb_field=f"{SPHERICAL_HARMONIC_KEY}_{VELOCITIES_KEY}",
                vel_len_emb_field=f"{VELOCITIES_KEY}_norm_embedding",
                vel_mlp_kwargs=self.config.get("vel_mlp_kwargs"),
                nl_gate_kwargs=self.config.get(
                    "nl_gate_kwargs",
                    {
                        "irreps_gates": f"{2*self.lmax*self.num_hidden_channels}x0e",
                        "activation_scalars": {"o": "tanh", "e": "silu"},
                        "activation_gates": {"e": "silu"},
                    },
                ),
                species_emb_field=ATOM_TYPE_EMBEDDING_KEY,
                cueq_config=cueq_config,
            )

            previous_irreps_out = mp_layer.irreps_out

            mp_layers.append(mp_layer)

        # readout
        # compress to a quarter of the size
        num_hidden_compress = self.num_hidden_channels // 4
        linear_compress = LinearTensorMixer(
            input_field=NODE_FEATURES_KEY,
            output_field="target",
            irreps_in=previous_irreps_out,
            irreps_out=f"{num_hidden_compress}x1o",
            cueq_config=cueq_config,
        )

        linear_target = LinearTensorMixer(
            input_field="target",
            output_field="target",
            irreps_in=linear_compress.irreps_out,
            irreps_out=self._target_irreps,
            cueq_config=cueq_config,
        )

        # physical constrains
        conservation = ConservationLayer(
            input_field="target",
            velocity_field=VELOCITIES_KEY,
            conserve_angular=self.config.get("conserve_ang_mom", False),
            disp_norm_const=rms_dict[DISPLACEMENTS_KEY],
            vel_norm_const=rms_dict[UPDATE_VELOCITIES_KEY],
            units=self.config.get("units", "real"),
            irreps_in=linear_target.irreps_out,
            index_disp_target=self._target_start_index.get(DISPLACEMENTS_KEY, 0),
            index_vel_target=self._target_start_index.get(UPDATE_VELOCITIES_KEY, 3),
            net_lin_mom=torch.tensor(self.config.get("net_lin_mom", [])),
            net_ang_mom=torch.tensor(self.config.get("net_ang_mom", [])),
        )

        int_out = {
            "LinearInit": linear_init_features,
        }

        for idx, layer in enumerate(mp_layers):
            int_out[f"mp_layer_{idx}"] = layer

        int_out["LinearCompression"] = linear_compress

        int_out["LinearReadOut"] = linear_target

        self._conservation = conservation

        self._int_out = torch.nn.Sequential(OrderedDict(int_out))


class TrajCastModel(AbstractModel):
    def __init__(
        self,
        rms_targets: Optional[Union[torch.Tensor, float, List]] = 1.0,
        mean_targets: Optional[Union[torch.Tensor, float, List]] = 0.0,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        if isinstance(rms_targets, float):
            rms_targets = torch.tensor(rms_targets).repeat(len(self._predicted_fields))
        else:
            if len(rms_targets) != len(self._predicted_fields):
                raise TypeError(
                    "Root mean squares of targets need to be same length as predicted field."
                )

            if isinstance(rms_targets, List):
                rms_targets = torch.tensor(rms_targets)

        if isinstance(mean_targets, float):
            mean_targets = torch.tensor(mean_targets).repeat(
                len(self._predicted_fields)
            )
        else:
            if len(mean_targets) != len(self._predicted_fields):
                raise TypeError(
                    "Means of targets need to be same length as predicted field."
                )

            if isinstance(mean_targets, List):
                mean_targets = torch.tensor(mean_targets)

        self.register_buffer("rms_targets", rms_targets)
        self.register_buffer("mean_targets", mean_targets)

        self.register_buffer(
            "lmax", torch.tensor(self.config.get("max_rotation_order"))
        )

        self.register_buffer(
            "edge_cutoff",
            torch.tensor(self.config.get("edge_cutoff")),
        )

        self.register_buffer(
            "num_hidden_channels", torch.tensor(self.config.get("num_hidden_channels"))
        )

        self.register_buffer(
            "avg_num_neighbors",
            torch.tensor(self.config.get("avg_num_neighbors", 10.0)),
        )

        self.register_buffer(
            "num_mp_layers", torch.tensor(self.config.get("num_mp_layers"))
        )

        self.layers = self._build_model()

    def forward(self, data: AtomicGraph) -> AtomicGraph:
        data.compute_edge_vectors()
        data = self.layers(data)

        return data

    def _build_model(self) -> torch.nn.Sequential:
        layers = {}

        cueq_config = None

        if self.o3_backend == "cueq":
            cueq_config = CuEquivarianceConfig(
                enabled=True,
                layout="ir_mul",
                group="O3_e3nn",
                optimize_all=True,
            )
            from cuequivariance import Irreps as cue_Irreps
            from cuequivariance import ir_mul, mul_ir
            from cuequivariance_torch import TransposeIrrepsLayout

            self._transpose_output = TransposeIrrepsLayout(
                cue_Irreps(cueq_config.group, "2x1o"),
                source=ir_mul,
                target=mul_ir,
            )

        # construct normalisation dicts
        rms_dict = {
            self._predicted_fields[i]: self.rms_targets[i]
            for i in range(len(self._predicted_fields))
        }
        mean_dict = {
            self._predicted_fields[i]: self.mean_targets[i]
            for i in range(len(self._predicted_fields))
        }

        # normalisation layer
        norm = NormalizationLayer(
            input_fields=self._predicted_fields,
            output_fields=self._predicted_fields,
            means=mean_dict,
            stds=rms_dict,
        )

        # Embedding
        atom_type_embedding = OneHotAtomTypeEncoding(
            number_of_species=self.config.get("num_chem_elements"),
            output_field=ATOM_TYPE_EMBEDDING_KEY,
        )
        edge_length_embedding = EdgeLengthEncoding(
            radial_basis="BesselBasisTrainable",
            cutoff_function="PolynomialCutoff",
            basis_kwargs={
                "rmax": self.edge_cutoff,
                "basis_size": self.config.get("num_edge_rbf"),
            },
            cutoff_kwargs={
                "rmax": self.edge_cutoff,
                "p": self.config.get("num_edge_poly_cutoff"),
            },
            output_field=EDGE_LENGTHS_EMBEDDING_KEY,
            irreps_in=atom_type_embedding.irreps_out,
        )
        velocity_norm_embedding = ElementBasedNormEncoding(
            input_field=VELOCITIES_KEY,
            atom_type_embedding_field=ATOM_TYPE_EMBEDDING_KEY,
            basis="FixedBasis",
            cutoff_kwargs={},
            basis_kwargs={
                "rmax": self.config.get("vel_max"),
                "rmin": 0,
                "basis_size": self.config.get("num_vel_rbf"),
                "basis_function": "gaussian",
            },
            irreps_in=edge_length_embedding.irreps_out,
            output_field=f"{VELOCITIES_KEY}_norm_embedding",
            cueq_config=cueq_config,
        )
        edge_sh = SphericalHarmonicProjection(
            max_rotation_order=self.lmax,
            project_on_unit_sphere=True,
            normalization_projection="component",
            input_field=EDGE_VECTORS_KEY,
            output_field=f"{SPHERICAL_HARMONIC_KEY}_{EDGE_VECTORS_KEY}",
            irreps_in=velocity_norm_embedding.irreps_out,
        )
        vel_sh = SphericalHarmonicProjection(
            max_rotation_order=self.lmax,
            project_on_unit_sphere=True,
            normalization_projection="component",
            input_field=VELOCITIES_KEY,
            output_field=f"{SPHERICAL_HARMONIC_KEY}_{VELOCITIES_KEY}",
            irreps_in=edge_sh.irreps_out,
        )
        # produce initial features
        linear_init_features = LinearTensorMixer(
            input_field=ATOM_TYPE_EMBEDDING_KEY,
            output_field=NODE_FEATURES_KEY,
            irreps_in=vel_sh.irreps_out,
            irreps_out=f"{self.num_hidden_channels}x0e",
            cueq_config=cueq_config,
        )

        # define feature dimension
        irreps_node_features = Irreps(
            ("+").join(
                f"{self.num_hidden_channels}x{i}o+{self.num_hidden_channels}x{i}e"
                for i in range(self.lmax + 1)
            )
        )

        # now do the message passing
        first_mp_layer = ConditionedMessagePassingLayer(
            max_rotation_order=self.lmax,
            edge_attributes_field=f"{SPHERICAL_HARMONIC_KEY}_{EDGE_VECTORS_KEY}",
            output_field=NODE_FEATURES_KEY,
            avg_num_neighbors=self.avg_num_neighbors,
            irreps_in=linear_init_features.irreps_out,
            irreps_out=irreps_node_features,
            edge_mlp_kwargs=self.config.get("edge_mlp_kwargs"),
            vel_emb_field=f"{SPHERICAL_HARMONIC_KEY}_{VELOCITIES_KEY}",
            vel_len_emb_field=f"{VELOCITIES_KEY}_norm_embedding",
            vel_mlp_kwargs=self.config.get("vel_mlp_kwargs"),
            nl_gate_kwargs=self.config.get(
                "nl_gate_kwargs",
                {
                    "irreps_gates": f"{2*self.lmax*self.num_hidden_channels}x0e",
                    "activation_scalars": {"o": "tanh", "e": "silu"},
                    "activation_gates": {"e": "silu"},
                },
            ),
            cueq_config=cueq_config,
        )

        # save to module list and subsequently add additional layers
        mp_layers = torch.nn.ModuleList([first_mp_layer])

        previous_irreps_out = first_mp_layer.irreps_out
        for lay_index in range(self.num_mp_layers - 1):
            mp_layer = ResidualConditionedMessagePassingLayer(
                max_rotation_order=self.lmax,
                edge_attributes_field=f"{SPHERICAL_HARMONIC_KEY}_{EDGE_VECTORS_KEY}",
                output_field=NODE_FEATURES_KEY,
                avg_num_neighbors=self.avg_num_neighbors,
                irreps_in=previous_irreps_out,
                irreps_out=irreps_node_features,
                edge_mlp_kwargs=self.config.get("edge_mlp_kwargs"),
                vel_emb_field=f"{SPHERICAL_HARMONIC_KEY}_{VELOCITIES_KEY}",
                vel_len_emb_field=f"{VELOCITIES_KEY}_norm_embedding",
                vel_mlp_kwargs=self.config.get("vel_mlp_kwargs"),
                nl_gate_kwargs=self.config.get(
                    "nl_gate_kwargs",
                    {
                        "irreps_gates": f"{2*self.lmax*self.num_hidden_channels}x0e",
                        "activation_scalars": {"o": "tanh", "e": "silu"},
                        "activation_gates": {"e": "silu"},
                    },
                ),
                species_emb_field=ATOM_TYPE_EMBEDDING_KEY,
                cueq_config=cueq_config,
            )

            previous_irreps_out = mp_layer.irreps_out

            mp_layers.append(mp_layer)

        # readout
        # compress to a quarter of the size
        num_hidden_compress = self.num_hidden_channels // 4
        linear_compress = LinearTensorMixer(
            input_field=NODE_FEATURES_KEY,
            output_field="target",
            irreps_in=previous_irreps_out,
            irreps_out=f"{num_hidden_compress}x0e+{num_hidden_compress}x1o",
            cueq_config=cueq_config,
        )

        linear_target = LinearTensorMixer(
            input_field="target",
            output_field="target",
            irreps_in=linear_compress.irreps_out,
            irreps_out=self._target_irreps,
            cueq_config=cueq_config,
        )

        # physical constrains
        conservation = ConservationLayer(
            input_field="target",
            velocity_field=VELOCITIES_KEY,
            conserve_angular=self.config.get("conserve_ang_mom", False),
            disp_norm_const=rms_dict[DISPLACEMENTS_KEY],
            vel_norm_const=rms_dict[UPDATE_VELOCITIES_KEY],
            units=self.config.get("units", "real"),
            irreps_in=linear_target.irreps_out,
            index_disp_target=self._target_start_index.get(DISPLACEMENTS_KEY, 0),
            index_vel_target=self._target_start_index.get(UPDATE_VELOCITIES_KEY, 3),
            net_lin_mom=torch.tensor(self.config.get("net_lin_mom", [])),
            net_ang_mom=torch.tensor(self.config.get("net_ang_mom", [])),
        )

        layers = {
            "Normalization": norm,
            "AtomTypeEncoding": atom_type_embedding,
            "EdgeLengthEncoding": edge_length_embedding,
            "VelocityNormEncoding": velocity_norm_embedding,
            "SHEdgeVectors": edge_sh,
            "SHVelVectors": vel_sh,
            "LinearTypeEncoding": linear_init_features,
        }

        for idx, layer in enumerate(mp_layers):
            layers[f"mp_layer_{idx}"] = layer

        layers["LinearCompression"] = linear_compress

        layers["LinearReadOut"] = linear_target
        layers["Conservation"] = conservation

        return torch.nn.Sequential(OrderedDict(layers))


class FlexibleModel(AbstractModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.layers = self._build_model()

    def forward(
        self, data: AtomicGraph, compute_forces: Optional[bool] = False
    ) -> AtomicGraph:
        data.compute_edge_vectors()
        data = self.layers(data)
        return data

    def _build_model(self) -> torch.nn.Sequential:
        layer_mapping = {
            "OneHotAtomTypeEncoding": OneHotAtomTypeEncoding,
            "EdgeLengthEncoding": EdgeLengthEncoding,
            "TensorNormEncoding": TensorNormEncoding,
            "SphericalHarmonicProjection": SphericalHarmonicProjection,
            "LinearTensorMixer": LinearTensorMixer,
            "MessagePassingLayer": MessagePassingLayer,
            "NormalizationLayer": NormalizationLayer,
            "Conservation": ConservationLayer,
            "ElementBasedNormEncoding": ElementBasedNormEncoding,
            "TimestepEncoding": TimestepEncoding,
            "ForecastHorizonConditioning": ForecastHorizonConditioning,
        }
        layers = []
        irreps_out = {}

        for layer_name, layer_dict in self.config.items():
            layer_type = list(layer_dict.keys())[0]
            layer_args = layer_dict[layer_type]
            layer_args["irreps_in"] = irreps_out
            layer = layer_mapping[layer_type](**layer_args)
            layers.append((layer_name, layer))
            irreps_out = layer.irreps_out

            if layer_name == "EdgeLengthEncoding":
                self.edge_cutoff = torch.tensor(float(layer.radial_basis.rmax))

            elif "Normalization" in layer_name:
                self.rms_targets = [std for std in layer.stds.values()]

        return torch.nn.Sequential(OrderedDict(layers))
