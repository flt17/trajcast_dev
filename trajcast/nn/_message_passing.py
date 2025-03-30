from abc import abstractmethod
from typing import Dict, Optional, Union

import torch
from e3nn.nn import FullyConnectedNet
from e3nn.o3 import Irreps
from e3nn.util.jit import compile_mode
from torch_scatter import scatter

from trajcast.data._keys import (
    ATOM_TYPE_EMBEDDING_KEY,
    ATOM_TYPES_KEY,
    EDGE_LENGTHS_EMBEDDING_KEY,
    EDGE_VECTORS_KEY,
    NODE_FEATURES_KEY,
    SPHERICAL_HARMONIC_KEY,
    VELOCITIES_KEY,
)
from trajcast.data.atomic_graph import AtomicGraph
from trajcast.nn._graph_module_irreps import GraphModuleIrreps
from trajcast.nn._non_linearity import GatedNonLinearity
from trajcast.nn._tensor_cross_interactions import DepthwiseTensorProduct
from trajcast.nn._tensor_self_interactions import LinearTensorMixer
from trajcast.nn._wrapper_ops import (
    CuEquivarianceConfig,
    FullyConnectedTensorProduct,
    Linear,
)
from trajcast.utils.misc import (
    build_mlp_for_tp_weights,
    determine_irreps_for_gate,
    mlp_config_from_dictionary,
)


class AbstractMessagePassingLayer(GraphModuleIrreps, torch.nn.Module):
    def __init__(
        self,
        max_rotation_order: Optional[int] = 3,
        node_features_field: Optional[str] = NODE_FEATURES_KEY,
        edge_attributes_field: Optional[
            str
        ] = f"{SPHERICAL_HARMONIC_KEY}_{EDGE_VECTORS_KEY}",
        edge_length_embedding_field: Optional[str] = EDGE_LENGTHS_EMBEDDING_KEY,
        output_field: Optional[str] = {},
        avg_num_neighbors: Optional[Union[float, torch.Tensor]] = 10.0,
        irreps_in: Optional[Dict[str, Irreps]] = {},
        irreps_out: Optional[Union[str, list]] = {},
        edge_mlp_kwargs: Optional[Dict] = {},
        cueq_config: Optional[CuEquivarianceConfig] = None,
    ) -> None:
        super().__init__()
        self.node_features_field = node_features_field
        self.edge_attributes_field = edge_attributes_field
        self.edge_length_embedding_field = edge_length_embedding_field
        self.output_field = output_field if output_field else node_features_field
        self.lmax = max_rotation_order
        self.edge_mlp_kwargs = edge_mlp_kwargs

        if isinstance(avg_num_neighbors, float):
            avg_num_neighbors = torch.tensor(
                avg_num_neighbors, dtype=torch.get_default_dtype()
            )

        self.register_buffer(
            "avg_num_neighbors",
            avg_num_neighbors,
        )

        # check whether irreps for the input fields available
        if (
            node_features_field not in irreps_in
            or edge_attributes_field not in irreps_in
            or edge_length_embedding_field not in irreps_in
        ):
            raise KeyError(
                f"Specification of the irreps of {self.node_features_field}, {self.edge_attributes_field}, and {self.edge_length_embedding_field} needed!"
            )

        # make sure output irreps are correctly given:
        irreps_out = (
            Irreps(irreps_out).sort().irreps.simplify()
            if irreps_out
            else irreps_in[node_features_field].sort().irreps.simplify()
        )

        # update irreps
        self._init_irreps(
            irreps_in=irreps_in,
            irreps_out={self.output_field: irreps_out},
        )

        # first we define the tensor product that handles the interaction
        # between tensors in node_features_field and edge_attributes_field.
        self.dtp_edge = DepthwiseTensorProduct(
            max_rotation_order=self.lmax,
            irreps_input1=self.irreps_in[self.node_features_field],
            irreps_input2=self.irreps_in[self.edge_attributes_field],
            multiplicity_mode="uvu",
            trainable=True,
            cueq_config=cueq_config,
        )

        # we now define the weights
        # they learned based on the weight field through a MLP
        self.edge_mlp = build_mlp_for_tp_weights(
            input_field=self.edge_length_embedding_field,
            irreps_in=self.irreps_in,
            mlp_kwargs=self.edge_mlp_kwargs,
            output_dim=self.dtp_edge.weight_numel,
        )

    @abstractmethod
    def forward(self, data: AtomicGraph) -> AtomicGraph:
        raise NotImplementedError


@compile_mode("script")
class ConditionedMessagePassingLayer(AbstractMessagePassingLayer):
    def __init__(
        self,
        vel_emb_field: Optional[str] = f"{SPHERICAL_HARMONIC_KEY}_{VELOCITIES_KEY}",
        vel_len_emb_field: Optional[str] = f"norm_encoding_{VELOCITIES_KEY}",
        vel_mlp_kwargs: Optional[Dict] = {},
        nl_gate_kwargs: Optional[Dict] = {},
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        cueq_config = kwargs.get("cueq_config", None)
        self.vel_emb_field = vel_emb_field
        self.vel_len_emb_field = vel_len_emb_field

        if (
            vel_len_emb_field not in self.irreps_in
            or vel_emb_field not in self.irreps_in
        ):
            raise KeyError(
                f"Need the specification of the irreps for {vel_emb_field} and {vel_len_emb_field}."
            )

        # we now condition to the output of the contraction layer on the velocity embeddings
        # using another tensor product
        self.dtp_vel = DepthwiseTensorProduct(
            max_rotation_order=self.lmax,
            irreps_input1=self.dtp_edge.irreps_out,
            irreps_input2=self.irreps_in[self.vel_emb_field],
            multiplicity_mode="uvu",
            trainable=True,
            cueq_config=cueq_config,
        )

        # the weights for this tensor product are computed via a MLP with velocity rbfs as input
        self.vel_mlp = build_mlp_for_tp_weights(
            input_field=self.vel_len_emb_field,
            irreps_in=self.irreps_in,
            mlp_kwargs=vel_mlp_kwargs,
            output_dim=self.dtp_vel.weight_numel,
        )

        # we now update the irreps of the output to make sure it only considers those not 0
        irreps_out = Irreps(
            [
                (mul, (ir.l, ir.p))
                for mul, ir in self.irreps_out[self.output_field]
                if ir in self.dtp_vel.irreps_out.simplify()
            ]
        )

        self._init_irreps(
            irreps_in=self.irreps_in, irreps_out={self.output_field: irreps_out}
        )

        # next we build the non-linear gate
        # we start by getting the l>0 (gated) irreps
        irreps_gated = Irreps(
            [(mul, ir) for mul, ir in self.irreps_out[self.output_field] if ir.l > 0]
        )
        # now we need to get the gate scalars, either they are given or we just assume we use 0e for this
        irreps_gates = Irreps(
            nl_gate_kwargs.get("irreps_gates", f"{irreps_gated.num_irreps}x0e")
        )
        # update the required output of the linear layer
        linear_out_irreps = (
            (irreps_gates + self.irreps_out[self.output_field]).sort().irreps.simplify()
        )

        # define scalars now
        irreps_scalars, _, _ = determine_irreps_for_gate(
            irreps_input=linear_out_irreps, irreps_gates=irreps_gates
        )

        # the different channels of similar rotation order such that
        # we obtain the dimensionality either for the next layer or to put into
        # a gated non-linearity

        # note we call here the Linear module directly from e3nn
        # this is because we don't want intermediate results to be
        # saved in the graph object
        self.linear_updates = Linear(
            self.dtp_vel.irreps_out,
            linear_out_irreps,
            cueq_config=cueq_config,
        )
        # now the gate
        self.gate = GatedNonLinearity(
            input_field=self.output_field,
            irreps_scalars=irreps_scalars,
            irreps_gates=irreps_gates,
            irreps_gated=irreps_gated,
            activation_scalars=nl_gate_kwargs.get("activation_scalars", ["tanh"]),
            activation_gates=nl_gate_kwargs.get("activation_gates", ["tanh"]),
            irreps_in={self.output_field: linear_out_irreps},
            cueq_config=cueq_config,
        )

    def forward(self, data: AtomicGraph) -> AtomicGraph:
        # compute the weights for tps
        weights_dtp_edge = self.edge_mlp(data[self.edge_length_embedding_field])

        # generate messages based on dtp
        # define neighbor indices, the message sender...
        sender_indices = data.edge_index[1]
        # ... and the receiver indices, which are basically atoms we
        # we are predicting the displacement for
        receiver_indices = data.edge_index[0]

        # note: the structure already assumes that input1 is node-based
        # and input2 is edge-based. This is why we do not need [sender_indices]
        # for the input2_field
        messages = self.dtp_edge(
            data[self.node_features_field][sender_indices],
            data[self.edge_attributes_field],
            weights_dtp_edge,
        )

        # now pool messages together and normalize by average number of neighbors
        pooled_messages = (
            scatter(
                messages, receiver_indices, dim=0, dim_size=data.num_nodes, reduce="sum"
            )
            / self.avg_num_neighbors
        )

        # get weights for next tp
        weights_dtp_vel = self.vel_mlp(data[self.vel_len_emb_field])
        # perform tp
        pooled_messages = self.dtp_vel(
            pooled_messages, data[self.vel_emb_field], weights_dtp_vel
        )

        # pass through linear layer
        update = self.linear_updates(pooled_messages)

        # update field and apply gate
        data[self.output_field] = update
        data = self.gate(data)

        return data


@compile_mode("script")
class ResidualConditionedMessagePassingLayer(AbstractMessagePassingLayer):
    def __init__(
        self,
        vel_emb_field: Optional[str] = f"{SPHERICAL_HARMONIC_KEY}_{VELOCITIES_KEY}",
        vel_len_emb_field: Optional[str] = f"norm_encoding_{VELOCITIES_KEY}",
        vel_mlp_kwargs: Optional[Dict] = {},
        nl_gate_kwargs: Optional[Dict] = {},
        species_emb_field: Optional[str] = ATOM_TYPE_EMBEDDING_KEY,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        cueq_config = kwargs.get("cueq_config", None)

        self.vel_emb_field = vel_emb_field
        self.vel_len_emb_field = vel_len_emb_field
        self.species_emb_field = species_emb_field

        if (
            vel_len_emb_field not in self.irreps_in
            or vel_emb_field not in self.irreps_in
        ):
            raise KeyError(
                f"Need the specification of the irreps for {vel_emb_field} and {vel_len_emb_field}."
            )

        # now ee define the linear contraction layer to avoid too many features and weights
        # define output for the contraction
        linear_contract_out = Irreps(
            [
                (mul, (ir.l, ir.p))
                for mul, ir in self.irreps_out[self.output_field]
                if ir in self.dtp_edge.irreps_out.simplify()
            ]
        )
        self.linear_contraction = Linear(
            self.dtp_edge.irreps_out,
            linear_contract_out,
            cueq_config=cueq_config,
        )
        # we now condition to the output of the contraction layer on the velocity embeddings
        # using another tensor product
        self.dtp_vel = DepthwiseTensorProduct(
            max_rotation_order=self.lmax,
            irreps_input1=linear_contract_out,
            irreps_input2=self.irreps_in[self.vel_emb_field],
            multiplicity_mode="uvu",
            trainable=True,
            cueq_config=cueq_config,
        )

        # the weights for this tensor product are computed via a MLP with velocity rbfs as input
        self.vel_mlp = build_mlp_for_tp_weights(
            input_field=self.vel_len_emb_field,
            irreps_in=self.irreps_in,
            mlp_kwargs=vel_mlp_kwargs,
            output_dim=self.dtp_vel.weight_numel,
        )

        # we now update the irreps of the output to make sure it only considers those not 0
        irreps_out = Irreps(
            [
                (mul, (ir.l, ir.p))
                for mul, ir in self.irreps_out[self.output_field]
                if ir in self.dtp_vel.irreps_out.simplify()
            ]
        )

        self._init_irreps(
            irreps_in=self.irreps_in, irreps_out={self.output_field: irreps_out}
        )

        # next we build the non-linear gate
        # we start by getting the l>0 (gated) irreps
        irreps_gated = Irreps(
            [(mul, ir) for mul, ir in self.irreps_out[self.output_field] if ir.l > 0]
        )
        # now we need to get the gate scalars, either they are given or we just assume we use 0e for this
        irreps_gates = Irreps(
            nl_gate_kwargs.get("irreps_gates", f"{irreps_gated.num_irreps}x0e")
        )
        # update the required output of the linear layer
        linear_out_irreps = (
            (irreps_gates + self.irreps_out[self.output_field]).sort().irreps.simplify()
        )

        # define scalars now
        irreps_scalars, _, _ = determine_irreps_for_gate(
            irreps_input=linear_out_irreps, irreps_gates=irreps_gates
        )

        # Now, define the linear layer after the message update to combine
        # the different channels of similar rotation order such that
        # we obtain the dimensionality either for the next layer or to put into
        # a gated non-linearity

        # note we call here the Linear module directly from e3nn
        # this is because we don't want intermediate results to be
        # saved in the graph object
        self.linear_updates = Linear(
            self.dtp_vel.irreps_out,
            linear_out_irreps,
            cueq_config=cueq_config,
        )

        # for the resnet we perform another tensor product of the features of the previous layer
        # with the one hot encoding of the atom types.
        self.tp_resnet = FullyConnectedTensorProduct(
            self.irreps_in[self.node_features_field],
            self.irreps_in[self.species_emb_field],
            self.irreps_out[self.output_field],
            internal_weights=True,
            shared_weights=True,
            cueq_config=cueq_config,
        )

        # now the gate
        self.gate = GatedNonLinearity(
            input_field=self.output_field,
            irreps_scalars=irreps_scalars,
            irreps_gates=irreps_gates,
            irreps_gated=irreps_gated,
            activation_scalars=nl_gate_kwargs.get("activation_scalars", ["tanh"]),
            activation_gates=nl_gate_kwargs.get("activation_gates", ["tanh"]),
            irreps_in={self.output_field: linear_out_irreps},
            cueq_config=cueq_config,
        )

    def forward(self, data: AtomicGraph) -> AtomicGraph:
        # compute the weights for tps
        weights_dtp_edge = self.edge_mlp(data[self.edge_length_embedding_field])

        # generate messages based on dtp
        # define neighbor indices, the message sender...
        sender_indices = data.edge_index[1]
        # ... and the receiver indices, which are basically atoms we
        # we are predicting the displacement for
        receiver_indices = data.edge_index[0]

        # note: the structure already assumes that input1 is node-based
        # and input2 is edge-based. This is why we do not need [sender_indices]
        # for the input2_field

        messages = self.dtp_edge(
            data[self.node_features_field][sender_indices],
            data[self.edge_attributes_field],
            weights_dtp_edge,
        )

        # now pool messages together and normalize by average number of neighbors
        pooled_messages = (
            scatter(
                messages, receiver_indices, dim=0, dim_size=data.num_nodes, reduce="sum"
            )
            / self.avg_num_neighbors
        )
        # reduce the output dimension via linear contraction layer
        pooled_messages = self.linear_contraction(pooled_messages)

        # get weights for next tp
        weights_dtp_vel = self.vel_mlp(data[self.vel_len_emb_field])
        # perform tp
        pooled_messages = self.dtp_vel(
            pooled_messages, data[self.vel_emb_field], weights_dtp_vel
        )

        # pass through linear layer
        update = self.linear_updates(pooled_messages)

        # resnet
        feat_resnet = self.tp_resnet(
            data[self.node_features_field],
            data[self.species_emb_field].to(torch.get_default_dtype()),
        )
        # update field and apply gate
        data[self.output_field] = update
        data = self.gate(data)
        data[self.output_field] += feat_resnet

        # normalise
        data[self.output_field] /= 2**0.5

        return data


class MessagePassingLayer(GraphModuleIrreps, torch.nn.Module):
    def __init__(
        self,
        max_rotation_order: Optional[int] = 3,
        input1_field: Optional[str] = NODE_FEATURES_KEY,
        input2_field: Optional[str] = f"{SPHERICAL_HARMONIC_KEY}_{EDGE_VECTORS_KEY}",
        weight_field: Optional[str] = EDGE_LENGTHS_EMBEDDING_KEY,
        conditioning_field: Optional[str] = {},
        conditioning_weight_field: Optional[str] = {},
        conditioning_weights_shared: Optional[bool] = True,
        linear_between_tps: Optional[bool] = False,
        number_of_species: Optional[int] = {},
        output_field: Optional[str] = {},
        irreps_in: Optional[Dict[str, Irreps]] = {},
        irreps_out: Optional[Union[str, list]] = {},
        avg_num_neighbors: Optional[float] = None,
        non_linearity: Optional[bool] = False,
        resnet: Optional[bool] = True,
        non_linearity_after_resnet: Optional[bool] = True,
        resnet_self_interaction: Optional[bool] = False,
        resnet_sc_element: Optional[bool] = False,
        fc_kwargs: Optional[Dict] = {},
        fc_conditioning_kwargs: Optional[Dict] = {},
        tp_message_kwargs: Optional[Dict] = {},
        tp_update_kwargs: Optional[Dict] = {},
        non_linearity_kwargs: Optional[Dict] = {},
        cueq_config: Optional[CuEquivarianceConfig] = None,
    ):
        """This is at the core of TrajCast, the message passing module.

        Args:
            max_rotation_order (Optional[int], optional): Highest rotation order (l) we allow. All higher frequencies are truncated. Defaults to 3.
            input1_field (Optional[str], optional): Name of input field 1 for. Defaults to NODE_FEATURES_KEY.
            input2_field (Optional[str], optional): Name of input field 2 for TP. Defaults to f"{SPHERICAL_HARMONIC_KEY}_{EDGE_VECTORS_KEY}".
            weight_field (Optional[str], optional): Name of field from which weights for TP are derived. Defaults to EDGE_LENGTHS_EMBEDDING_KEY.
            conditioning_field (Optional[str], optional): Name of field aggregated messages are conditioned on. Defaults to {}.
            conditioning_weight_field (Optional[str], optional): Name of field the correspoding weights are derived from. Defaults to {}.
            conditioning_weights_shared (Optional[bool], optional): Whether the weights for the conditioning should be shared across all elements.
                If false, each element will have its own weights. Defaults to True.
            linear_between_tps (Optional[bool]): Whether a linear layer is used between the two tensor products to reduce the multiplicity
                of the irreps produced back to the target multiplicity. This can help to reduce the number of weights considerably.
            number_of_species (Optional[int], optional): How many species the model is designed to model overall. Needed if weights are not shared. Defaults to {}.
            output_field (Optional[str], optional): Name of field where the output is saved. Defaults to {}.
            irreps_in (Optional[Dict[str, Irreps]], optional): Dictionary with all irreps of all fields needed. Defaults to {}.
            irreps_out (Optional[Union[str, list]], optional): Updated dictionary with all irreps of all fields at end of layer. Defaults to {}.
            avg_num_neighbors (Optional[float], optional): Average number of neighbors. Can be preset here or computed from data. Both MACE and NequiP
                use precomputed values here rather than varying them based on the data.
            non_linearity (Optional[bool], optional): Whether a non linearity (gate) should be added.. Defaults to False.
            resnet (Optional[bool], optional): Whether a resnet is used. Defaults to True.
            non_linearity_after_resnet (Optional[bool], optional): Whether the non linear gate is applied before the resnet (only on update) or
                after (update+data). Defaults to True (update+data).
            resnet_self_interaction (Optional[bool], optional): Whether a self interaction in the resnet is used. Defaults to False.
            resnet_sc_element (Optional[bool], optional): Whether the self interaction in resnet uses weights based on the atomic species. Default to False.
            fc_kwargs (Optional[Dict], optional): Args for the MLP used to generated weights for TP. Defaults to {}.
            fc_conditioning_kwargs (Optional[Dict], optional): Args for the MLP used to generate weights for conditioning TP. Defaults to {}.
            tp_message_kwargs (Optional[Dict], optional): Args for the first TP. Defaults to {}.
            tp_update_kwargs (Optional[Dict], optional): Args for the conditioning TP. Defaults to {}.
            non_linearity_kwargs (Optional[Dict], optional): Args for the non-linear gate. Defaults to {}.

        Raises:
            KeyError: If irreps not given for input fields.
            KeyError: If irreps for weight field not given.
            KeyError: If irreps for conditioning field not given.

        Returns:
            _type_: _description_
        """
        super().__init__()
        self.input1_field = input1_field
        self.input2_field = input2_field
        self.weight_field = weight_field
        self.conditioning_field = conditioning_field
        self.conditioning_weight_field = conditioning_weight_field
        self.conditioning_weights_shared = conditioning_weights_shared
        self.lmax = max_rotation_order
        self.output_field = output_field if output_field else input1_field
        self.resnet = resnet
        self.non_linearity = non_linearity
        self.non_linearity_after_resnet = non_linearity_after_resnet
        self.avg_num_neighbors = avg_num_neighbors

        # having an extra linear layer to compress the output of the first TP only makes sense if there is a conditioning field
        if not self.conditioning_field and linear_between_tps:
            raise ValueError(
                "When you only have one tensor product there is no need for an extra linear layer."
            )
        self.linear_between_tps = linear_between_tps
        # if we have a resnet and non-linear gate, we need to have a self interaction otherwise irreps won't add up
        # unless we non_linearity is applied before the non_linearity
        self.resnet_self_interaction = (
            True if self.resnet and self.non_linearity else resnet_self_interaction
        )
        self.resnet_sc_element = (
            False if not self.resnet_self_interaction else resnet_sc_element
        )

        # check whether irreps for the input fields available
        if input1_field not in irreps_in or input2_field not in irreps_in:
            raise KeyError(
                f"Specification of the irreps of {self.input1_field} and {self.input2_field} needed!"
            )

        # get irreps out:
        irreps_out = (
            Irreps(irreps_out).sort().irreps.simplify()
            if irreps_out
            else irreps_in[input1_field].sort().irreps.simplify()
        )
        # we start with the message

        # first we define the tensor product that handles the interaction
        # between tensors in input1_field and input2_field.
        self.dtp_message = DepthwiseTensorProduct(
            max_rotation_order=self.lmax,
            irreps_input1=irreps_in[self.input1_field],
            irreps_input2=irreps_in[self.input2_field],
            **tp_message_kwargs,
            cueq_config=cueq_config,
        )

        # we define here the output irreps of the tensor product, which will be fed into
        # a linear layer or further updated via conditioning
        linear_in_irreps = self.dtp_message.irreps_out
        # the output will be just the output defined in the output field,
        # or, in case of a gate it will be updated later
        linear_out_irreps = irreps_out

        # we now define the weights
        # they learned based on the weight field through a MLP
        def build_fully_connected_net_for_tp_weights(
            weight_field, irreps_in, fc_kwargs, tp
        ) -> FullyConnectedNet:
            if weight_field not in irreps_in:
                raise KeyError(
                    f"Need the specification of the irreps of the field {weight_field} to get multiplicity."
                )

            # extract information from kwargs
            neurons_per_layer, activation = (
                mlp_config_from_dictionary(fc_kwargs)
                if fc_kwargs
                else ([16], torch.nn.SiLU())
            )

            # build mlp
            n_input_neurons = irreps_in[weight_field][0].mul

            # add potentially the raise of an error if more than scalars
            return FullyConnectedNet(
                [n_input_neurons] + neurons_per_layer + [tp.weight_numel],
                activation,
            )

        if weight_field:
            self.fc = build_fully_connected_net_for_tp_weights(
                weight_field=weight_field,
                irreps_in=irreps_in,
                fc_kwargs=fc_kwargs,
                tp=self.dtp_message,
            )

        # next we define the update function
        # this will depend on whether we'd like to condition the messages
        # on another tensor via a tensor product
        # and whether we'd like to have a non-linear gate included

        # first, we need to check whether there is another tensor product in the
        # update function involved, such that the summed messages are conditioned on
        # another property

        if self.conditioning_field:
            if conditioning_field not in irreps_in:
                raise KeyError("Need the specification of the conditioning irreps.")

            if linear_between_tps:
                # define output for the contraction
                linear_contract_out = Irreps(
                    [
                        (mul, (ir.l, ir.p))
                        for mul, ir in irreps_out
                        if ir in linear_in_irreps
                    ]
                )
                self.linear_contraction = Linear(linear_in_irreps, linear_contract_out)
                # update linear_in_irreps
                linear_in_irreps = linear_contract_out

            # first the tensor product
            # note that we currently don't allow for this have trainable parameters
            self.dtp_update = DepthwiseTensorProduct(
                max_rotation_order=self.lmax,
                irreps_input1=linear_in_irreps,
                irreps_input2=irreps_in[self.conditioning_field],
                **tp_update_kwargs,
                cueq_config=cueq_config,
            )
            linear_in_irreps = self.dtp_update.irreps_out

            if self.conditioning_weight_field:
                # dependent on whether the parameters are shared
                if self.conditioning_weights_shared:
                    self.fc_conditioning = build_fully_connected_net_for_tp_weights(
                        weight_field=self.conditioning_weight_field,
                        irreps_in=irreps_in,
                        fc_kwargs=fc_conditioning_kwargs,
                        tp=self.dtp_update,
                    )
                else:
                    # if we want to have different weights for different elements we create a list of MLPs
                    if not number_of_species:
                        raise ValueError(
                            "Please specify how many species the model is designed for."
                        )
                    # instead of one MLP we create a list
                    self.fc_conditioning = torch.nn.ModuleList(
                        [
                            build_fully_connected_net_for_tp_weights(
                                weight_field=self.conditioning_weight_field,
                                irreps_in=irreps_in,
                                fc_kwargs=fc_conditioning_kwargs,
                                tp=self.dtp_update,
                            )
                            for species in range(number_of_species)
                        ]
                    )

        # we will update the output irreps here:
        # if we do not have a certain irrep in input, a linear layer will just produce zeros
        # therefore, here we remove all irreps in output which are not in the input
        irreps_out = Irreps(
            [(m, ir) for (m, ir) in linear_out_irreps if ir in linear_in_irreps]
        )
        # this will be for now the output of the linear layers
        linear_out_irreps = irreps_out
        # update irreps
        self._init_irreps(
            irreps_in=irreps_in,
            irreps_out={self.output_field: irreps_out},
        )

        # next let's first check whether there's a non-linear gate and get the irreps
        # the linear layer has to produce
        if non_linearity:
            # we start by getting the l>0 (gated) irreps
            irreps_gated = Irreps(
                [
                    (mul, ir)
                    for mul, ir in self.irreps_out[self.output_field]
                    if ir.l > 0
                ]
            )

            # now we need to get the gate scalars, either they are given or we just assume we use 0e for this
            irreps_gates = Irreps(
                non_linearity_kwargs.get(
                    "irreps_gates", f"{irreps_gated.num_irreps}x0e"
                )
            )

            # update the required output of the linear layer
            linear_out_irreps = (
                (irreps_gates + self.irreps_out[self.output_field])
                .sort()
                .irreps.simplify()
            )

            # define scalars now
            irreps_scalars, _, _ = determine_irreps_for_gate(
                irreps_input=linear_out_irreps, irreps_gates=irreps_gates
            )
        # Now, define the linear layer after the message update to combine
        # the different channels of similar rotation order such that
        # we obtain the dimensionality either for the next layer or to put into
        # a gated non-linearity

        # note we call here the Linear module directly from e3nn
        # this is because we don't want intermediate results to be
        # saved in the graph object
        self.linear_updates = Linear(
            linear_in_irreps,
            linear_out_irreps,
            cueq_config=cueq_config,
        )

        # finally, we also have to do a linear layer if self interaction in resnet is given:
        if resnet and resnet_self_interaction:
            if self.non_linearity_after_resnet:
                irreps_out_resnet_sc = linear_out_irreps
            else:
                irreps_out_resnet_sc = irreps_out
            # here we use the LinearTensorMixer because we use the resnet track

            if not resnet_sc_element:
                self.linear_resnet = LinearTensorMixer(
                    input_field=input1_field,
                    output_field=self.output_field,
                    irreps_in=self.irreps_in,
                    irreps_out=irreps_out_resnet_sc,
                    cueq_config=cueq_config,
                )
            else:
                self.tp_resnet = FullyConnectedTensorProduct(
                    self.irreps_in[self.input1_field],
                    self.irreps_in[ATOM_TYPE_EMBEDDING_KEY],
                    irreps_out_resnet_sc,
                    internal_weights=True,
                    shared_weights=True,
                    cueq_config=cueq_config,
                )

        # after all this we place our gate
        if non_linearity:
            self.gate = GatedNonLinearity(
                input_field=self.output_field,
                irreps_scalars=irreps_scalars,
                irreps_gates=irreps_gates,
                irreps_gated=irreps_gated,
                activation_scalars=non_linearity_kwargs.get(
                    "activation_scalars", ["tanh"]
                ),
                activation_gates=non_linearity_kwargs.get("activation_gates", ["tanh"]),
                irreps_in={self.output_field: linear_out_irreps},
                cueq_config=cueq_config,
            )

    def forward(self, data: AtomicGraph) -> AtomicGraph:
        # compute the weights for tps
        weights_tp = self.fc(data[self.weight_field])

        # generate messages based on dtp
        # define neighbor indices, the message sender...
        sender_indices = data.edge_index[1]
        # ... and the receiver indices, which are basically atoms we
        # we are predicting the displacement for
        receiver_indices = data.edge_index[0]

        # note: the structure already assumes that input1 is node-based
        # and input2 is edge-based. This is why we do not need [sender_indices]
        # for the input2_field
        messages = self.dtp_message(
            data[self.input1_field][sender_indices],
            data[self.input2_field],
            weights_tp,
        )

        # get average number of neighbors, either via preset value or data
        if not self.avg_num_neighbors:
            self.avg_num_neighbors = data.ave_n_neighbors

        # now pool messages together and normalize by average number of neighbors
        pooled_messages = (
            scatter(
                messages, receiver_indices, dim=0, dim_size=data.num_nodes, reduce="sum"
            )
            / self.avg_num_neighbors
        )

        # now, we can proceed with the update
        if self.conditioning_field:
            # if linear layer for contraction is requested
            if self.linear_between_tps:
                pooled_messages = self.linear_contraction(pooled_messages)

            if self.conditioning_weight_field:
                if self.conditioning_weights_shared:
                    weights_tp_update = self.fc_conditioning(
                        data[self.conditioning_weight_field]
                    )
                else:
                    weights_tp_update = torch.stack(
                        [
                            self.fc_conditioning[data[ATOM_TYPES_KEY][node]](
                                data[self.conditioning_weight_field][node]
                            )
                            for node in range(data.num_nodes)
                        ]
                    )
            pooled_messages = self.dtp_update(
                pooled_messages, data[self.conditioning_field], weights_tp_update
            )

        update = self.linear_updates(pooled_messages)

        # get the node features (potentially passed through Linear) from input
        # to be added in the resnet
        if self.resnet:
            if self.resnet_self_interaction:
                if not self.resnet_sc_element:
                    feat_resnet = self.linear_resnet(data)[self.output_field]
                else:
                    feat_resnet = self.tp_resnet(
                        data[self.input1_field], data[ATOM_TYPE_EMBEDDING_KEY]
                    )
            else:
                feat_resnet = data[self.input1_field]

            data[self.output_field] = update
            if self.non_linearity:
                if not self.non_linearity_after_resnet:
                    data = self.gate(data)
                    data[self.output_field] += feat_resnet
                else:
                    data[self.output_field] += feat_resnet
                    data = self.gate(data)
            else:
                data[self.output_field] += feat_resnet

            data[self.output_field] /= 2**0.5
        else:
            data[self.output_field] = update
            if self.non_linearity:
                data = self.gate(data)

        return data
