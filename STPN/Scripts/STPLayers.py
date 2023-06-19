import pdb
from math import sqrt
from typing import Union, Tuple, List, Optional, Dict

import numpy as np
import torch

from torch.nn.functional import (
    linear as f_linear,
)

from torch.nn import (
    Module as nnModule,
    Parameter,
)

from torch import (
    Tensor, zeros,
    einsum, clamp, nn, empty, cat, no_grad
)
from torch.autograd import Variable
from STPN.Scripts import utils


class STPNBase(nnModule):
    def __init__(
            self,
            learn_plastic_weight_params: bool = True,
            learn_plastic_weight_params_dims: Optional[Union[List[int], Tuple[int]]] = None,
            plastic_weights_update_kwargs: Optional[Dict] = None,
            plastic_weight_clamp_val: float = 1000.0,
            learn_plastic_weight_params_kwargs: Optional[dict] = None,
            plastic_weights_init_config: Optional[dict] = None,
            plastic_bias: bool = False,
            learn_plastic_bias_params: bool = False,
            learn_plastic_bias_params_kwargs: Optional[dict] = None,
            learn_plastic_bias_params_dims: Optional[List[int]] = None,
            plastic_bias_clamp_val: float = 1000.0,
            plasticity_type: Optional[Union[dict, str]] = None,
            plasticity_type_kwargs: Optional[Dict] = None,
            plasticity_rule: str = "hebb",
            init_stp: bool = True,
            **kwargs
    ):
        """
        Base class for all modules with plastic weights.
        Weight and bias are your usuaal fixed, backprop trained and inference fixed parameters plastic weights and
        bias are dynamic weights that are sequence lived, updated based on the sequence being processed plastic
        weights and bias' params usually refer to parameters (learnt or otherwise) that control how plastic weights
        are updated.

        @param learn_plastic_weight_params:
        Whether parameters (lambda, gamma) controlling the update of plastic_weight should be learnt through backprop
        @param learn_plastic_weight_params_dims: dimensions along which plastic weight params lambda & gamma specialise
        @param plastic_weights_update_kwargs: configure the timing of the plastic weights update within the forward
        pass. See _set_plastic_weights_update() for arguments and default values.
        @param plastic_weight_clamp_val: Value at which plastic weight is clamped
        @param learn_plastic_weight_params_kwargs: kwargs for plastic weights params,
        like fixed values in case no learning is chosen. See _set_plastic_weight_param for arguments.
        Currently, {'param_name': {'fixed_parameter_value': float}}
        @param plastic_weights_init_config: kwargs for the random initialisation for each plastic weight (and bias) to
        be learnt. Keys should be parameter names, values dictionaries with the config for each parameter's init. See
        specialised_stp_weight_init() for arguments and default values.
        @param plastic_bias: Whether to use a plastic bias or not
        @param learn_plastic_bias_params: Learn plastic bias' parameters (lambda and gamma) thorugh backprop.
        If false, you need to provide default values in learn_plastic_bias_params_kwargs.
        @param learn_plastic_bias_params_kwargs:
        @param plastic_bias_clamp_val: Value at which plastic bias is clamped.
        @param plasticity_type: Plasticity type used for update. If dictionary is given, keys should be weight/bias and
        value the plasticity type. String means uniform plasticity type is used for weight and bias.
        Supported plasticity types are stp, prompt_decay_stp and stp_weight_dependent.
        @param plasticity_type_kwargs: Detail configuration of plastic update. If further plastic weight params are to
        be learnt besides lambda and gamma (like the weight dependent factor in weight dependent STP), its config
        should be defined here.
        # @param which_relative_layer_input_use_postsynaptic:
        """
        super().__init__()
        if init_stp is True:
            assert plastic_weights_init_config is None or isinstance(plastic_weights_init_config, dict)
            self.plastic_weights_init_config = {} if plastic_weights_init_config is None\
                else plastic_weights_init_config

            self._set_plasticity_type(plasticity_type=plasticity_type)

            # Plastic bias
            self.plastic_bias = plastic_bias
            self.learn_plastic_bias_params = learn_plastic_bias_params

            # STP updates
            if plastic_weights_update_kwargs is None:
                plastic_weights_update_kwargs = {}
            self._set_plastic_weights_update(**plastic_weights_update_kwargs)

            # Layer STP weights
            if learn_plastic_weight_params_kwargs is None:
                learn_plastic_weight_params_kwargs = {}

            self.weight_lambda = self._set_plastic_weight_param(
                weight_shape=self.weight_shape,
                learn_parameter=learn_plastic_weight_params, learn_parameter_dims=learn_plastic_weight_params_dims,
                **learn_plastic_weight_params_kwargs.get("weight_lambda", {})
            )
            # TODO in case we wanted differentiated dimensions or learnable characteristis for gamma and lambda,
            #  just turn learn_plastic_weight_params etc into a dict keyed by "gamma", "lambda"
            self.weight_gamma = self._set_plastic_weight_param(
                weight_shape=self.weight_shape,
                learn_parameter=learn_plastic_weight_params, learn_parameter_dims=learn_plastic_weight_params_dims,
                **learn_plastic_weight_params_kwargs.get("weight_gamma", {})
            )

            # Clamping of the dynamic weights to control stability
            assert isinstance(plastic_weight_clamp_val, (float, int)) or plastic_weight_clamp_val is None
            self.plastic_weight_clamp_val = plastic_weight_clamp_val if plastic_weight_clamp_val is not None \
                else float('inf')

            if plasticity_type["weight"] == "stp_weight_dependent":
                # Weight dependent plasticity, legacy
                raise NotImplementedError
            else:
                # we still have to instatiate it since we pass it to the static plastic_update call inside
                # update_dynamic_efficacies.
                self.weight_dependence_factor = None

            if self.plastic_bias is True:
                # clamping of bias
                assert isinstance(plastic_bias_clamp_val, (float, int))
                self.plastic_bias_clamp_val = plastic_bias_clamp_val

                if learn_plastic_bias_params_kwargs is None:
                    learn_plastic_bias_params_kwargs = {}
                self.bias_lambda = self._set_plastic_weight_param(
                    weight_shape=self.bias_shape, learn_parameter=learn_plastic_weight_params,
                    learn_parameter_dims=learn_plastic_bias_params_dims,
                    **learn_plastic_bias_params_kwargs.get("bias_lambda", {})
                )
                # TODO in case we wanted differentiated dimensions or learnable characteristis for gamma and lambda,
                #  just turn learn_plastic_weight_params etc into a dict keyed by "gamma", "lambda"
                self.bias_gamma = self._set_plastic_weight_param(
                    weight_shape=self.bias_shape, learn_parameter=learn_plastic_weight_params,
                    learn_parameter_dims=learn_plastic_bias_params_dims,
                    **learn_plastic_bias_params_kwargs.get("bias_gamma", {})
                )

                # set up for learnable plastic bias params
                if self.learn_plastic_bias_params is True:
                    if plasticity_type["bias"] == "stp_weight_dependent":
                        raise NotImplementedError
                    else:
                        self.bias_dependence_factor = None

            if plasticity_rule == "hebb":
                self.delta_stp_fun = self.delta_hebb
            elif plasticity_rule == "oja":
                self.delta_stp_fun = self.delta_oja
            else:
                raise Exception(f"Plasticity rule {plasticity_rule} not supported")
            self.plasticity_rule = plasticity_rule
        else:
            pass
        self.plasticity_type_kwargs = plasticity_type_kwargs or {}

        # Use non-dict parameters
        self.plastic_weight_norm = self.plasticity_type_kwargs.get('plastic_weight_norm', {}).get("norm", None)
        self.plastic_weight_norm_time = self.plasticity_type_kwargs.get('plastic_weight_norm', {}).get("time", None)
        self.weight_norm_time = self.plasticity_type_kwargs.get('weight_norm', {}).get("time", None)
        self.weight_norm_dim = self.plasticity_type_kwargs.get('weight_norm', {}).get("dim", 2)
        self.weight_norm = 'weight_norm' in self.plasticity_type_kwargs
        self.learn_weight_norm = self.plasticity_type_kwargs.get('weight_norm', {}).get("learn", False)
        self.learn_plastic_weight_norm = self.plasticity_type_kwargs.get('plastic_weight_norm', {}).get("learn", False)

        if isinstance(self.weight_norm_dim, int):
            g_weight_norm_shape = [s if (dim + 1 != self.weight_norm_dim) else 1 for (dim, s) in
                                   enumerate(self.weight_shape)]

        elif isinstance(self.weight_norm_dim, (tuple, list)):
            g_weight_norm_shape = [s if (dim + 1 not in self.weight_norm_dim) else 1 for (dim, s) in
                                   enumerate(self.weight_shape)]
        else:
            raise ValueError
        if self.learn_weight_norm:
            self.g_weight_norm = Parameter(torch.ones(g_weight_norm_shape))
            print('ENSUE TIS inits to 1')
        if self.learn_plastic_weight_norm:
            self.g_plastic_weight_norm = Parameter(torch.ones(g_weight_norm_shape))

    def forward(
            self,
            x: Tensor,
            states=None,
    ):
        """
        @param x: external input, (batch, *)
        @param states: neural and synaptic states, in respective order. For synaptic states, weight before bias
        @return:
        """
        if states is None:
            states = self.states_init(batch_size=x.shape[0], device=x.device)
        if self._stp_update_within_forward is True:
            # In this general implementation, we want to make the call to update_states explicit
            raise NotImplementedError
        else:
            h_tp1, states = self.layer_forward(x, states)

        return (h_tp1, states)  # noqa

    def forward_energy_consumption(
            self,
            x: Tensor,
            states=None,
    ):
        raise NotImplementedError

    def states_init(
            self, batch_size: int, device: torch.device
    ):
        raise NotImplementedError

    def plastic_bias_init(self, batch_size: int):
        raise NotImplementedError

    def layer_forward(
            self, x: Tensor, states
    ):
        raise NotImplementedError

    def update_states(
            self,
            states,
            first_element: Tensor,
            second_element: Tensor,
            **kwargs
    ):
        """
        Update all states in the net. By default, update only synaptic states.
        Override if your net uses neuronal states (eg SPNr, STPNl)
        """
        return (
            self.update_dynamic_efficacies(
                synaptic_states=states[-1], first_element=first_element, second_element=second_element, **kwargs
            ),
        )

    def update_dynamic_efficacies(
            self,
            synaptic_states,
            first_element: Tensor,
            second_element: Tensor,
            **kwargs  # noqa
    ):
        """
        Return new plastic weights after Hebbian update
        @param synaptic_states: Synaptic states are synaptic memroies or plastic biases (technically non-synaptic0>
        Any state/memory that is non-neural  and hence not updated like usual RNN or LSTM states.
        @param first_element: presynaptic activity for Hebbian associativity calculation
        @param second_element: postsynaptic activity for Hebbian associativity calculation
        @param kwargs:
        @return: Updated non neural (synaptic) states
        """
        weight_norm_time = self.plasticity_type_kwargs.get('plastic_weight_norm', {}).get("time", None)
        if weight_norm_time == 'pre':
            old_plastic_weight = synaptic_states[0]
        else:
            old_plastic_weight = synaptic_states[0]

        new_plastic_weight = self.plastic_update(
            plastic_weight=old_plastic_weight,
            delta_plastic_weight=self.delta_stp_weights(first_element, second_element, synaptic_states[0]),
            weight_lambda=self.weight_lambda, weight_gamma=self.weight_gamma,
            weight_clamp_val=self.plastic_weight_clamp_val,
            plasticity_type=self.plasticity_type["weight"],
            weight=self.weight,
            weight_dependence_factor=self.weight_dependence_factor
        )

        # # # normalise only the plastic weights after update
        if weight_norm_time == 'post':
            new_plastic_weight = self._plastic_weight_norm(
                new_plastic_weight, returns=['F'], norm=self.plasticity_type_kwargs['plastic_weight_norm']['norm'])[0]

        if self.plastic_bias is False:
            return (new_plastic_weight, )
        else:
            return (
                new_plastic_weight,
                self.plastic_update(
                    plastic_weight=synaptic_states[1],
                    delta_plastic_weight=self.delta_plastic_bias(first_element, second_element),
                    weight_lambda=self.bias_lambda, weight_gamma=self.bias_gamma,
                    weight_clamp_val=self.plastic_bias_clamp_val,
                    plasticity_type=self.plasticity_type["bias"],
                    weight=self.bias,
                    weight_dependence_factor=self.bias_dependence_factor
                )
            )

    # TODO may be static
    def plastic_update(
            self,
            plastic_weight: Tensor,
            delta_plastic_weight: Tensor,
            plasticity_type: str = "stp",
            weight_lambda: Optional[Union[Tensor, float]] = None, weight_gamma: Optional[Union[Tensor, float]] = None,
            weight_clamp_val: Optional[float] = None,
            weight: Optional[Tensor] = None, weight_dependence_factor: Optional[Union[Tensor, float]] = None
    ) -> Tensor:
        neg_weight_clamp_val = -weight_clamp_val if weight_clamp_val is not None else weight_clamp_val
        if plasticity_type == "stp":
            return clamp(  # type: ignore
                plastic_weight * weight_lambda + weight_gamma * delta_plastic_weight,
                min=neg_weight_clamp_val, max=weight_clamp_val
            )
        elif plasticity_type == "stp_prompt_decay":
            # lambda is applied after delta_weight is added, so decay comes before new information acts on input
            return clamp(  # type: ignore
                weight_lambda * (plastic_weight + weight_gamma * delta_plastic_weight),
                min=neg_weight_clamp_val, max=weight_clamp_val
            )
        elif plasticity_type == "stp_weight_dependent":
            return clamp(
                plastic_weight * weight_lambda + weight_gamma * (
                        weight_dependence_factor + (1 - weight_dependence_factor) * weight) * delta_plastic_weight,
                min=neg_weight_clamp_val, max=weight_clamp_val
            )  # type: ignore
        else:
            raise Exception(f"Type of plastic update {plasticity_type} not supported")

    def _plastic_weight_norm(
            self, plastic_weight: Tensor, returns: Optional[List[str]] = None, norm: Optional[str] = None,
            norm_vals: Optional[Dict[str, Tensor]] = None,
    ) -> Union[Tuple, Tuple[Tensor], Tuple[Tensor, Tensor], Tuple[Tensor, Tensor, Tensor]]:
        if returns is None:
            returns = ['F']

        # Chose what weights to normalise
        if norm == 'F':
            total_weights = plastic_weight
        elif norm == 'G':
            # broadcasting directly instead of repeat
            if self.plastic_weight_connections == 'all':
                total_weights = self.weight[:, self.plastic_input_slice] + plastic_weight
            else:
                # for W and F of different dimensionality, need to repeat W to batch dimension,
                # then W + F by slicing the required synapses; which is slower.
                total_weights = self.weight.unsqueeze(0).repeat(plastic_weight.shape[0], 1, 1)
                total_weights[:, :, self.plastic_input_slice] = total_weights[:, :, self.plastic_input_slice] + \
                                                                plastic_weight
        else:
            raise Exception(f"Weight norm type {norm} not supported")

        # Obtain the norm (compute or use provided norm previously computed)
        if isinstance(norm_vals, dict) and norm in norm_vals:
            norm_tensor = norm_vals[norm]
        else:
            if isinstance(self.weight_norm_dim, int):
                norm_tensor = torch.linalg.norm(
                    input=total_weights,
                    ord=2,
                    dim=self.weight_norm_dim,
                    keepdim=True,
                )
            else:
                norm_tensor = torch.linalg.norm(
                    input=torch.flatten(total_weights, 1),
                    ord=2,
                    dim=1,  # we flatten and normalise all
                )[:, None, None]  # expand

        if norm == 'G' and self.learn_weight_norm:
            norm_tensor = self.g_weight_norm * norm_tensor
        elif norm == 'F' and self.learn_plastic_weight_norm:
            norm_tensor = self.g_plastic_weight_norm * norm_tensor
        results = ()
        if 'N' in returns:
            results += norm_tensor,
        if 'G' in returns:
            results += total_weights / (norm_tensor + 1e-16),
        if 'F' in returns:
            results += plastic_weight / (norm_tensor + 1e-16),
        if 'W' in returns:
            self.weight.div_(norm_tensor.mean(0) + 1e-16)
        return results

    def init_weights(self):
        default_plastic_weights_init_config = {
            "weight_lambda": {
                "mode": "uniform", "mean": 0.5, "spread": 0.5, "hidden_weighting": None,
            },
            "weight_gamma": {
                "mode": "uniform", "mean": 0, "spread": 0.001, "hidden_weighting": "both"
            },
            "bias_lambda": {},
            "bias_gamma": {},
        }

        # add keys not present in given config dict with default values.
        plastic_weights_init_config = {**default_plastic_weights_init_config, **self.plastic_weights_init_config}

        stdv = sqrt(self.k_for_weight_init())
        for name, param in self.state_dict().items():
            # print("weight", dir(weight))
            if name in ["weight_gamma", "weight_lambda", "bias_lambda", "bias_gamma"]:
                self.specialised_stp_weight_init(
                    param=param, stdv=stdv, **plastic_weights_init_config[name]
                )
            elif name in ['g_weight_norm', 'g_plastic_weight_norm']:
                continue
            else:
                print(f'WARNING: initialising parameter {name} with default values')
                param.data.uniform_(-stdv, stdv)

    # TODO may be static
    def specialised_stp_weight_init(
            self,
            param: Tensor,
            mode: Optional[str] = None, hidden_weighting: Optional[str] = None,
            mean: Optional[float] = None, stdv: Optional[float] = None,
            spread: Optional[float] = None,
    ):
        if mode == "uniform":
            if hidden_weighting == "both":
                param.data.uniform_(-spread * stdv + mean, spread * stdv + mean)
            elif hidden_weighting is None:
                param.data.uniform_(-spread + mean, spread + mean)
            elif hidden_weighting == "below":
                param.data.uniform_(-spread * 2 * stdv + mean, mean)
            elif hidden_weighting == "above":
                param.data.uniform_(mean, spread * 2 * stdv + mean)
        else:
            raise Exception(f"Mode {mode} for stp weight initialisation not supported.")

    def delta_stp_weights(
            self, first_element: Tensor, second_element: Tensor, plastic_weights: Tensor,
    ) -> Tensor:
        raise NotImplementedError

    def delta_hebb(
            self, first_element: Tensor, second_element: Tensor, *args, **kwargs
    ) -> Tensor:
        return einsum('bf,bh->bhf', first_element, second_element)

    def delta_oja(
            self, first_element: Tensor, second_element: Tensor, plastic_weights: Tensor,
    ) -> Tensor:
        return einsum('bh,bf->bhf', second_element,
                      first_element - einsum('bh,bhf->bf', second_element, plastic_weights))

    def delta_plastic_bias(
            self, presynaptic: Union[Tensor, Tuple[Tensor]], postsynaptic: Tensor,
    ) -> Tensor:
        # TODO: consitent arguments with delta_plastic_weight (which takes presynaptic Tensor, not Tuple[Tensor])
        if isinstance(presynaptic, tuple):
            return postsynaptic[0]  # given as a tuple, chose the hidden state at idx 0
        else:
            return presynaptic

    def k_for_weight_init(self):
        raise NotImplementedError

    def set_stp_parameters(self, stp_params_dict: Dict[str, float]):
        if self._learn_stp_parameters is False:
            self.weight_lambda = stp_params_dict["weight_lambda"]
            self.weight_gamma = stp_params_dict["weight_gamma"]
        else:
            raise Exception("Setting of STP parameters for learnable STP not implemented")

    def _set_plasticity_type(self, plasticity_type: str):
        """ Set plasticity type. Called during STPLayerBase init"""
        valid_plasticity_types = ["stp", "stp_prompt_decay", "stp_weight_dependent"]
        if plasticity_type is None:
            # use default
            self.plasticity_type = {
                "weight": "stp",
                "bias": "stp"
            }
        elif isinstance(plasticity_type, str):
            assert plasticity_type in valid_plasticity_types
            self.plasticity_type = {
                "weight": plasticity_type,
                "bias": plasticity_type
            }
        elif isinstance(plasticity_type, dict):
            # we expect plasticity_type = {"weight":"stp", "bias": "prompt_stp"}
            valid_keys_for_plasticity_type = ["weight", "bias"]
            # make sure we have not passed plasticity type for a non-supported parameter
            assert all(key in valid_keys_for_plasticity_type for key in plasticity_type)
            # check all plasticity types are valid
            assert all(value in valid_plasticity_types for value in plasticity_type.values())
            self.plasticity_type = {
                "weight": plasticity_type["weight"],  # weight MUST be given
                "bias": plasticity_type.get("bias", plasticity_type["weight"])
                # if bias is not given, use the same as for weight
            }
        else:
            raise Exception(f"Plasticity type must be dict, str or None, not {type(plasticity_type)}")

    def detach_state_from_graph(self, state: Union[Tuple[Tensor], Tensor], device):
        if isinstance(state, tuple):
            return tuple(self.detach_state_from_graph(state=each_state, device=device) for each_state in state)
        else:
            return Variable(state.data).to(device)

    # TODO may be static
    def _set_plastic_weight_param(
            self,
            weight_shape: Union[Tensor, None] = None,
            learn_parameter: bool = False,
            learn_parameter_dims: Union[Tuple, None] = None,
            fixed_parameter_value: Union[float, None] = None,
    ):
        if learn_parameter is True:
            assert weight_shape is not None
            # If no dimensions to learn are given, it assumes learning all dimensions
            if learn_parameter_dims is None or len(learn_parameter_dims) == 0:  # learn scalar param
                parameter_dims = [1]
            elif len(learn_parameter_dims) > 0:  # learn higher order dims
                # for instance if weight is (40, 784) and we want to learn the hidden states,
                # (40,784)[0] would pick the first dim
                assert len(weight_shape) >= len(
                    learn_parameter_dims), f"You are learning STP params with higher dimensions than their weight! " \
                                           f"weight_shape={weight_shape}, learn_parameter_dims={learn_parameter_dims}"
                parameter_dims = [weight_shape[dim] for dim in learn_parameter_dims]
            else:
                raise Exception(f"learn_parameter_dims {learn_parameter_dims} is not valid. Must be list of ints ("
                                f"dims to learn), empty list or None (scalar parameter)")
            return Parameter(empty(parameter_dims))
        else:
            assert fixed_parameter_value is not None, "you need to give a value if you don't train it. specifiy in " \
                                                      "stp.learn_plastic_weight_params_kwargs.{param_name}." \
                                                      "fixed_parameter_value"
            return fixed_parameter_value

    def _set_plastic_weights_update(
            self, stp_update_within_forward: bool = False, stp_update_before_transformation: bool = False,
            stp_update_postsynaptic: bool = True
    ):
        """
        @param stp_update_within_forward:
        Carry out update of plastic weight and bias during the forward pass of STPlasticLayer
        @param stp_update_before_transformation:
        Carry out update of plastic weight and bias before calulating the output of the forward pass
        (only with presynaptic activity)
        @param stp_update_postsynaptic:
        Use postsynaptic activity to calculate the update to plastic weight and bias
        """
        assert (type(stp_update_within_forward) is bool)
        self._stp_update_within_forward = stp_update_within_forward
        assert (type(stp_update_before_transformation) is bool)
        self._stp_update_before_transformation = stp_update_before_transformation

        # if update before transformation, no update with postsynaptic is possible
        assert (isinstance(stp_update_postsynaptic, bool) or self._stp_update_before_transformation is True)
        self._stp_update_postsynaptic = stp_update_postsynaptic

    def save_input(self, x: Tensor, states: Union[Tensor, Tuple[Tensor]]) -> Tuple[Tensor]:
        """
        Tuple of elements to be used as presynaptic activity for Hebbian update for this layer
        As a convention, store input first, neural states second, synaptiv states / plastic weights last (connectionist
        weights first, biases second)

        @param x: input to layer
        @param states: states before making forward pass. This includes neural and synaptic states.
        """
        return (x,)

    def save_output(self, x: Tensor, states: Union[Tensor, Tuple[Tensor]]) -> Tuple[Tensor]:
        return (x,)  # store hidden output only by default


class STPNF(STPNBase):
    def __init__(
            self,
            in_features,  # size of input
            out_features,  # size of output
            stp: dict,
            bias=True,
            layer_norm=False,
    ):

        # ==== Layer characteristics ====
        self.in_features = in_features
        self.out_features = out_features
        self.weight_shape = (out_features, in_features)
        self.bias_shape = (out_features,)

        self.layer_norm = torch.nn.LayerNorm(out_features) if layer_norm is True else None
        self.plastic_input_slice = slice(in_features)
        self.plastic_weight_connections = stp.get("plastic_weight_connections")
        super().__init__(**stp)

        self.weight = Parameter(empty(out_features, in_features))
        if bias is True:
            self.bias = Parameter(empty(out_features))
        else:
            self.bias = None

        # Numerical initialisation of the network parameters
        super().init_weights()

    def layer_forward(self, x, states, plastic_bias=None):
        # functional bias combining fixed and plastic biases.
        bias = self.bias if self.plastic_bias is False else self.bias + states[-1][1]
        if 'weight_norm' in self.plasticity_type_kwargs:
            (total_weights,) = self._plastic_weight_norm(states[-1][0], returns=['G'], norm='G')
            h_tp1 = einsum('bf,bhf->bh', x, total_weights)
        else:
            # separated fixed weights and plastic weights
            h_tp1 = nn.functional.linear(x, self.weight, bias) + einsum('bf,bhf->bh', x, states[-1][0])
        if self.layer_norm is not None:
            h_tp1 = self.layer_norm(h_tp1)
        return h_tp1, states

    def flat_list_to_tuple_states(self, states):
        """Cast states in flat list to tuple, for RLLib"""
        assert len(states) == 1 + int(self.plastic_bias), \
            f'Expected {1 + int(self.plastic_bias)} plastic states but got {len(states)}, states'
        tupled_states = ((states[-1],),)
        if self.plastic_bias is True:
            tupled_states = (tupled_states[-1] + (states[1],))
        return tupled_states

    def tuple_states_to_flat_list(self, states, batch_size=None):
        """Cast states in tuple to flat list, for RLLib"""
        if batch_size is None:
            raise Exception("Batch size must be provided")

        stp_states = 1 + int(self.plastic_bias)

        if len(states[-1]) == stp_states:
            assert all([states[-1][i].shape[0] == batch_size for i in range(stp_states)]), \
                "STP states are separated by type, but do not have the correct amount of elements according to batch size"
            # that's great! we have batches grouped be type of stp memory!
            if isinstance(states[-1], tuple):
                # all is done, it's already a tuple!
                stp_states = states[-1]
            elif isinstance(states[-1], list):
                stp_states = [stp_state for stp_state in states[-1]]
            else:
                raise NotImplementedError
        elif len(states[-1]) == batch_size:
            assert stp_states == 1, "STP layer has plastic bias, however only one type of stp memories where provided " \
                                    "(either weights or bias)"
            stp_states = [states[-1]]
        else:
            raise Exception(
                f"dim 1 STP states in state[1] have the wrong dimensions. Expected batch size {batch_size} "
                f"or number of stp states {stp_states}, but got {len(states[-1])}"
            )

        flat_states = []
        for stp_state in stp_states:
            flat_states.append(stp_state)
        return flat_states

    def states_init(self, batch_size, device):
        if self.plastic_bias is False:
            return ((self.plastic_weight_init(batch_size).to(device),),)
        else:
            return (
                (self.plastic_weight_init(batch_size).to(device), self.plastic_bias_init(batch_size).to(device)),
            )

    def plastic_weight_init(self, batch_size):
        return zeros(batch_size, self.out_features, self.in_features)

    def plastic_bias_init(self, batch_size):
        return zeros(batch_size, self.out_features)

    def delta_stp_weights(self, first_element, second_element, plastic_weights):
        if isinstance(first_element, tuple):
            first_element = first_element[0]
        if isinstance(second_element, tuple):
            second_element = second_element[0]
        return self.delta_stp_fun(first_element, second_element, plastic_weights)

    def k_for_weight_init(self):
        return 1 / self.in_features

    def forward_energy_consumption(self, x, states, forward=None):
        with no_grad():
            if states is None:
                states = self.states_init(batch_size=x.shape[0], device=x.device)
            # inputs as voltage, square
            total_input = torch.mul(x, x)
            # abs
            if 'weight_norm' in self.plasticity_type_kwargs:
                (total_weights,) = self._plastic_weight_norm(states[-1][0], returns=['G'], norm='G')
            else:
                total_weights = torch.empty_like(self.weight).copy_(self.weight).unsqueeze(0).repeat(x.shape[0], 1, 1)
                total_weights = total_weights + states[-1][0]
            total_weights = total_weights.abs()
            return einsum('bf,bhf->bh', total_input, total_weights)


class HebbFF(STPNF):
    def __init__(
            self,
            in_features,  # size of input
            out_features,  # size of output
            stp: dict,
            bias=True,
            layer_norm=False,
    ):

        assert bias is True
        assert layer_norm is False

        # check we only learn scalar stp
        assert stp['learn_plastic_weight_params_dims'] is None or \
               (isinstance(stp['learn_plastic_weight_params_dims'], list) and len(
                   stp['learn_plastic_weight_params_dims']) == 0)
        assert stp['plastic_bias'] is False
        assert stp["plastic_weight_clamp_val"] is None
        assert 'weight_norm' not in stp['plasticity_type_kwargs']
        assert 'plastic_weight_norm' not in stp['plasticity_type_kwargs']

        super().__init__(
            in_features,  # size of input
            out_features,  # size of output
            stp,
            bias,
            layer_norm,
        )

        # using their prefered init and boundary values
        # see init_hebb() https://github.com/dtyulman/hebbff/blob/master/networks.py#L81
        lam = 0.99
        self._weight_lambda = nn.Parameter(torch.tensor(np.log(lam / (1. - lam))))
        if self.weight_lambda:
            del self.weight_lambda
        self.weight_lambda = torch.sigmoid(self._weight_lambda)

        gamma = -5. / out_features
        if self.weight_gamma:
            del self.weight_gamma

        # force AntiHebb
        self._weight_gamma = nn.Parameter(torch.log(torch.abs(torch.tensor(gamma))))  # eta = exp(_eta)
        self.weight_gamma = -torch.exp(self._weight_gamma)

    def layer_forward(self, x, states, plastic_bias=None):
        h_tp1, states = super().layer_forward(x, states, plastic_bias)

        # they call this every time in update_hebb, after A (our F) is used
        self.weight_lambda = torch.sigmoid(self._weight_lambda)
        # force AntiHebb
        self.weight_gamma = -torch.exp(self._weight_gamma)
        # no forcing gives slightly better results in ART & Maze

        return h_tp1, states


class STPNr(STPNBase):
    def __init__(
            self,
            input_size: int,  # size of input
            hidden_size: int,  # size of output
            stp: dict,
            bias: bool = True,
            layer_norm: bool = False,
    ):

        # ====  Layer characteristics ====
        self.input_size = input_size
        self.hidden_size = hidden_size

        self.weight_ih_shape = (hidden_size, input_size)
        self.weight_hh_shape = (hidden_size, hidden_size)

        # Optionally specialise stp
        assert stp.get("plastic_weight_connections", None) in ["input", "hidden", "all"], (
            f"stp.plastic_weight_connections must be one of 'input', 'hidden', 'all',"
            f" not {stp.get('plastic_weight_connections', None)}"
        )
        connections2dims = {"input": input_size, "hidden": hidden_size, "all": input_size + hidden_size}
        connections2slices = {"input": slice(input_size), "hidden": slice(input_size, input_size + hidden_size),
                              "all": slice(input_size + hidden_size)}
        self.plastic_in_features = connections2dims[stp.get("plastic_weight_connections")]
        self.plastic_input_slice = connections2slices[stp.get("plastic_weight_connections")]
        self.plastic_weight_connections = stp.get("plastic_weight_connections")

        self.weight_shape = (hidden_size, self.plastic_in_features)
        self.bias_shape = (hidden_size,)

        super().__init__(**stp)

        self.layer_norm = torch.nn.LayerNorm(hidden_size) if layer_norm is True else None
        self.weight = Parameter(empty(hidden_size, hidden_size + input_size))
        if bias is True:
            self.bias = Parameter(empty(hidden_size))
        else:
            self.bias = None

        # Numerical initialisation of the network parameters
        super().init_weights()

    # @profile
    def layer_forward(
            self,
            x: torch.Tensor,
            states: Tuple[Tuple[Tensor], Tuple[Tensor]]
    ) -> Tuple[Tensor, Tuple[Tuple[Tensor], Tuple[Tensor]]]:
        """
        @param x: input element (batch, in_features)
        @param states: tuple containing hidden neural states (batch, out_features),
        plastic_weight (batch, out_features, ...) [last dim can be out_features, in_features or out_features +
        in_features; depending on plasticity configuration and optionally plastic_bias (batch, out_features)
        @return:
        """
        # concatenate external input x and neural states into a total input vector
        total_input = cat((x, states[0][0]), dim=1)
        # functional bias combining fixed and plastic biases.
        bias = self.bias if self.plastic_bias is False else self.bias + states[-1][1]

        if self.weight_norm is True:
            if self.plastic_weight_norm_time == 'pre':
                (total_weights, plastic_weight) = self._plastic_weight_norm(states[-1][0], returns=['G', 'F'], norm='G')
                states = (states[0], (plastic_weight, states[-1][0:]))
            else:
                (total_weights,) = self._plastic_weight_norm(states[-1][0], returns=['G'], norm='G')
            h_tp1 = einsum('bf,bhf->bh', total_input, total_weights)
            if bias is not None:
                h_tp1 += bias
        else:
            h_tp1 = f_linear(total_input, self.weight, bias) + \
                    einsum('bf,bhf->bh', total_input[:, self.plastic_input_slice], states[-1][0])

        if self.layer_norm is not None:
            h_tp1 = self.layer_norm(h_tp1)
        return h_tp1, ((h_tp1,), states[-1])

    def states_init(self, batch_size: int, device: torch.device):
        if self.plastic_bias is False:
            return (
                (self.neural_states_init(batch_size).to(device),), (self.plastic_weight_init(batch_size).to(device),)
            )
        else:
            return (
                (self.neural_states_init(batch_size).to(device),),
                (self.plastic_weight_init(batch_size).to(device), self.plastic_bias_init(batch_size).to(device))
            )

    def flat_list_to_tuple_states(self, states: List[Tensor]):
        tupled_states = ((states[0],),)
        tupled_states += ((states[1],),)
        if self.plastic_bias is True:
            tupled_states = (tupled_states[0], tupled_states[1] + (states[2],))

        return tupled_states

    def tuple_states_to_flat_list(
            self,
            states,
            supposed_batch_size: Optional[int] = None,
    ) -> List[Tensor]:
        if supposed_batch_size is None:
            supposed_batch_size = states[0][0].shape[0]

        stp_states = 1 + int(self.plastic_bias)

        if len(states[1]) == stp_states:
            assert all([states[1][i].shape[0] == supposed_batch_size for i in range(stp_states)]), \
                "STP states do not have the correct amount of elements according to batch size"
            # we have batches grouped be type of stp memory
            if isinstance(states[1], tuple):
                # all is done, it's already a tuple!
                stp_states = states[1]
            elif isinstance(states[1], list):
                stp_states = [stp_state for stp_state in states[1]]
            else:
                raise NotImplementedError
        elif len(states[1]) == supposed_batch_size:
            assert stp_states == 1, "STP layer has plastic bias, however only one type of stp memories where" \
                                    " provided (either weights or bias)"
            stp_states = [states[1]]
        else:
            raise Exception(
                f"dim 1 STP states in state[1] have the wrong dimensions. Expected batch size {supposed_batch_size} "
                f"or number of stp states {stp_states}, but got {len(states[1])}"
            )
        if isinstance(states[0], torch.Tensor):
            flat_states = [states[0]]
        elif isinstance(states[0], list) and isinstance(states[0][0], torch.Tensor):
            flat_states = states[0]
        elif isinstance(states[0], tuple) and isinstance(states[0][0], torch.Tensor):
            flat_states = [states[0][0]]
        else:
            raise NotImplementedError

        for stp_state in stp_states:
            flat_states.append(stp_state)
        return flat_states

    def plastic_weight_init(self, batch_size: int) -> Tensor:
        return zeros(batch_size, self.hidden_size, self.plastic_in_features)

    def plastic_bias_init(self, batch_size: int) -> Tensor:
        return zeros(batch_size, self.hidden_size)

    def neural_states_init(self, batch_size: int) -> Tensor:
        return zeros(batch_size, self.hidden_size)

    def delta_stp_weights(self, first_element: Tensor, second_element: Tensor, plastic_weights: Tensor) -> Tensor:
        # Still just an outer product. delta_stp_weights is called by update_dynamic_efficacies, who is also provided
        # with first_element. This is called by the net, who must provide correct first_element if update is called
        # outside of forward.
        return self.delta_stp_fun(first_element[:, self.plastic_input_slice], second_element, plastic_weights)

    def k_for_weight_init(self):
        return 1 / self.hidden_size

    # There are two options to deal with extra state: override update_dynamic_efficacies, so we preprocess the states
    # and concatenate hidden and synaptic for instance (although this means making a decission on for instance shape
    # of presynaptic activity in this function), or saving input at location [0]
    def update_states(
            self,
            states,
            first_element: Tuple[Tensor, Tensor],  # (x, h)
            second_element: Tuple[Tensor],  # (x, )
            **kwargs
    ):
        first_element = cat(first_element[0:2], dim=1)
        return (
            (second_element[0],),
            self.update_dynamic_efficacies(
                synaptic_states=states[-1], first_element=first_element, second_element=second_element[0]
            ))

    def save_input(
            self,
            x: Tensor,
            states,
    ) -> Tuple[Tensor, Tensor]:
        """ Returns presynaptic used as first_element in update_states"""
        # this is not ideal, but sometimes at the start of a sequence we call the saving of input before the hidden
        # state is actually initialised, as this happens inside forward. Therefore the states are None.
        hidden_state = zeros(x.shape[0], self.hidden_size).to(x.device) if states is None else states[0][0]
        return (x, hidden_state)

    def forward_energy_consumption(
            self,
            x: Tensor,
            states=None,
    ) -> Tensor:
        with no_grad():
            if states is None:
                states = self.states_init(batch_size=x.shape[0], device=x.device)

            total_input = cat((x, states[0][0]), dim=1)
            # inputs as voltage, square
            total_input = torch.mul(total_input, total_input)
            if 'weight_norm' in self.plasticity_type_kwargs:
                (total_weights,) = self._plastic_weight_norm(states[-1][0], returns=['G'], norm='G')
            else:
                total_weights = torch.empty_like(self.weight).copy_(self.weight).unsqueeze(0).repeat(x.shape[0], 1, 1)
                # add plastic weights for those synapases that do have them
                total_weights[:, :, self.plastic_input_slice] = total_weights[:, :, self.plastic_input_slice] + \
                                                                states[1][0]
            # abs
            total_weights = total_weights.abs()
            return einsum('bf,bhf->bh', total_input, total_weights)


class FastWeights(STPNr):
    def __init__(
            self,
            input_size: int,  # size of input
            hidden_size: int,  # size of output
            stp: Optional[dict],
            S: int = 1,
            bias: bool = True,
            layer_norm: bool = True,
            activation: str = 'tanh',
    ):
        self.S = S  # inner loop iterations

        if stp is None:
            stp = {
                "learn_plastic_weight_params": False,
                "plastic_weight_clamp_val": None,
                "relative_output_postsynaptic": 1,
                "plastic_bias": False,
                "plasticity_type": {
                    "weight": "stp",
                    "bias": "stp"
                },
                "plastic_weight_connections": "hidden"
            }
        super().__init__(
            input_size,
            hidden_size,
            stp=stp,
            bias=bias,
            layer_norm=layer_norm,
        )

        self.activation = utils.torch_str_to_object(activation, instantiate=True)

    def layer_forward(self, x, states):
        """
        @param x: input element (batch, in_features)
        @param states: tuple containing hidden neural states (batch, out_features),
        plastic_weight (batch, out_features, ...) [last dim can be out_features, in_features or out_features +
        in_features; depending on plasticity configuration and optionally plastic_bias (batch, out_features)
        @return:
        """
        # concatenate external input x and neural states into a total input vector
        total_input = cat((x, states[0][0]), dim=1)
        # update fast weights
        A = self.weight_lambda * states[-1][0] + self.weight_gamma * einsum('bx,by->bxy', states[0][0], states[0][0])
        # calculate initial mantained boundary conditions
        pre_vec = torch.nn.functional.linear(total_input, self.weight, bias=self.bias)
        # process this initially mantained boundary contintions to obtain the first inner loop hidden state
        h_s = pre_vec
        # it is not clear whether this h_0 is layer normalised, but my intuition is yes
        if self.layer_norm is not None:
            h_s = self.layer_norm(h_s)
        h_s = self.activation(h_s)
        for s in range(self.S):
            h_sp1 = pre_vec + einsum('bf,bhf->bh', h_s, A)
            if self.layer_norm is not None:
                h_sp1 = self.layer_norm(h_sp1)
            h_sp1 = self.activation(h_sp1)
            h_s = h_sp1
        return h_s, ((h_s,), (A,))

    def forward_energy_consumption(self, x, states, forward=None):
        """
        Calculate energy consumption of performing forward pass on x
        @param x:
        @param states:
        @param forward:
        @return:
        """
        with no_grad():
            self.eval()
            if states is None:
                states = self.states_init(batch_size=x.shape[0], device=x.device)
            total_input = cat((x, states[0][0]), dim=1)
            pre_vec = torch.nn.functional.linear(total_input, self.weight, bias=self.bias)
            A = self.weight_lambda * states[-1][0] + \
                self.weight_gamma * einsum('bx,by->bxy', states[0][0], states[0][0])
            # energy from computing maintained boundary conditions
            energy = torch.nn.functional.linear(torch.mul(total_input, total_input), self.weight.abs())
            h_s = pre_vec
            if self.layer_norm is not None:
                h_s = self.layer_norm(h_s)
            h_s = self.activation(h_s)
            for s in range(self.S):
                h_sp1 = pre_vec + einsum('bf,bhf->bh', h_s, A)
                # energy from fast weights acting on inner loop
                energy += einsum('bf,bhf->bh', torch.mul(h_s, h_s), A.abs())
                if self.layer_norm is not None:
                    h_sp1 = self.layer_norm(h_sp1)
                h_sp1 = self.activation(h_sp1)
                h_s = h_sp1
            self.train()
            return energy


class CustomLSTMCell(nn.Module):
    def __init__(self, input_size, hidden_size, bias=True, layer_norm=False, weight_norm=None, **kwargs):
        super().__init__()
        self.plastic_bias = False
        self.hidden_size = hidden_size

        self.weight = Parameter(empty(4 * hidden_size, input_size + hidden_size))
        assert isinstance(bias, bool)
        self.bias = Parameter(empty(4 * hidden_size)) if bias else None
        assert isinstance(layer_norm, bool)
        self.layer_norm = nn.LayerNorm(4 * hidden_size) if layer_norm else None
        if weight_norm is not None:
            raise NotImplementedError
        self.weight_norm = weight_norm
        self.init_weights()

    def states_init(self, batch_size, device):
        return (zeros(batch_size, self.hidden_size).to(device), zeros(batch_size, self.hidden_size).to(device))

    def forward(self, input: Tensor, states: Optional[Tuple[Tensor, Tensor]] = None):
        if states is None:
            states = self.states_init(batch_size=input.shape[0], device=input.device)

        total_input = torch.cat((input, states[0]), dim=1)
        gates = torch.nn.functional.linear(total_input, self.weight, bias=self.bias)
        if self.layer_norm is not None:
            gates = self.layer_norm(gates)
        i_gate = torch.sigmoid(gates[:, 0:self.hidden_size])
        f_gate = torch.sigmoid(gates[:, self.hidden_size:2 * self.hidden_size])
        g_gate = torch.tanh(gates[:, 2 * self.hidden_size:3 * self.hidden_size])
        o_gate = torch.sigmoid(gates[:, 3 * self.hidden_size:])

        c_tp1 = torch.mul(f_gate, states[1]) + torch.mul(i_gate, g_gate)
        h_tp1 = torch.mul(o_gate, torch.tanh(c_tp1))
        return h_tp1, (h_tp1, c_tp1)

    def forward_energy_consumption(self, x, states, forward=None):
        with no_grad():
            if states is None:
                states = self.states_init(batch_size=x.shape[0], device=x.device)
            total_input = torch.cat((x, states[0]), dim=1)
            return torch.nn.functional.linear(torch.mul(total_input, total_input), self.weight.abs())

    def init_weights(self):
        stdv = 1 / sqrt(self.hidden_size)
        for name, param in self.state_dict().items():
            param.data.uniform_(-stdv, stdv)


class CustomRNNCell(nn.Module):
    def __init__(self, input_size, hidden_size, bias=True, nonlinearity='tanh', layer_norm=False, weight_norm=False,
                 **kwargs):
        super().__init__()
        self.plastic_bias = False
        self.hidden_size = hidden_size
        self.weight = Parameter(empty(input_size + hidden_size, hidden_size))

        assert isinstance(bias, bool)
        self.bias = Parameter(empty(hidden_size)) if bias else None
        self.activation = utils.torch_str_to_object(nonlinearity) if nonlinearity is not None else None
        assert isinstance(layer_norm, bool)
        self.layer_norm = nn.LayerNorm(hidden_size) if layer_norm else None
        self.weight_norm = weight_norm

        self.init_weights()

    def states_init(self, batch_size, device):
        return zeros(batch_size, self.hidden_size).to(device)

    def forward(self, input: Tensor, hx: Optional[Tensor] = None):
        # concatenate external input x and neural states into a total input vector
        if hx is None:
            hx = self.states_init(batch_size=input.shape[0], device=input.device)
        if self.weight_norm != 0:
            norm = torch.linalg.norm(
                input=self.weight,
                ord=2,
                dim=1,  # (hidden, features)
            ).unsqueeze(-1)
            weight = self.weight / (norm + 1e-16)
        else:
            weight = self.weight
        total_inputs = cat((input, hx), dim=1)
        h_tp1 = total_inputs @ weight
        if self.bias is not None:
            h_tp1 += self.bias
        if self.layer_norm:
            h_tp1 = self.layer_norm(h_tp1)
        if self.activation is not None:
            h_tp1 = self.activation(h_tp1)
        return h_tp1, h_tp1

    def init_weights(self):
        stdv = 1 / sqrt(self.hidden_size)
        for name, param in self.state_dict().items():
            param.data.uniform_(-stdv, stdv)

    def forward_energy_consumption(self, x, states):
        with no_grad():
            if states is None:
                states = self.states_init(batch_size=x.shape[0], device=x.device)
            if self.weight_norm is True:
                norm = torch.linalg.norm(
                    input=self.weight,
                    ord=self.weight_norm,
                    dim=1,
                ).unsqueeze(-1)
                weight = self.weight / (norm + 1e-16)
            else:
                weight = self.weight

            total_inputs = cat((x, states), dim=1)
            total_inputs = torch.mul(total_inputs, total_inputs)
            weight = weight.abs()
            energy = total_inputs @ weight
        return energy


class CustomMiconiNetwork(nn.Module):
    def __init__(
            self, input_size, hidden_size, device,
            net_type='modplast', da='tanh', addpw=3, clamp_val=1.0, NBDA=1, eval_energy=False
    ):
        super(CustomMiconiNetwork, self).__init__()
        self.type = net_type
        self.softmax = torch.nn.functional.softmax
        self.activ = torch.tanh
        self.da = da
        self.addpw = addpw
        self.clamp_val = clamp_val

        self.hidden_size = hidden_size
        self.input_size = input_size
        self.device = device

        if self.type == 'rnn':
            self.i2h = torch.nn.Linear(input_size, hidden_size).to(device)
            self.w = torch.nn.Parameter((.01 * torch.rand(hidden_size, hidden_size)).to(device),
                                        requires_grad=True)
        elif net_type == 'lstm':
            self.lstm = torch.nn.LSTMCell(input_size, hidden_size, bias=True).to(device)
        elif net_type == 'modplast':
            self.i2h = torch.nn.Linear(input_size, hidden_size).to(device)
            self.w = torch.nn.Parameter((.01 * torch.t(torch.rand(hidden_size, hidden_size))).to(device),
                                        requires_grad=True)
            self.alpha = torch.nn.Parameter(
                (.01 * torch.t(torch.rand(hidden_size, hidden_size))).to(device), requires_grad=True)
            self.h2DA = torch.nn.Linear(hidden_size, NBDA).to(device)
        elif net_type == 'plastic':
            self.i2h = torch.nn.Linear(input_size, hidden_size).to(device)
            self.w = torch.nn.Parameter((.01 * torch.rand(hidden_size, hidden_size)).to(device),
                                        requires_grad=True)
            self.alpha = torch.nn.Parameter((.01 * torch.rand(hidden_size, hidden_size)).to(device),
                                            requires_grad=True)
            self.eta = torch.nn.Parameter((.01 * torch.ones(1)).to(device),
                                          requires_grad=True)  # Everyone has the same eta
        elif net_type == 'modul' or net_type == 'modul2':
            self.i2h = torch.nn.Linear(input_size, hidden_size).to(device)
            self.w = torch.nn.Parameter((.01 * torch.rand(hidden_size, hidden_size)).to(device),
                                        requires_grad=True)
            self.alpha = torch.nn.Parameter((.01 * torch.rand(hidden_size, hidden_size)).to(device),
                                            requires_grad=True)
            self.etaet = torch.nn.Parameter((.01 * torch.ones(1)).to(device),
                                            requires_grad=True)  # Everyone has the same etaet
            self.h2DA = torch.nn.Linear(hidden_size, NBDA).to(device)
        else:
            raise ValueError("Which network type?")

        self.eval_energy = eval_energy
        self.zero_idx_energy = None

    def forward(self, inputs, states=None):
        if states is None:
            states = self.states_init(inputs.shape[0], self.device)
        hidden, hebb, et, pw = states
        BATCHSIZE = inputs.shape[0]  # self.bs
        HS = self.hidden_size

        if self.type == 'rnn':
            if self.eval_energy is True:
                with no_grad():
                    x_energy = torch.empty_like(inputs).copy_(inputs)
                    if self.zero_idx_energy is not None:
                        x_energy[:, self.zero_idx_energy] = 0
                    energy = (
                            torch.matmul(
                                torch.empty_like(self.i2h.weight).copy_(self.i2h.weight).abs(),
                                torch.mul(x_energy, x_energy).t()).t().view(BATCHSIZE, HS) +
                            torch.matmul(
                                torch.empty_like(self.w).copy_(self.w).abs().view(1, HS, HS),
                                torch.mul(hidden.view(BATCHSIZE, HS, 1), hidden.view(BATCHSIZE, HS, 1))
                            ).view(BATCHSIZE, HS)
                    ).view(BATCHSIZE, HS)

            hactiv = self.activ(self.i2h(inputs).view(BATCHSIZE, HS, 1) +
                                torch.matmul(self.w.view(1, HS, HS), hidden.view(BATCHSIZE, HS, 1))).view(BATCHSIZE, HS)
            hidden = hactiv
        elif self.type == 'lstm':
            if self.eval_energy is True:
                with no_grad():
                    x_energy = torch.empty_like(inputs).copy_(inputs)
                    if self.zero_idx_energy is not None:
                        x_energy[:, self.zero_idx_energy] = 0
                    x_2 = torch.mul(x_energy, x_energy)
                    h_2 = torch.mul(hidden[0], hidden[0])

                    weight_ih = torch.empty_like(self.lstm.weight_ih).copy_(self.lstm.weight_ih)
                    weight_ih = weight_ih.abs()
                    weight_hh = torch.empty_like(self.lstm.weight_hh).copy_(self.lstm.weight_hh)
                    weight_hh = weight_hh.abs()

                    energy = x_2 @ weight_ih.t() + h_2 @ weight_hh.t()
            hactiv, c = self.lstm(inputs, hidden)
            hidden = [hactiv, c]

        elif self.type == 'plastic':
            if self.eval_energy is True:
                raise NotImplementedError
            # Each row of w and hebb contains the input weights to a single neuron
            # hidden = x, hactiv = y
            hactiv = self.activ(self.i2h(inputs).view(BATCHSIZE, HS, 1) +
                                torch.matmul((self.w + torch.mul(self.alpha, hebb)), hidden.view(BATCHSIZE, HS, 1))
                                ).view(BATCHSIZE, HS)
            # batched outer product...should it be other way round?
            deltahebb = torch.bmm(hactiv.view(BATCHSIZE, HS, 1), hidden.view(BATCHSIZE, 1, HS))

            if self.addpw == 3:
                # Note that there is no decay, even in the Hebb-rule case : additive only!
                # Hard clamp
                hebb = torch.clamp(hebb + self.eta * deltahebb, min=-self.clamp_val, max=self.clamp_val)
            elif self.addpw == 2:
                # Note that there is no decay, even in the Hebb-rule case : additive only!
                # Soft clamp
                hebb = torch.clamp(hebb + torch.clamp(self.eta * deltahebb, min=0.0) * (1 - hebb) +
                                   torch.clamp(self.eta * deltahebb, max=0.0) * (hebb + 1), min=-1.0, max=1.0)
            elif self.addpw == 1:  # Purely additive, tends to make the meta-learning diverge. No decay/clamp.
                hebb = hebb + self.eta * deltahebb
            elif self.addpw == 0:
                # We do it the normal way. Note that here, Hebb-rule is decaying.
                # There is probably a way to make it more efficient.
                hebb = (1 - self.eta) * hebb + self.eta * deltahebb

            hidden = hactiv

        elif self.type == 'modplast':
            if self.eval_energy is True:
                with no_grad():
                    x_energy = torch.empty_like(inputs).copy_(inputs)
                    if self.zero_idx_energy is not None:
                        x_energy[:, self.zero_idx_energy] = 0
                    energy = (
                            torch.matmul(
                                torch.empty_like(self.i2h.weight).copy_(self.i2h.weight).abs(),
                                torch.mul(x_energy, x_energy).t()).t().view(BATCHSIZE, HS) +
                            torch.matmul(
                                (self.w + torch.mul(self.alpha, hebb)).abs(),
                                torch.mul(hidden.view(BATCHSIZE, HS, 1), hidden.view(BATCHSIZE, HS, 1))
                            ).view(BATCHSIZE, HS)
                    ).view(BATCHSIZE, HS)

            # Here we compute the same deltahebb for the whole network, and use
            # the same addpw for the whole network too.

            # The rows of w and hebb are the inputs weights to a single neuron
            # hidden = x, hactiv = y
            hactiv = self.activ(self.i2h(inputs).view(BATCHSIZE, HS, 1) +
                                torch.matmul((self.w + torch.mul(self.alpha, hebb)), hidden.view(BATCHSIZE, HS, 1))
                                ).view(BATCHSIZE, HS)
            # Now computing the Hebbian updates...
            # With batching, DAout is a matrix of size BS x 1 (Really BS x NBDA, but we assume NBDA=1 for now in the deltahebb multiplication below)
            if self.da == 'tanh':
                DAout = torch.tanh(self.h2DA(hactiv))
            elif self.da == 'sig':
                DAout = torch.sigmoid(self.h2DA(hactiv))
            elif self.da == 'lin':
                DAout = self.h2DA(hactiv)
            else:
                raise ValueError("Which transformation for DAout ?")

            # deltahebb has shape BS x HS x HS
            # Each row of hebb contain the input weights to a neuron
            # batched outer product...should it be other way round?
            deltahebb = torch.bmm(hactiv.view(BATCHSIZE, HS, 1), hidden.view(BATCHSIZE, 1, HS))

            if self.addpw == 3:  # Hard clamp, purely additive
                # Note that we do the same for Hebb and Oja's rule
                hebb1 = torch.clamp(hebb + DAout.view(BATCHSIZE, 1, 1) * deltahebb, min=-1.0, max=1.0)
            elif self.addpw == 2:
                # Note that there is no decay, even in the Hebb-rule case : additive only!
                hebb1 = torch.clamp(hebb + torch.clamp(DAout.view(BATCHSIZE, 1, 1) * deltahebb, min=0.0) * (1 - hebb) +
                                    torch.clamp(DAout.view(BATCHSIZE, 1, 1) * deltahebb, max=0.0) * (hebb + 1),
                                    min=-1.0, max=1.0)
            elif self.addpw == 1:  # Purely additive. This will almost certainly diverge, don't use it!
                hebb1 = hebb + DAout.view(BATCHSIZE, 1, 1) * deltahebb

            elif self.addpw == 0:
                # We do it the old way. Note that here, Hebb-rule is decaying.
                # There is probably a way to make it more efficient
                # NOTE: THIS WILL GO AWRY if DAout is allowed to go outside [0,1]!
                # Note 2: For Oja's rule, there is no difference between addpw 0 and addpw1
                hebb1 = (1 - DAout.view(BATCHSIZE, 1, 1)) * hebb + DAout.view(BATCHSIZE, 1, 1) * deltahebb
            else:
                raise ValueError("Which additive form for plastic weights?")

            hebb = hebb1
            hidden = hactiv


        elif self.type == 'modul':
            if self.eval_energy is True:
                raise NotImplementedError
            # The rows of w and hebb are the inputs weights to a single neuron
            # hidden = x, hactiv = y
            hactiv = self.activ(self.i2h(inputs).view(BATCHSIZE, HS, 1) +
                                torch.matmul((self.w + torch.mul(self.alpha, pw)), hidden.view(BATCHSIZE, HS, 1))
                                ).view(BATCHSIZE, HS)
            # Now computing the Hebbian updates...
            # With batching, DAout is a matrix of size BS x 1
            # (Really BS x NBDA, but we assume NBDA=1 for now in the deltahebb multiplication below)
            if self.da == 'tanh':
                DAout = torch.tanh(self.h2DA(hactiv))
            elif self.da == 'sig':
                DAout = torch.sigmoid(self.h2DA(hactiv))
            elif self.da == 'lin':
                DAout = self.h2DA(hactiv)
            else:
                raise ValueError("Which transformation for DAout ?")

            # We need to select the order of operations;
            # network update, e.t. update, neuromodulated incorporation into plastic weights
            # One possibility (for now go with this one):
            #    - computing all outputs from current inputs, including DA
            #    - incorporating neuromodulated Hebb/eligibility trace into plastic weights
            #    - computing updated hebb/eligibility traces
            # Another possibility (modul2):
            #    - computing all outputs from current inputs, including DA
            #    - computing updated Hebb/eligibility traces
            #    - incorporating this modified Hebb into plastic weights through neuromodulation

            # In modul2 we would compute deltaet and update et here too; here we compute them later

            if self.addpw == 3:
                # Hard clamp
                # From modplast/addpw=3:
                # hebb1 = torch.clamp(hebb + DAout.view(BATCHSIZE, 1, 1) * deltahebb, min=-1.0, max=1.0)
                deltapw = DAout.view(BATCHSIZE, 1, 1) * et
                pw1 = torch.clamp(pw + deltapw, min=-1.0, max=1.0)
            elif self.addpw == 2:
                deltapw = DAout.view(BATCHSIZE, 1, 1) * et
                # This constrains the pw to stay within [-1, 1] (we could also do that by putting a tanh on top of it,
                # but instead we want pw itself to remain within that range, to avoid large gradients and facilitate
                # movement back to 0)
                # The outer clamp is there for safety. In theory the expression within that clamp is "softly"
                # constrained to stay within [-1, 1], but finite-size effects might throw it off.
                pw1 = torch.clamp(
                    pw + torch.clamp(deltapw, min=0.0) * (1 - pw) + torch.clamp(deltapw, max=0.0) * (pw + 1),
                    min=-.99999, max=.99999)
            elif self.addpw == 1:  # Purely additive, tends to make the meta-learning diverge
                deltapw = DAout.view(BATCHSIZE, 1, 1) * et
                pw1 = pw + deltapw
            elif self.addpw == 0:
                # We do it the old way, with a decay term.
                # This will FAIL if DAout is allowed to go outside [0,1]
                # Note: this makes the plastic weights decaying!
                pw1 = (1 - DAout.view(BATCHSIZE, 1, 1)) * pw + DAout.view(BATCHSIZE, 1, 1) * et

            pw = pw1

            # Updating the eligibility trace - always a simple decay term.
            # batched outer product...should it be other way round?
            deltaet = torch.bmm(hactiv.view(BATCHSIZE, HS, 1), hidden.view(BATCHSIZE, 1, HS))
            et = (1 - self.etaet) * et + self.etaet * deltaet

            hidden = hactiv

        else:
            raise ValueError("Must select network type")
        if self.eval_energy is True:
            return hidden, [hidden, hebb, et, pw], energy
        else:
            return hidden, [hidden, hebb, et, pw]

    def forward_energy_consumption(self, inputs, states=None):
        with no_grad():
            if states is None:
                states = self.states_init(inputs.shape[0], self.device)
            hidden, hebb, et, pw = states
            BATCHSIZE = inputs.shape[0]
            HS = self.hidden_size

            if self.type == 'rnn':
                x_energy = torch.empty_like(inputs).copy_(inputs)
                if self.zero_idx_energy is not None:
                    x_energy[:, self.zero_idx_energy] = 0
                energy = (
                        torch.matmul(
                            torch.empty_like(self.i2h.weight).copy_(self.i2h.weight).abs(),
                            torch.mul(x_energy, x_energy).t()).t().view(BATCHSIZE, HS) +
                        torch.matmul(
                            torch.empty_like(self.w).copy_(self.w).abs().view(1, HS, HS),
                            torch.mul(hidden.view(BATCHSIZE, HS, 1), hidden.view(BATCHSIZE, HS, 1))
                        ).view(BATCHSIZE, HS)
                ).view(BATCHSIZE, HS)

            elif self.type == 'lstm':
                x_energy = torch.empty_like(inputs).copy_(inputs)
                if self.zero_idx_energy is not None:
                    x_energy[:, self.zero_idx_energy] = 0

                x_2 = torch.mul(x_energy, x_energy)
                h_2 = torch.mul(hidden[0], hidden[0])

                weight_ih = torch.empty_like(self.lstm.weight_ih).copy_(self.lstm.weight_ih)
                weight_ih = weight_ih.abs()
                weight_hh = torch.empty_like(self.lstm.weight_hh).copy_(self.lstm.weight_hh)
                weight_hh = weight_hh.abs()

                energy = x_2 @ weight_ih.t() + h_2 @ weight_hh.t()

            elif self.type == 'plastic':
                raise NotImplementedError

            elif self.type == 'modplast':
                x_energy = torch.empty_like(inputs).copy_(inputs)
                if self.zero_idx_energy is not None:
                    x_energy[:, self.zero_idx_energy] = 0
                energy = (
                        torch.matmul(
                            torch.empty_like(self.i2h.weight).copy_(self.i2h.weight).abs(),
                            torch.mul(x_energy, x_energy).t()).t().view(BATCHSIZE, HS) +
                        torch.matmul(
                            (self.w + torch.mul(self.alpha, hebb)).abs(),
                            torch.mul(hidden.view(BATCHSIZE, HS, 1), hidden.view(BATCHSIZE, HS, 1))
                        ).view(BATCHSIZE, HS)
                ).view(BATCHSIZE, HS)

            elif self.type == 'modul':
                raise NotImplementedError

            else:
                raise ValueError("Must select network type")

            return energy

    def initialZeroHebb(self, batch_size):
        if self.type == 'lstm':
            return None
        else:
            return Variable(torch.zeros(batch_size, self.hidden_size, self.hidden_size),
                            requires_grad=False).to(self.device)

    def initialZeroPlasticWeights(self, batch_size):
        if self.type == 'lstm':
            return None
        else:
            return Variable(torch.zeros(batch_size, self.hidden_size, self.hidden_size),
                            requires_grad=False).to(self.device)

    def initialZeroState(self, batch_size):
        BATCHSIZE = batch_size
        if self.type == 'lstm':
            return [
                Variable(torch.zeros(BATCHSIZE, self.hidden_size), requires_grad=False).to(self.device)
                for _ in range(2)
            ]
        else:
            return Variable(torch.zeros(BATCHSIZE, self.hidden_size), requires_grad=False).to(self.device)

    def states_init(self, batch_size, device):
        # [hidden, hebb, et, pw]
        return [
            self.initialZeroState(batch_size), self.initialZeroHebb(batch_size),
            self.initialZeroPlasticWeights(batch_size), self.initialZeroPlasticWeights(batch_size)
        ]

