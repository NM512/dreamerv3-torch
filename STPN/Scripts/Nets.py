import pdb

import torch
import torch.nn as nn

from STPN.Scripts.STPLayers import (
    FastWeights,
    STPNr,
    STPNF, STPNBase,
    HebbFF
)
from torch import Tensor, no_grad
from STPN.Scripts import utils

"""
Networks for Maze are descirbed in Maze/Scripts/maze_nets.py , instead of here
"""


class SimpleNetSTPN(nn.Module):
    """ Used in Pong RLLib"""

    def __init__(self, input_size, hidden_size, stp=None, activation="tanh", bias= True, layer_norm=False):
        super().__init__()
        if stp is None:
            stp= {}

        if activation == "tanh":
            activation = torch.tanh

        self.rnn = STPNr(input_size=input_size, hidden_size=hidden_size, stp=stp, bias=bias)
        self.activation = activation
        self.layer_norm = torch.nn.LayerNorm(hidden_size) if layer_norm is True else None

    def forward(self, x, states):
        h_tp1, states_tp1 = self.rnn(x=x, states=states)
        if self.layer_norm is not None:
            h_tp1 = self.layer_norm(h_tp1)
        h_tp1 = self.activation(h_tp1)
        states_tp1 = self.rnn.update_states(
            states=states_tp1, first_element=(x, states[0][0]), second_element=(h_tp1,)
        )
        return h_tp1, states_tp1
    def forward_energy_consumption(self, x, states):
        """Only returns energy, no integrated forward pass like in SimpleNetSTPNEnergyConsumption"""
        return self.rnn.forward_energy_consumption(x, states=states)


class SimpleNetSTPMLP(nn.Module):
    def __init__(self, input_size, hidden_size, stp=None, activation="tanh", bias= True):
        super().__init__()
        if stp is None:
            stp= {}

        if activation == "tanh":
            activation = torch.tanh

        # self.rnn = jitstp.STPNF(in_features=input_size, out_features=hidden_size, stp=stp, bias=bias)
        self.rnn = STPNF(in_features=input_size, out_features=hidden_size, stp=stp, bias=bias)
        self.activation = activation

        self.zero_idx_energy = None  # TODO: support this

    def forward(self, x, states):
        h_tp1, states_tp1 = self.rnn(x=x, states=states)
        h_tp1 = self.activation(h_tp1)
        states_tp1 = self.rnn.update_states(
            states=states_tp1, first_element=(x,), second_element=(h_tp1,)
        )
        return h_tp1, states_tp1

    # with no_grad()
    @torch.no_grad()
    def forward_energy_consumption(self, x, states):
        if self.zero_idx_energy is not None:
            x[:, self.zero_idx_energy] = 0
        return self.rnn.forward_energy_consumption(x, states)


class NetParallelEnergy(nn.Module):
    """
    General Net for Maze
    Does energy evaluation in the same forward call
    Has option of separate energy evaluation
    """

    def __init__(
            self,
            input_size, hidden_size,
            rnn_type='stpn', stp=None,
            activation="tanh", bias= True, layer_norm=False,
            extra_config=None,
            eval_energy=True, energy=None,
    ):
        super().__init__()
        if stp is None: stp = {}
        if energy is None: energy = {"zero_idx": None}
        if extra_config is None: extra_config = {}

        if activation == "tanh":
            activation = torch.tanh

        # self.rnn = jitstp.STPNF(in_features=input_size, out_features=hidden_size, stp=stp, bias=bias)
        self.activation = activation
        self.zero_idx_energy = energy.get('zero_idx', None)
        self.eval_energy = eval_energy
        self.update_states_within_forward = False

        # stpnr
        if rnn_type == "stpn":
            self.rnn = STPNr(input_size=input_size, hidden_size=hidden_size, stp=stp, bias=bias, layer_norm=layer_norm)
            assert self.activation is not None
        elif rnn_type == "stplinear":
            self.rnn = STPNF(in_features=input_size, out_features=hidden_size, stp=stp, bias=bias, layer_norm=layer_norm)
            assert self.activation is not None
        elif rnn_type == "FastWeights":
            self.rnn = FastWeights(input_size=input_size, hidden_size=hidden_size, stp=stp)
            self.activation = None
            self.update_states_within_forward = True
        elif rnn_type == "hebbff":
            self.rnn = HebbFF(in_features=input_size, out_features=hidden_size, stp=stp, bias=bias)
            self.activation = torch.sigmoid
        else:
            raise NotImplementedError
        self.rnn_type = rnn_type


    def forward(self, x, states):
        if self.eval_energy is True:
            with no_grad():
                x_energy = torch.empty_like(x).copy_(x)
                if self.zero_idx_energy is not None:
                    x_energy[:, self.zero_idx_energy] = 0
                this_energy_cons = self.rnn.forward_energy_consumption(x_energy, states)
        h_tp1, states_tp1 = self.rnn(x=x, states=states)
        if self.activation is not None:
            h_tp1 = self.activation(h_tp1)
        if self.update_states_within_forward is False:
            if self.rnn_type == "stpn":
                states_tp1 = self.rnn.update_states(
                    states=states_tp1, first_element=(x, states[0][0]), second_element=(h_tp1,)
                )
            elif self.rnn_type in ["stplinear", "hebbff"]:
                states_tp1 = self.rnn.update_states(
                    states=states_tp1, first_element=(x, ), second_element=(h_tp1,)
                )
            else:
                print(f"self.rnn_type {self.rnn_type} not in stpn or stplinear")
                raise ValueError
        if self.eval_energy is True:
            return h_tp1, states_tp1, this_energy_cons
        else:
            return h_tp1, states_tp1


    def forward_energy_consumption(self, x, states):
        with no_grad():
            if self.zero_idx_energy is not None:
                x[:, self.zero_idx_energy] = 0
        return self.rnn.forward_energy_consumption(x, states)


class NoEmbed_AssociativeNet(nn.Module):
    """
    Network for Associative Retrieval Task (Ba et al 2016) without embedding.
    """

    def __init__(
            self,
            dictionary_size: int,
            hidden_dim: int,
            output_size: int,
            recurrent_unit,
            rnn_args: dict,
            rnn_activation='tanh',
    ):
        """
        Note the STPN rnn cells used do not have an activation, so the net wrapper should provide it.
        However other baselines do, so no rnn_activation should be passed.
        """
        super().__init__()

        # Parameters
        self._dictionary_size = dictionary_size
        self._hidden_dim = hidden_dim

        # Layers
        if 'stp' in rnn_args:
            print(rnn_args['stp'])
        self.rnn = recurrent_unit(dictionary_size, hidden_dim, **rnn_args)
        # please provide functional activations
        # (eg. 'functional_tanh' so we get torch.tanh instead of torch.nn.Tanh )
        assert (rnn_activation is None or rnn_activation[:11] == 'functional_')
        self.rnn_activation = utils.torch_str_to_object(rnn_activation, instantiate=False) \
            if rnn_activation is not None else None
        print('RNN activation', self.rnn_activation)
        self.hidden2tag = nn.Linear(hidden_dim, output_size)

    def forward(self, sentence: Tensor, states):
        # input character passed to embedding
        # Batch first assumption, seq second
        seq_len = sentence.size()[1]

        for seq_element_idx in range(seq_len):
            seq_element = sentence[:, seq_element_idx]
            output, states = self.rnn(seq_element, states)
            # make sure self.rnn has an activation
            if self.rnn_activation is not None:
                output = self.rnn_activation(output)

            if isinstance(self.rnn, STPNBase):
                if self.rnn_activation is not None:
                    states = tuple(tuple(self.rnn_activation(state) if (i_s == 0 and i_sg == 0) else state
                                         for i_s, state in enumerate(state_group))
                                   for i_sg, state_group in enumerate(states))
                if isinstance(self.rnn, STPNF):
                    # output = torch.tanh(output)
                    assert self.rnn_activation is not None or self.rnn.activation is not None, \
                        "Net wrapper or RNN cell should have activation function"
                    second_element = (output,)
                elif isinstance(self.rnn, STPNr):
                    assert self.rnn_activation is not None or self.rnn.activation is not None, \
                        "Net wrapper or RNN cell should have activation function"
                    second_element = (*states[0],)
                else:
                    # if this is stpnr uncomment the previous elif, just check the class of self.rnn
                    # i just dont know for sure which one is stpnr
                    raise NotImplementedError
                states = self.rnn.update_states(
                    states=states, first_element=(seq_element,*states[0]), second_element=second_element
                )

        # This is many to one, so fully conected only after final hidden state
        tag_space = self.hidden2tag(output)  # pass only hidden state

        return tag_space, states

    def forward_energy(self, sentence, states, train=False):
        # input character passed to embedding
        # Batch first assumption, seq second
        seq_len = sentence.size()[1]

        energy = torch.empty(sentence.shape[0], seq_len).to(sentence.device)
        for seq_element_idx in range(seq_len):
            seq_element = sentence[:, seq_element_idx]
            with no_grad():
                self.rnn.eval()
                energy[:, seq_element_idx] = torch.sum(self.rnn.forward_energy_consumption(seq_element, states), dim=-1)
                if train is True:
                    self.rnn.train()

            output, states = self.rnn(seq_element, states)
            # make sure self.rnn has an activation
            if self.rnn_activation is not None:
                output = self.rnn_activation(output)

            if isinstance(self.rnn, STPNBase):
                if self.rnn_activation is not None:
                    states = tuple(tuple(self.rnn_activation(state) if (i_s == 0 and i_sg == 0) else state
                                         for i_s, state in enumerate(state_group))
                                   for i_sg, state_group in enumerate(states))
                if isinstance(self.rnn, STPNF):
                    assert self.rnn_activation is not None or self.rnn.activation is not None, \
                        "Net wrapper or RNN cell should have activation function"
                    second_element = (output,)
                elif isinstance(self.rnn, STPNr):
                    assert self.rnn_activation is not None or self.rnn.activation is not None, \
                        "Net wrapper or RNN cell should have activation function"
                    second_element = (*states[0],)
                else:
                    # if this is stpnr uncomment the previous elif, just check the class of self.rnn
                    # i just dont know for sure which one is stpnr
                    raise NotImplementedError
                states = self.rnn.update_states(
                    states=states, first_element=(seq_element,*states[0]), second_element=second_element
                )

        # This is many to one, so fully conected only after final hidden state
        tag_space = self.hidden2tag(output)  # pass only hidden state

        return tag_space, states, energy
