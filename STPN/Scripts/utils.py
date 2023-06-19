import collections
import copy
import logging
import os
import pdb
import warnings
# from random import random # didn't work for random.seed(seed)
import random
from typing import Union, List, Dict, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.autograd import Variable

DATA = os.path.realpath(os.path.expanduser('~/workspace/data/'))
RESULTS = os.path.realpath(os.path.expanduser('~/workspace/STPN/results/'))
PROJECT = os.path.realpath(os.path.expanduser('~/workspace/STPN/'))


def torch_str_to_object(nn_str: Union[str, List], instantiate=True, kwargs=None):
    dictionary_nn_str_to_object = {
        # activations
        "Tanh": nn.Tanh,
        "tanh": nn.Tanh,
        "sigmoid": nn.Sigmoid,
        "relu": nn.ReLU,
        "ReLU": nn.ReLU,
        "Softmax": nn.Softmax,
        "LogSoftmax": nn.LogSoftmax,
        "functional_tanh": torch.tanh,
        "functional_sigmoid": torch.sigmoid,
        # losses
        "CrossEntropyLoss": nn.CrossEntropyLoss,
        "NLLLoss": nn.NLLLoss,
        # optimizers
        "Adam": Adam,
        # norms
        "BatchNorm2d": torch.nn.BatchNorm2d,
        "BatchNorm1d": torch.nn.BatchNorm1d,
        # pooling operators
        "MaxPool2d": torch.nn.MaxPool2d,
    }
    if isinstance(nn_str, str):
        nn_object = dictionary_nn_str_to_object.get(nn_str)
    elif isinstance(nn_str, list):
        nn_object = []
        for this_nn_str in nn_str:
            this_nn_object = dictionary_nn_str_to_object.get(this_nn_str)
            if this_nn_object is None:
                raise ValueError(f"torch.nn object returned for name {nn_str} is None."
                                 f" This might mean it is not supported by utils.nn_str_to_object")
            nn_object.append(this_nn_object)
    else:
        raise Exception(f"Input from which to instantiate torch object must be str or list of strings,"
                        f" not {type(nn_str)} for {nn_str}")

    if nn_object is None:
        raise ValueError(
            f"torch.nn object returned for name {nn_str} is None."
            f" This might mean it is not supported by utils.nn_str_to_object")

    if instantiate is True:
        if kwargs is None:
            if nn_str in ["Softmax", "LogSoftmax"]:
                kwargs = {'dim': -1}
            else:
                kwargs = {}
        nn_object = nn_object(**kwargs)

    return nn_object


def nested_list2tuple_list(nested_list):
    """Casts nested list to list of tuples
    Especially useful for parsing nested arrays encoded in JSON  that where meant to be lists of python tuples"""
    tuples_list = []
    for arrayified_tuple in nested_list:
        if arrayified_tuple is None:
            tuples_list.append(None)
        else:
            tuples_list.append(tuple(arrayified_tuple))
    return tuples_list


def _compute_conv2d_out_H_and_W(H_and_W_in, kernel_size, stride, padding, dilation):
    if stride is None:
        stride = kernel_size  # As is the default for MaxPool2d
    # Int is basically like floor, rounding down
    return (int((H_and_W_in[-2] + 2 * padding[0] - dilation[0] * (kernel_size[0] - 1) - 1) / stride[0] + 1),
            int((H_and_W_in[-1] + 2 * padding[1] - dilation[1] * (kernel_size[1] - 1) - 1) / stride[1] + 1))


def sorted_hyperparams_names_values_to_dict(params_names, params_values):
    sorted_params_names = sorted(params_names)
    hyper_params_dict = {}
    for i_param, param_name in enumerate(sorted_params_names):
        hyper_params_dict[param_name] = params_values[i_param]
    return hyper_params_dict


def get_submodule_state_dict(
        parent_state_dict: dict,
        child_prefix: str
):
    """
    Given the state_dict of parameters of parent module,
    return the state_dict of child module determined by child_prefix
    """
    submodule_state_dict = {}
    for key, value in parent_state_dict.items():
        # need to ensure it's for the first chars, as subsubmodules could also match
        if child_prefix == key[:len(child_prefix)]:
            # for new key, remove the prefix (chars) associated with parent module, plus 1 for the dot in between them
            submodule_state_dict[key[(len(child_prefix) + 1):]] = value
    return submodule_state_dict


def detach_state_from_graph(states: tuple):
    return tuple(Variable(state.data) for state in states)


def merge(a: Dict, b: Dict, path=None) -> Dict:
    """merges b into a"""
    if path is None: path = []
    for key in b:
        if key in a:
            if isinstance(a[key], dict) and isinstance(b[key], dict):
                merge(a[key], b[key], path + [str(key)])
            elif a[key] == b[key]:
                pass  # same leaf value
            else:
                a[key] = b[key]
        else:
            a[key] = b[key]
    return a


def set_seed(seed=0):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def index_else(id, db: list, else_val=None):
    assert not isinstance(else_val, int), "Altenative value cannot be an int or could be confused with index"
    try:
        db.index(id)
    except:
        return else_val


def index_no_try(id: str, db: List[str], else_val: Optional = None) -> Optional[int]:
    assert not isinstance(else_val, int), "Altenative value cannot be an int or could be confused with index"
    if id in db:
        return db.index(id)
    else:
        return else_val


def remove_unexpected_keys(model: torch.nn.Module, state_dict: dict):
    model_state_dict = model.state_dict()
    compatible_state_dict = copy.deepcopy(state_dict)
    for k in state_dict.keys():
        if not (k in model_state_dict):
            print(f"WARNING: Removing key {k} from state_dict")
            compatible_state_dict.pop(k)
    return compatible_state_dict


class GlobalParams:
    lstm_gate_list = ['f', 'g', 'i', 'o']
    gru_gate_list = [
        'n',  # new
        'r',  # reset
        'z'  # update
    ]


def prepare_two_dim_vars(x, y):
    if isinstance(x, int) and isinstance(y, int):
        joint_vars = (x, y)
    elif isinstance(x, int) and y is None:
        joint_vars = (x, x)
    elif isinstance(x, list) and y is None:  # TODO: check List[int]
        joint_vars = tuple((k_s_x, k_s_x) for k_s_x in x)
    elif isinstance(x, list) and isinstance(y, list):
        joint_vars = tuple((ksx, ksy) for ksx, ksy in zip(x, y))
    else:
        raise ValueError  # TODO: also check len
    return joint_vars


def none_or_str(value):
    if value == 'None':
        return None
    return value


def get_model_size(net: nn.Module, verbose: bool = False):
    """ Number of parameters"""
    allsizes = [torch.numel(x.data.cpu()) for x in net.parameters()]
    named_allsizes = {n: torch.numel(x.data.cpu()) for n, x in net.named_parameters()}
    n_named_params = sum(named_allsizes.values())
    assert sum(allsizes) == n_named_params, "Some parameters are not in named_parameters, parameter count is wrong"
    if verbose:
        print("Size (numel) of all optimized elements:", named_allsizes)
        print("Total size (numel) of all optimized elements:", n_named_params)
    return n_named_params, named_allsizes


def shuffle_odict(d, shuffle_f=random.shuffle):
    items = list(d.items())
    shuffle_f(items)
    o_d = collections.OrderedDict(items)
    return o_d


def sort_odict(d, sort_f=lambda x: x[0]):
    return collections.OrderedDict(sorted(d.items(), key=sort_f))
