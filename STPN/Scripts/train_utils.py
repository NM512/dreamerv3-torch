import pdb

from torch.utils.data import DataLoader



from datetime import datetime

import numpy as np
import torch
from torch.nn import CrossEntropyLoss
from torch.optim import SGD
from torch.autograd import Variable
from torch.nn.utils import clip_grad_value_
from torch import isnan, cuda, no_grad, save
from torch import device as thdevice
from tqdm.notebook import tqdm
from torch.utils.data import TensorDataset

from STPN.Scripts import utils


def capture_stats(stats, stats_args, metric_name, metric_value):
    if metric_name + '_list' in stats_args.get('output'):
        stats[metric_name + '_list'].append(metric_value)
    if metric_name + '_max' in stats_args.get('output'):
        if metric_value > stats[metric_name + '_max']:
            stats[metric_name + '_max'] = metric_value
    return stats


def early_stopping(model, states, stats, convergence_args, stats_args, train_args):
    if stats["i_epoch"] > convergence_args["max_epochs"]:  # reached max epochs
        print("Early stopping, max_epochs reached!")
        return True, convergence_args, stats
    elif train_args.get("train_convergence") is True:
        if convergence_args["convergence_evaluation_metric"] + '_list' in stats_args.get("output"):
            metric = stats[convergence_args["convergence_evaluation_metric"] + '_list'][-1]
        else:
            metric = convergence_args["convergence_evaluation"](model, convergence_args["validation_dataloader"],
                                                                train_args=train_args,
                                                                **convergence_args.get("convergence_evaluation_args",
                                                                                       {}))
        # print(f"metric {metric}, best metric {convergence_args['best_metric']}. New best metric? {(-1+2*int(convergence_args['metric_higher_is_better'])) * metric > convergence_args['best_metric']}")
        if (-1 + 2 * int(convergence_args["metric_higher_is_better"])) * metric > convergence_args[
            "best_metric"]:  # new best metric found
            print("New best metric")
            convergence_args["best_metric"] = metric  # update new best metric
            convergence_args["epochs_no_improvement"] = 1  # reset

            # save it!
            model_name = stats_args['model_name']
            # TODO: change timestamp format as it cant be used in windows
            final_path = f"{train_args['model_path']}/{model_name}-best-{stats_args['timestamp']}"
            # final_path = f"{MODEL_SAVING_PATH}/{model_name}-best-{stats_args['timestamp']}"
            args_not_to_save = ["convergence_evaluation", "validation_dataloader"]
            convergence_args_to_save = {i: convergence_args[i] for i in convergence_args if i not in args_not_to_save}
            if stats_args['keep_checkpoints'] is True:
                save(
                    obj={
                        'epoch': stats['i_epoch'],
                        'model_state_dict': model.state_dict(),
                        # 'optimizer_state_dict': optimizer.state_dict(),
                        # 'loss': loss,
                        'stats': stats,
                        # 'convergence_args':convergence_args_to_save,
                        # 'stats_args':  stats_args,
                        # 'train_args': train_args,
                    },
                    f=final_path
                )
        elif convergence_args["epochs_no_improvement"] > convergence_args[
            "tolerance"]:  # no improvement for longer than patience, EARLY STOP!
            print("Early stopping, no improvenment for longer than patience!")
            return True, convergence_args, stats  # convergence = True
        else:  # no improvement, still haven't reached patience
            convergence_args["epochs_no_improvement"] += 1

        # We calculate it outside now
        # # save validation results if desired
        # stats = capture_stats(stats, stats_args, "validation_acc", metric)

    return False, convergence_args, stats  # recommend convergence = False


def accuracy_evaluation(
        model: torch.nn.Module,
        dataloader: torch.utils.data.DataLoader,
        train_args: dict,
        init_states=None,  # TODO: deprecate this
        verbose: bool = False,
        **kwargs
):
    """
    Used in ART
    """
    if type(dataloader.dataset) is TensorDataset:
        label_set_shape = dataloader.dataset.tensors[1].size()
    else:
        raise Exception(f"Dataset of type {type(dataloader.dataset)} not supported")

    states = init_states
    acc = 0

    model.eval()
    with no_grad():
        for i, (datapoints_, labels_) in enumerate(dataloader):
            # preds = model(datapoints_.to(device))
            datapoints_ = datapoints_.to(train_args.get("device"))
            if states is not None and train_args.get("stateful", True) is True:
                states = tuple((state.to(train_args["device"]) for state in states))
            else:
                states = None

            preds, states = model(datapoints_, states)

            # preds, _ = model(datapoints_, states)
            if train_args.get("output_type", "many2many") == "many2many":
                acc += (preds.argmax(dim=2).view(-1) == labels_.view(-1).to(
                    train_args.get("device"))).float().sum().cpu().item()
            elif train_args.get("output_type", "many2many") == "many2one":
                acc += (preds.argmax(dim=1) == labels_.to(train_args.get("device"))).float().sum().cpu().item()
            else:
                raise Exception(
                    f"train_args.get('output_type') should be many2many or many2one in order to calculate accuracy properly")
    number_predictions = 1
    for dim in range(len(label_set_shape)):
        number_predictions = int(number_predictions * label_set_shape[dim])

    acc /= number_predictions
    if verbose is True:
        print(f"Model achieved {acc} test accuracy")
    model.train()
    return acc


def energy_evaluation(model, dataloader, train_args, init_states=None, verbose=False, **kwargs):
    if type(dataloader.dataset) is TensorDataset:
        label_set_shape = dataloader.dataset.tensors[1].size()
    else:
        raise Exception(f"Dataset of type {type(dataloader.dataset)} not supported")
    # print("init_states", init_states)
    states = init_states
    # acc = 0

    energy_consumption_per_forward = []
    model.eval()
    with no_grad():
        for i, (datapoints_, labels_) in enumerate(dataloader):
            # preds = model(datapoints_.to(device))
            datapoints_ = datapoints_.to(train_args.get("device"))
            if states is not None and train_args.get("stateful", True) is True:
                raise Exception('Are you sure you want to do this statefully?')
                # states = tuple((state.to(train_args["device"]) for state in states))
            else:
                states = None

            # preds, states = model(datapoints_, states)

            preds, states, energy_per_forward = model.forward_energy(datapoints_, states)
            # energy = (seqs, batch_size, seq_len, hidden) -> (total_seqs, seq_len)
            energy_consumption_per_forward.append(energy_per_forward)  # .unsqueeze(0))

            # # preds, _ = model(datapoints_, states)
            # if train_args.get("output_type", "many2many") == "many2many":
            #     acc += (preds.argmax(dim=2).view(-1) == labels_.view(-1).to(
            #         train_args.get("device"))).float().sum().cpu().item()
            # elif train_args.get("output_type", "many2many") == "many2one":
            #     acc += (preds.argmax(dim=1) == labels_.to(train_args.get("device"))).float().sum().cpu().item()
            # else:
            #     raise Exception(
            #         f"train_args.get('output_type') should be many2many or many2one in order to calculate accuracy properly")
    number_predictions = 1
    for dim in range(len(label_set_shape)):
        number_predictions = int(number_predictions * label_set_shape[dim])

    model.train()
    return torch.cat(energy_consumption_per_forward, dim=0)  # (total_seqs, seq_len)



def train(
        model: torch.nn.Module,
        data_args: dict,
        train_args: dict = {
            "loss_function": CrossEntropyLoss,
            "optimizer": SGD,
            "optimizer_args": {"lr": 0.001},
            "device": None,
            "train_convergence": True,
        },
        convergence_args: dict = {
            "max_epochs": 1000,
            "tolerance": 10,
            "metric_higher_is_better": True,  # eg. for accuracy True, for loss (in general lower loss is better) False
        },
        stats_args: dict = {
            "output": None,
            "display_rate": 1000,
            "model_name": "MODEL",
        },
        states=None,
):
    # TODO default arguments should beinmutable -> mae them Optional[Dict], then if None, use default

    # Check input convergence args
    required_convergence_args = ["tolerance", "metric_higher_is_better", "validation_dataloader", "max_epochs"]
    if train_args.get("train_convergence") is True and None in (convergence_args.get(convergence_arg) for
                                                                convergence_arg in convergence_args):
        raise Exception(
            f"Not all required convergence arguments have been provided. convergence_args= {convergence_args}, "
            f"required_convergence_args = {required_convergence_args}"
        )

    # Initialise training objects
    if train_args.get("device") is None:
        if cuda.is_available():
            train_args["device"] = thdevice('cuda')
        else:
            train_args["device"] = thdevice('cpu')

    model.to(train_args.get("device"))

    if train_args.get("resume_checkpoint", False) is True:
        # If resuming from checkpoint
        stats = train_args["resume_checkpoint_stats"]
        # TODO: be consistent with this behavious: when loading, we expect instatiated objects, when from scratch we instatiate within train function
        loss_function = train_args["loss_function"]
        optimizer = train_args["optimizer"]
    else:
        # If starting from scratch
        stats = {"i_epoch": 1, "best_model_state": None}
        loss_function = utils.torch_str_to_object(train_args["loss_function"])
        optimizer = utils.torch_str_to_object(train_args["optimizer"], instantiate=False)(
            model.parameters(), **train_args["optimizer_args"]
        )

    # Initialise convergence tracking vars
    # TODO should these be in stats???? It is specially confusing for resuming training for checkpoint, as these do matter
    if convergence_args.get("epochs_no_improvement") is None:
        convergence_args["epochs_no_improvement"] = 1
    if convergence_args.get("best_metric") is None:
        convergence_args["best_metric"] = -np.float('inf')

    # Initialise stats tracking
    for metric_name in stats_args.get("output"):
        if "_list" in metric_name:
            stats[metric_name] = []
        if "_max" in metric_name:
            stats[metric_name] = -np.float('inf')
    timestamp = datetime.now().isoformat(timespec='seconds')
    stats_args["timestamp"] = timestamp

    # Instantiate dummy variables
    continue_training = True
    for _ in tqdm(range(convergence_args["max_epochs"]), position=0, leave=True):
        for sentence, tags in data_args["train_dataloader"]:
            if sentence.size()[0] != data_args["batch_size"]:
                # drop last batch which isn't full
                continue
            model.zero_grad()
            # Send batch to cuda if still in cpu
            sentence = sentence.to(train_args.get("device"))
            tags = tags.to(train_args.get("device"))
            # forward
            tag_scores, states = model(sentence, states)

            # flatten the batch and sequences, so all frames are treated equally
            if train_args.get("output_type", "many2many") == "many2many":
                loss = loss_function(tag_scores.view(-1, data_args["output_size"]), tags.view(-1))
                # TODO: use 2D loss_function, ie. reshape to be (batch, seq_len, *)
            elif train_args.get("output_type", "many2many") == "many2one":
                loss = loss_function(tag_scores.view(-1, data_args["output_size"]), tags.view(-1))
            else:
                raise Exception(
                    f"train_args.get('output_type') should be many2many or many2one in order to calculate accuracy properly")
            loss.backward()
            # TODO: clip grad ?
            if train_args.get("grad_clip_val", None) is not None:
                clip_grad_value_(model.parameters(), train_args.get("grad_clip_val"))

            optimizer.step()

            # stateful, but detach (create new Variable) so that backpropagation doesn't update previous batch
            if train_args.get("stateful", True) is True:
                states = tuple(Variable(state.data).to(train_args.get("device")) for state in states)
            else:
                states = None

        if "validation_acc_list" in stats_args.get("output"):
            metric = convergence_args["convergence_evaluation"](model, convergence_args["validation_dataloader"],
                                                                train_args=train_args,
                                                                **convergence_args.get("convergence_evaluation_args",
                                                                                       {}))
            stats = capture_stats(stats, stats_args, "validation_acc", metric)
            if metric > convergence_args["best_metric"] and train_args["train_convergence"] is False:
                convergence_args["best_metric"] = metric
        if "validation_energy_list" in stats_args.get("output"):
            metric = energy_evaluation(model, convergence_args["validation_dataloader"], train_args=train_args,
                                       **convergence_args.get("convergence_evaluation_args", {}))
            stats = capture_stats(stats, stats_args, "validation_energy", metric)

        if stats["i_epoch"] % stats_args.get("display_rate") == 0:
            print()
            print('-' * 90)
            print(f"Stats at epoch {stats['i_epoch']}:")
            print("best_metric", convergence_args["best_metric"])
            for stat_name, stat_val in stats.items():
                if stat_name in ["best_model_state", "i_epoch"]:
                    pass
                elif "_list" in stat_name:
                    print(f"Last {stat_name[:-5]}: {stat_val[-1]}")
                else:
                    print(f"{stat_name}: {stat_val}")
            model_name = stats_args['model_name']
            final_path = f"{train_args['model_path']}/{model_name}-checkpoint-{stats_args['timestamp']}"
            if stats_args['keep_checkpoints'] is True:
                save(
                    obj={
                        'epoch': stats['i_epoch'],
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                    },
                    f=final_path
                )

        stats["i_epoch"] += 1

        stopping_recommendation, convergence_args, stats = early_stopping(model, states, stats, convergence_args,
                                                                          stats_args, train_args)
        if stopping_recommendation is True:
            return model, states, stats
    return model, states, stats
