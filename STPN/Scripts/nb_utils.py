import copy
import math
import os
import pdb
import ipdb
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np

from matplotlib.patches import Ellipse
import matplotlib.transforms as transforms


def print_values_bar(bar_vals, x_offset=-0.15, y_offset=0.05, n_dec=2, clipped_bar_vals=None):
#     x, y = [], []
    for idx, val in enumerate(bar_vals):
        y = val if clipped_bar_vals is None else clipped_bar_vals[idx]
        plt.text(x=idx + x_offset, y=y + y_offset, s=f"{val:.{n_dec}f}")

def read_acc_results_file(filename, base=None, separator=" "):
    if base is None:
        filepath = filename
    else:
        filepath = os.path.join(base, filename)
    with open(filepath, 'rb') as file:
        readlines = file.readlines()

    processing_array = False
    results = []
    for line in readlines:
        line = line.strip().decode("utf-8")
        if line[0] == "#":
            continue
        elif line[0] in ["(", '{']:
            continue
        elif line[0] == "[":
            if line.strip()[-1] == "]":
                results.append(list(map(float, line.strip()[1:-1].split(separator))))
                processing_array = False
            else:
                results.append(list(map(float, line.strip()[1:].split(separator))))
                processing_array = True
        elif processing_array is True:
            results[-1].extend(list(map(float, line.strip()[:-1].split(separator))))
            if line.strip()[-1] == "]":
                processing_array = False
        else:
            raise Exception("Not valid status line", line)
    return results

def read_single_result(filename, base=None):
    if base is None:
        filepath = filename
    else:
        filepath = os.path.join(base, filename)
    with open(filepath, 'rb') as file:
        readlines = file.readlines()

    results = []
    for line in readlines:
        results.append(float(line))
    return results


def moving_average(a, n=3, return_smoothed=False):
    """

    @param a: original array (un-smoothed)
    @param n: smoothing window size
    @param return_smoothed: return only smoothed values (first n values of a won't be returned)
    @return:
    """
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    if return_smoothed:
        return ret[n - 1:] / n
    else:
        b = copy.deepcopy(a)
        b[n - 1:] = ret[n - 1:] / n
        return b

def load_energy_results(model_name, args, test_dataset_name, energy_results=None, path_base=None, seed=None):
    if energy_results is None:
        energy_results = {}
    ending = f"_on_{test_dataset_name}"
    if not seed is None:
        ending += f"_seed_{seed}"
    ending += ".npy"

    file_path = os.path.join(
        args.path_experiment_results, "efficiency",
        os.path.basename(args.config_file_path).replace(
            args.pretrained_model_config_name, model_name).replace('.json', f'') + ending
    )

    with open(file_path, 'rb') as energy_results_file:
        energy_results[model_name] = np.load(energy_results_file)

    return energy_results

def load_energy_results_art(filename, model_name, base, energy_results=None):
    if energy_results is None:
        energy_results = {}
    file_path = os.path.join(
        base, filename + '.npy'
    )
    with open(file_path, 'rb') as energy_results_file:
        energy_results[model_name] = np.load(energy_results_file)

    return energy_results


def get_path_results(metric, base_path, secondary_base_path, filename_prefix, file_format, exp_cfg=None):
    d_cfg = {
        'bs': 512,
        'eval': False,
        'hs': 100,
        'nbiter': 200000
    }
    if exp_cfg is None:
        cfg = d_cfg
    else:
        cfg = {**d_cfg, **exp_cfg}


    suffix_base = f"bs_{cfg['bs']}_eplen_200_eval_{cfg['eval']}_hs_{exp_cfg.get('hs',100)}_lr_0.0001_nbiter" \
                  f"_{exp_cfg.get('nbiter',200000)}_net_type_{exp_cfg['net_type']}_rngseed_{exp_cfg.get('rngseed',0)}" \
                  f"_type_{exp_cfg['type']}"
    if 'config' in exp_cfg:
        suffix_base += f"_config_{exp_cfg['config']}"
    suffix = "_".join([filename_prefix[metric], suffix_base])
    suffix += file_format[metric]
    return os.path.join(base_path, secondary_base_path[metric], suffix)


def avg_diff_runs(runs, line_operator=np.mean, ci_operator=np.std):
    assert len(runs[0].shape) == 1 , "can only use for flat arrays"
    run_lens = []
    for run in runs:
        run_lens.append(len(run))
    live_runs = len(run_lens)
    done = np.zeros(live_runs)
    all_mean, all_std = [], []
    for step in range(np.amax(run_lens)):
        this_mean, this_std = [], []
        live_runs = []
        for i_run, run in enumerate(runs):
            if done[i_run] == 1:
                continue
            elif run_lens[i_run] < step + 1:
                done[i_run] = 1
            else:
                if np.isnan(run[step]):
                    print(f'WARNING: values for this metric become NaN for run {i_run} at step {step}.'
                          f' Discarding this run from now on')
                    done[i_run] = 1
                else:
                    live_runs.append(run[step])

        try:
            if np.isnan(np.mean(live_runs)) or np.isnan(np.std(live_runs)):
                print("NaN value found by avg_diff_runs in mean or std")
                print("Live runs", live_runs)
                print("Done runs", done)
                print("Step", step)
                print("Length of runs", run_lens)
                print("Value for run 0 at current step", runs[0][step])
                pdb.set_trace()
                ipdb.set_trace()
        except:
            pdb.set_trace()

        all_mean.append(line_operator(live_runs))
        all_std.append(ci_operator(live_runs))
        if done.sum() == len(done):
            break
    return all_mean, all_std


def stats_diff_lens(runs, line_operator=np.mean, ci_operator=np.std):
    """
    Get statistics from runs of different lengths. Fast version of avg_diff_runs.
    @param runs_no_nan:
    @param line_operator:
    @param ci_operator:
    @return:
    """
    assert len(runs[0].shape) == 1 , "can only use for flat arrays"
    nan_ids = [np.any(np.isnan(r)) for r in runs]
    if np.sum(nan_ids) > 0:
        print(f'WARNING: run(s) with id(s) {[i for i, v in enumerate(nan_ids) if v == True]} contain NaNs.'
              f' They will not be used by stats_diff_lens()')
        runs_no_nan_ids = [i for i, v in enumerate(nan_ids) if v == False]
        runs_no_nan = [runs[i_r] for i_r in runs_no_nan_ids]
        print("len(runs_no_nan) != len(runs)", len(runs_no_nan) != len(runs))
        print(f'After removing runs with NaN values, {len(runs_no_nan)} runs will be processed (vs {len(runs)} previously)')
    else:
        runs_no_nan = runs

    run_lens = []
    for run in runs_no_nan:
        run_lens.append(len(run))
    run_argsort = np.argsort(run_lens)
    max_len = run_lens[run_argsort[-1]]

    live_runs = len(run_lens)
    done = np.zeros(live_runs)

    all_mean, all_std = np.empty(max_len), np.empty(max_len)
    for i_block, id_next_finishing_run in enumerate(run_argsort):
        start_step = 0 if i_block == 0 else run_lens[run_argsort[i_block - 1]]
        end_step = run_lens[id_next_finishing_run]
        block_runs = np.stack([r[start_step:end_step] for i_r, r in enumerate(runs_no_nan) if done[i_r] == 0])
        all_mean[start_step:end_step] = line_operator(block_runs, axis=0)
        all_std[start_step:end_step] = ci_operator(block_runs, axis=0)
        done[id_next_finishing_run] = 1

    return all_mean, all_std


def get_immediate_subdirectories(a_dir):
    return [name for name in os.listdir(a_dir)
            if os.path.isdir(os.path.join(a_dir, name))]

def set_to_list(set):
    """ Jupyer overrides list(), so you need to call it from outside for this to work """
    return list(set)


# import math
def iqr_operator(
    array, # List[List] or np.array()
    quantile=0.25,
    axis=0,
    operation='mean', # ['mean', 'std']
    count_partial=True, # whether to count the partial quartiles
):
    quantile_elems = quantile * len(array)
    quantiles_zero_weighted = math.floor(quantile_elems*2)
    weights = np.ones(len(array))

    if quantiles_zero_weighted > 0:
        weights[:int(quantiles_zero_weighted/2)] = 0
        weights[-int(quantiles_zero_weighted/2):] = 0
    partial_quantile = 1 - (quantile_elems*2 - quantiles_zero_weighted)/2
    if partial_quantile > 0:
        if count_partial is True:
            weights[int(quantiles_zero_weighted/2)] = partial_quantile
            weights[-int((quantiles_zero_weighted/2)+1)] = partial_quantile
    else:
        weights[int(quantiles_zero_weighted/2)] = 0
        weights[-int((quantiles_zero_weighted/2)+1)] = 0

    sorted_array = np.sort(array, axis=axis)
    iqm = np.average(sorted_array, axis=axis, weights=weights)
    if operation =='mean':
        return iqm
    elif operation =='std':
        return np.sqrt(np.average((sorted_array-iqm)**2, weights=weights, axis=axis))


def cat_diff_len(
        arrays,  #: List[List],
        operator=np.mean,
        operator_kwargs=None
):
    ' Operate on list of lists of diff lengths'
    if operator_kwargs is None:
        operator_kwargs = {}

    max_len = 0
    for seq in arrays:
        if len(seq) > max_len:
            max_len = len(seq)
    done = len(arrays) * [False]

    result = []
    for i_step in range(max_len):
        this_step_elems = []
        for i_seq in range(len(arrays)):
            if done[i_seq] is True:
                continue
            elif len(arrays[i_seq]) - 1 < i_step:
                done[i_seq] = True
            else:
                this_step_elems.append(arrays[i_seq][i_step])
        result.append(operator(np.array(this_step_elems), **operator_kwargs))
    return np.array(result)


def minmax(array, axis=0):
    return np.concatenate(np.max(array, axis=axis), np.min(array, axis=axis))


def vectorize_seeds_results(
        exp_cfg,
        metrics,
        seeds,
        last_n_iters,
        max_iters,
        plot_line='mean',  # ['mean', 'median']
        plot_ci='std',  # ['std', 'minmax']
        weight_metric_by: Optional[list] = None,  # ['hs', 'gates']
        prior_dim_averaging=None,  # for energy, dim = 1; for reward, None
        ratio_iters_to_result=1,
        num_ticks=1000
):
    # for plotting
    my_xticks = np.arange(ratio_iters_to_result * (max_iters - last_n_iters), ratio_iters_to_result * max_iters + 1,
                          num_ticks * ratio_iters_to_result)
    x_my_ticks = np.arange(max_iters - last_n_iters, max_iters + 1, num_ticks)
    x = np.arange(max_iters - last_n_iters, max_iters)
    #
    line_operator = {'mean': np.mean, 'median': np.median}[plot_line]
    ci_operator = {'std': np.std, 'minmax': minmax}[plot_ci]
    # for tracking
    labels = []
    mean_rewards, std_rewards = [], []

    for exp_name, this_exp_cfg in exp_cfg.items():
        rewards_seeds = []
        for seed in seeds:
            try:
                this_seed_rewards = np.expand_dims(np.array(metrics[exp_name][seed][
                    int((max_iters - last_n_iters) / ratio_iters_to_result):int(max_iters / ratio_iters_to_result)
                ]), 0)
            except:
                print('failed loading of reward for this exp name and seed')
                pdb.set_trace()
            if prior_dim_averaging is not None:
                this_seed_rewards = np.mean(this_seed_rewards, axis=prior_dim_averaging + 1)  # +1 is for seeds
            if weight_metric_by is not None:
                for weighting_metric in weight_metric_by:
                    this_seed_rewards *= this_exp_cfg[weighting_metric]
            rewards_seeds.append(this_seed_rewards)

        try:
            rewards_seeds = np.concatenate(rewards_seeds, axis=0)  # np.array(rewards_seeds)
            this_mean_rewards = line_operator(rewards_seeds, axis=0)
            this_std_rewards = ci_operator(rewards_seeds, axis=0)
        except:
            try:
                this_mean_rewards, this_std_rewards = stats_diff_lens(
                    [reward_seed.reshape(-1) for reward_seed in rewards_seeds])
            except:
                print('failed concatenation of rewards for all seeds in this exp_name')
                pdb.set_trace()
        mean_rewards.append(this_mean_rewards)
        std_rewards.append(this_std_rewards)
        labels.append(exp_name)
    mean_rewards, std_rewards = np.array(mean_rewards), np.array(std_rewards)
    return mean_rewards, std_rewards, labels


def vectorize_results(
        exp_cfg,
        results,
        # seeds,
        last_n_iters,
        max_iters,
        metric_name=None,
        plot_line='mean',  # ['mean', 'median']
        plot_ci='std',  # ['std', 'minmax']
        weight_metric_by: Optional[list] = None,  # ['hs', 'gates']
        prior_dim_averaging=None,  # for energy, dim = 1; for reward, None
        ratio_iters_to_result=1,
        num_ticks=1000
):
    """
    Difference with vectorize_seeds_results is that we do not assume results to be indexed by seed,
    we just iterate over whatever type of key. For instance, instead of key, this can be experiment id like in KWS.
    """
    if not (max_iters is None or last_n_iters is None or num_ticks is None):
        # for plotting
        my_xticks = np.arange(ratio_iters_to_result * (max_iters - last_n_iters), ratio_iters_to_result * max_iters + 1,
                              num_ticks * ratio_iters_to_result)
        x_my_ticks = np.arange(max_iters - last_n_iters, max_iters + 1, num_ticks)
        x = np.arange(max_iters - last_n_iters, max_iters)
    #
    line_operator = {'mean': np.mean, 'median': np.median}[plot_line]
    ci_operator = {'std': np.std, 'minmax': minmax}[plot_ci]
    # for tracking
    labels = []
    mean_rewards, std_rewards = [], []
    for exp_name in results.keys():
        rewards_seeds = []
        for run_id, run_results in results[exp_name].items():
            try:
                this_run_results = run_results[metric_name] if metric_name is not None else run_results
                this_seed_rewards = np.expand_dims(np.array(this_run_results), 0)
                if not (max_iters is None or last_n_iters is None or num_ticks is None):
                    this_seed_rewards = this_seed_rewards[
                                        int((max_iters - last_n_iters) / ratio_iters_to_result):int(
                                            max_iters / ratio_iters_to_result)
                                        ]
            except:
                print('failed loading of reward for this exp name and seed')
                pdb.set_trace()
            if prior_dim_averaging is not None:
                this_seed_rewards = np.mean(this_seed_rewards, axis=prior_dim_averaging + 1)  # +1 is for seeds
            if weight_metric_by is not None:
                for weighting_metric in weight_metric_by:
                    this_seed_rewards *= exp_cfg[exp_name][weighting_metric]
            rewards_seeds.append(this_seed_rewards)

        try:
            rewards_seeds = np.concatenate(rewards_seeds, axis=0)  # np.array(rewards_seeds)
            this_mean_rewards = line_operator(rewards_seeds, axis=0)
            this_std_rewards = ci_operator(rewards_seeds, axis=0)
        except Exception as e_np_concat:
            try:
                this_mean_rewards, this_std_rewards = stats_diff_lens(
                    [reward_seed.reshape(-1) for reward_seed in rewards_seeds])
            except Exception as e_diff_len_concat:
                print('failed concatenation of rewards for all seeds in this exp_name')
                print(f'Failed to (naively) concatenate using numpy: {e_np_concat}')
                print(f'Failed to (carefully) concatenate runs with different lengths: {e_diff_len_concat}')
                pdb.set_trace()
        mean_rewards.append(this_mean_rewards)
        std_rewards.append(this_std_rewards)
        # append custom label if provided
        if exp_cfg is not None and 'label' in exp_cfg[exp_name]:
            labels.append(exp_cfg[exp_name]['label'])
        else:
            labels.append(exp_name)
    mean_rewards, std_rewards = np.array(mean_rewards), np.array(std_rewards)
    return mean_rewards, std_rewards, labels


def merge_no_conflict(
        a,
        b,  # Reference dict
        ignore_keys,  # do not add to dict
        #     varying_keys,
        conflict_handling: str = 'pdb'  # ['pdb', 'error']
):
    """
    Union of a and b, handling conflicts as desired.
    If dictionaries have unequal number of keys, having a smaller a would be generally faster.
    """
    b2 = copy.deepcopy(b)
    aUb = {}
    for k, v in a.items():
        if ignore_keys is not None and k in ignore_keys:
            continue
        elif k in b2:
            if v == b2[k]:
                aUb[k] = v
                b2.pop(k)
            else:
                handle_conflict(conflict_handling,
                                error_message=f'conflict between dictionaries for key {k}: {v} and {b2[k]}')
        else:
            # this might cause some issue if the initial reference_dict doesn't have all the keys necessary
            handle_conflict(conflict_handling,
                            error_message=f'Key {k} is present in dictionary a not present in b (thought as reference)')
    # now add all the keys in b that werent in a, supposedly there should be no conflict or overwritting
    return {**aUb, **b2}

def handle_conflict(mode: str = 'pdb', **kwargs):
    if mode == 'pdb':
        if 'error_message' in kwargs:
            print(kwargs['error_message'])
        pdb.set_trace()
    elif mode == 'error':
        if 'error_message' in kwargs:
            raise Exception(kwargs['error_message'])
        else:
            raise Exception()


class GlobalColors:
    # single variant
    sv = {
        # non plastic nets
        "LSTM" : 'brown',
        "RNN" : 'red',
        "GRU" : 'darkorange',
        "MLP" : 'gold',

        # STPN variants
        "STPN" : 'dodgerblue',
        "STPNr" : 'royalblue',
        "STPNf" : 'blueviolet',
        "STPNl" : 'purple',

        # other plastic nets
        # Miconi
        # FW
    }
    synonyms = {
        'STPNr': ['STPNR', 'STPN-r', 'STPN-R'],
        'STPNf': ['STPNF', 'STPN-f', 'STPN-F'],
        'STPNl': ['STPNL', 'STPN-l', 'STPN-L'],
    }
    for main, syn_list in synonyms.items():
        for syn in syn_list:
            sv[syn] = sv[main]

def confidence_ellipse(x, y, ax, n_std=3.0, facecolor='none', **kwargs):
    """
    Create a plot of the covariance confidence ellipse of *x* and *y*.

    Parameters
    ----------
    x, y : array-like, shape (n, )
        Input data.

    ax : matplotlib.axes.Axes
        The axes object to draw the ellipse into.

    n_std : float
        The number of standard deviations to determine the ellipse's radiuses.

    **kwargs
        Forwarded to `~matplotlib.patches.Ellipse`

    Returns
    -------
    matplotlib.patches.Ellipse
    """
    if x.size != y.size:
        raise ValueError("x and y must be the same size")

    cov = np.cov(x, y)
    pearson = cov[0, 1] / np.sqrt(cov[0, 0] * cov[1, 1])
    # Using a special case to obtain the eigenvalues of this
    # two-dimensionl dataset.
    ell_radius_x = np.sqrt(1 + pearson)
    ell_radius_y = np.sqrt(1 - pearson)
    ellipse = Ellipse((0, 0), width=ell_radius_x * 2, height=ell_radius_y * 2,
                      facecolor=facecolor, **kwargs)

    # Calculating the stdandard deviation of x from
    # the squareroot of the variance and multiplying
    # with the given number of standard deviations.
    scale_x = np.sqrt(cov[0, 0]) * n_std
    mean_x = np.mean(x)

    # calculating the stdandard deviation of y ...
    scale_y = np.sqrt(cov[1, 1]) * n_std
    mean_y = np.mean(y)

    transf = transforms.Affine2D().rotate_deg(45).scale(scale_x, scale_y).translate(mean_x, mean_y)

    ellipse.set_transform(transf + ax.transData)
    return ax.add_patch(ellipse)


rescale = lambda y, min_scale: (y - np.min(y)*min_scale) / (np.max(y) - np.min(y)*min_scale)
get_single_color_cmap = lambda cmap, last_metric : cmap(rescale(last_metric, min_scale=0.95))


def get_common_labels(
    fig,
    axs=None, axs_loc_for_legend=None, # only pass these if you want to plot legend in a specific subplot
    bbox_to_anchor=(0.925, 0.5), loc="center left", borderaxespad=-0.35,
    legend_line_size=None, legend_dot_size=None,
    ncol=1,
    sorted_labels=None, # optional list with labels sorted. label must match the exact label string, and lenght of common labels
):
    # TODO pass option to set legend for a specific subplot. Very useful here !
    lines_labels = [ax.get_legend_handles_labels() for ax in fig.axes]  # get all
    lines, labels = [sum(lol, []) for lol in zip(*lines_labels)]

    unique_labels, unique_lines = [], []
    for i in range(len(labels)):
        if labels[i] not in unique_labels:
            unique_labels.append(labels[i])
            unique_lines.append(lines[i])
    #     fig.legend(unique_lines, unique_labels, loc=(1.04, 0.5))
    if sorted_labels is not None:
        assert len(sorted_labels) == len(unique_labels), f"Different number of labels found and provided for sortign.\n Given: {sorted_labels}.\n Found: {unique_labels}"
        assert set(sorted_labels) == set(unique_labels), f"Different labels found and provided for sorting.\n Given: {sorted_labels}.\n Found: {unique_labels}"
        # pdb.set_trace() # not sure if im using lines and labels correctly
        new_unique_lines = []
        #         unique_labels = sorted_labels
        for i_l in sorted_labels:
            old_idx = unique_labels.index(i_l)
            new_unique_lines.append(unique_lines[old_idx])
        unique_lines = new_unique_lines
        unique_labels = sorted_labels
    if axs_loc_for_legend is None:
        leg = fig.legend(unique_lines, unique_labels,
                         bbox_to_anchor=bbox_to_anchor, loc=loc, borderaxespad=borderaxespad,
#                          scatterpoints=int(legend_dot_size is not None),
                         ncol=ncol)
        # Not sure if this works for fig.legend()
        if legend_line_size is not None:
            for legobj in leg.legendHandles:
                legobj.set_linewidth(legend_line_size)
        if legend_dot_size is not None:
            for i in range(len(unique_lines)):
                leg.legendHandles[i]._sizes = [legend_dot_size]
        return fig
    else:
        assert axs is not None
        if len(axs_loc_for_legend) ==2:
            plt.sca(axs[axs_loc_for_legend[0]][axs_loc_for_legend[1]])
        elif len(axs_loc_for_legend) == 1:
            plt.sca(axs[axs_loc_for_legend[0]])
        else: raise ValueError
        leg = plt.legend(unique_lines, unique_labels,
                         bbox_to_anchor=bbox_to_anchor, loc=loc, borderaxespad=borderaxespad,
                         ncol=ncol)
        if legend_line_size is not None:
            for legobj in leg.legendHandles:
                legobj.set_linewidth(legend_line_size)
        if legend_dot_size is not None:
            for i in range(len(unique_lines)):
                leg.legendHandles[i]._sizes = [legend_dot_size]
        return fig, axs