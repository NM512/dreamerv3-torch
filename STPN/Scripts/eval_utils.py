import torch
from torch import no_grad

def simple_eval(
        model,
        test_dataloader,
        device,
        data_args,
        init_states=None,
        energy=False,
):
    """
    @param model: trained network
    @param test_dataloader: dataloader with evaluation data. Only tested with TensorDataset
    @param device: Torch object for CPU/GPU device
    @param data_args: config arguments for running evaluation
    @param init_states: optionally provide initial states for model
    @param energy: evaluate energy
    @return: evaluation accuracy (and energy consumption if energy==True)

    """
    # only tested for dataloader which uses a TensorDataset
    test_label_set_shape = test_dataloader.dataset.tensors[1].size()
    n_test_elements = 1
    n_test_elements = int([n_test_elements * test_label_set_shape[dim] for dim in range(len(test_label_set_shape))][0])

    acc = 0
    skipped_batches_size = 0
    states = init_states if init_states is None or len(init_states) == 1 else init_states
    energy_consumption_per_forward = []
    model.eval()
    with no_grad():
        for i, (datapoints_, labels_) in enumerate(test_dataloader):
            # ignore last element in batch. TODO: reduce states in order to use last not full batch
            if len(labels_) < data_args['batch_size']:
                print(f"skipping batch with size {len(labels_)}")
                skipped_batches_size += len(labels_)
                continue
            # send data to device
            datapoints_, labels_ = datapoints_.to(device), labels_.to(device)

            # infer, optionally with energy consumption calculation
            if energy is True:
                preds, states, energy_per_forward = model.forward_energy(datapoints_, states)
                # energy = (seqs, batch_size, seq_len, hidden) -> (total_seqs, seq_len)
                energy_consumption_per_forward.append(energy_per_forward)
            else:
                preds, states = model(datapoints_, states)

            acc += (preds.argmax(dim=1) == labels_).float().sum().cpu().item()
            states = None  # stateless. TODO: allow stateful inference
    n_test_elements -= skipped_batches_size
    acc /= n_test_elements
    print(f"Model achieved {acc} test accuracy")

    if energy is True:
        return acc, torch.cat(energy_consumption_per_forward, dim=0) # (total_seqs, seq_len)
    else:
        return acc