import glob
import json
import os
import re
import resource
import time

import h5py
import numpy as np
import torch


class NumpyEncoder(json.JSONEncoder):
    """
    Helper class to JSON serialize numpy arrays.
    """

    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        return json.JSONEncoder.default(self, obj)


def get_hdf5_dataset_key_list(f_input):
    """
    Walks hdf5 file and returns list of all dataset keys.

    Args
    ----
    f_input (str or h5py.File): hdf5 filename or file object

    Returns
    -------
    hdf5_dataset_key_list (list): list of paths to datasets in f_input
    """
    if isinstance(f_input, str):
        f = h5py.File(f_input, "r")
    else:
        f = f_input
    hdf5_dataset_key_list = []

    def get_dataset_keys(name, node):
        if isinstance(node, h5py.Dataset):
            hdf5_dataset_key_list.append(name)

    f.visititems(get_dataset_keys)
    if isinstance(f_input, str):
        f.close()
    return hdf5_dataset_key_list


def get_model_progress_display_str(
    epoch=None,
    step=None,
    num_steps=None,
    t0=None,
    mem=True,
    loss=None,
    task_loss={},
    task_acc={},
    single_line=True,
):
    """
    Returns a string to print model progress.

    Args
    ----
    epoch (int): current training epoch
    step (int): current training step
    num_steps (int): total steps taken since t0
    t0 (float): start time in seconds
    mem (bool): if True, include total memory usage
    loss (float): current loss
    task_loss (dict): current task-specific losses
    task_acc (dict): current task-specific accuracies
    single_line (bool): if True, remove linebreaks

    Returns
    -------
    display_str (str): formatted string to print
    """
    display_str = ""
    if (epoch is not None) and (step is not None):
        display_str += "step {:02d}_{:06d} | ".format(epoch, step)
    if (num_steps is not None) and (t0 is not None):
        display_str += "{:.4f} s/step | ".format((time.time() - t0) / num_steps)
    if mem:
        display_str += "mem: {:06.3f} GB | ".format(
            resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024 / 1024
        )
    if loss is not None:
        display_str += "loss: {:.4f} | ".format(loss)
    if task_loss:
        if isinstance(task_loss, dict):
            display_str += "\n|___ task loss | "
            for k, v in task_loss.items():
                display_str += "{}: {:.4f} ".format(
                    k.replace("label_", "").replace("_int", ""), v
                )
            display_str += "| "
        else:
            display_str += "task_loss: {:.4f} | ".format(task_loss)
    if task_acc:
        if isinstance(task_acc, dict):
            display_str += "\n|___ task accs | "
            for k, v in task_acc.items():
                display_str += "{}: {:.4f} ".format(
                    k.replace("label_", "").replace("_int", ""), v
                )
            display_str += "| "
        else:
            display_str += "task_acc: {:.4f} | ".format(task_acc)
    if single_line:
        display_str = display_str.replace("\n|___ ", "")
    return display_str


def rms(x, dim=None, keepdim=False):
    """
    Compute root-mean-squared amplitude of `x` along axis `dim`
    """
    out = torch.sqrt(
        torch.mean(
            torch.square(x),
            dim=dim,
            keepdim=keepdim,
        ),
    )
    return out


def set_dbspl(x, dbspl, mean_subtract=True, dim=None, keepdim=False):
    """
    Set root-mean-squared amplitude of `x` along axis `dim`
    to the sound pressure level `dbspl` (in dB re 20e-6 Pa)
    """
    if mean_subtract:
        x = x - torch.mean(x, dim=dim, keepdim=keepdim)
    rms_output = 20e-6 * torch.pow(10, dbspl / 20)
    rms_input = rms(x, dim=dim, keepdim=keepdim)
    if torch.is_nonzero(rms_input):
        x = rms_output * (x / rms_input)
    return x


def pad_or_trim_to_len(x, n, dim=-1, kwargs_pad={}):
    """
    Symmetrically pad or trim `x` to length `n` along axis `dim`
    """
    n_orig = int(x.shape[dim])
    if n_orig < n:
        n0 = (n - n_orig) // 2
        n1 = (n - n_orig) - n0
        pad = []
        for _ in range(x.ndim):
            pad.extend([n0, n1] if _ == dim else [0, 0])
        x = torch.nn.functional.pad(x, pad, **kwargs_pad)
    if n_orig > n:
        n0 = (n_orig - n) // 2
        ind = [slice(None)] * x.ndim
        ind[dim] = slice(n0, n0 + n)
        x = x[ind]
    return x


def load_tf_model_checkpoint(model, filename):
    """
    Load parameters from a tensorflow checkpoint file into a torch model
    (includes only learned parameters; excludes optimizer hyperparameters)
    """
    import tensorflow.train

    # Get list of layers with named parameters from torch model
    list_layer_name = []
    for n, p in model.named_parameters():
        name = n.replace(".bias", "").replace(".weight", "")
        if name not in list_layer_name:
            list_layer_name.append(name)
    # Read the tensorflow checkpoint file into `tf_state_dict`
    tf_state_dict = {}
    reader = tensorflow.train.load_checkpoint(filename)
    for k in reader.get_variable_to_shape_map():
        if ("layer_with_weights" in k) and ("OPTIMIZER_SLOT" not in k):
            tf_state_dict[
                k.replace("/.ATTRIBUTES/VARIABLE_VALUE", "")
            ] = reader.get_tensor(k)
    # Map parameters from `tf_state_dict` to `torch_state_dict`
    torch_state_dict = {}
    for k, v in sorted(tf_state_dict.items()):
        layer_index = int(k[k.find("-") + 1 : k.find("/")])
        layer_name = list_layer_name[layer_index]
        if ("/kernel" in k) or ("/gamma" in k):
            name = "weight"
        elif ("/bias" in k) or ("/beta" in k):
            name = "bias"
        elif "/moving_variance" in k:
            name = "running_var"
        elif "/moving_mean" in k:
            name = "running_mean"
        else:
            raise ValueError(f"Unrecognized tensorflow parameter: `{k}`")
        name = "{}.{}".format(layer_name, name)
        if v.ndim == 2:
            torch_state_dict[name] = torch.tensor(np.transpose(v, [1, 0]))
        elif v.ndim == 4:
            torch_state_dict[name] = torch.tensor(np.transpose(v, [3, 2, 0, 1]))
        else:
            torch_state_dict[name] = torch.tensor(v)
    # Ensure all torch model parameters are accounted for and shapes align
    for n, p in model.named_parameters():
        msg0 = f"torch model parameter `{n}` not found in checkpoint"
        msg1 = f"shape mismatch for `{n}`: {p.shape} | {torch_state_dict[n].shape}"
        assert n in torch_state_dict, msg0
        assert np.array_equal(torch_state_dict[n].shape, p.shape), msg1
    # Load parameters into torch model
    out = model.load_state_dict(torch_state_dict, strict=False, assign=False)
    assert not out.unexpected_keys, out
    if out.missing_keys:
        print(f"[load_tf_model_checkpoint] missing_keys ({filename})")
        for k in out.missing_keys:
            print(f"|__ {k}")
    print(f"[load_tf_model_checkpoint] {filename}")
    return filename


def save_model_checkpoint(
    model,
    dir_model=None,
    step=None,
    fn_ckpt="ckpt_BEST.pt",
    fn_ckpt_step="ckpt_{:04d}.pt",
    **kwargs,
):
    """ """
    filename = fn_ckpt
    if step is not None:
        filename = fn_ckpt_step.format(step)
    if dir_model is not None:
        filename = os.path.join(dir_model, filename)
    torch.save(model.state_dict(), filename + "~", **kwargs)
    os.rename(filename + "~", filename)
    print(f"[save_model_checkpoint] {filename}")
    return filename


def load_model_checkpoint(
    model,
    dir_model=None,
    step=None,
    fn_ckpt="ckpt_BEST.pt",
    fn_ckpt_step="ckpt_{:04d}.pt",
    **kwargs,
):
    """ """
    filename = fn_ckpt
    if step is not None:
        filename = fn_ckpt_step
    if dir_model is not None:
        filename = os.path.join(dir_model, filename)
    if (step is not None) and (step >= 0):
        # Load checkpoint specified by step
        filename = filename.format(step)
    elif step is not None:
        # Load recent checkpoint if step < 0
        unformatted = filename.replace(
            filename[filename.find("{") + 1 : filename.find("}")],
            "",
        )
        list_filename = []
        list_step = []
        for filename in glob.glob(unformatted.format("*")):
            output = re.findall(r"\d+", os.path.basename(filename))
            if len(output) == 1:
                list_filename.append(filename)
                list_step.append(int(output[0]))
        if len(list_filename) == 0:
            print("[load_model_checkpoint] No prior checkpoint found")
            return 0
        list_filename = [list_filename[_] for _ in np.argsort(list_step)]
        filename = list_filename[step]
    state_dict = torch.load(filename, **kwargs)
    model.load_state_dict(state_dict, strict=True, assign=False)
    print(f"[load_model_checkpoint] {filename}")
    if (step is not None) and (step < 0):
        return list_step[step]
    return filename


def set_trainable(
    model,
    trainable=False,
    trainable_layers=None,
    trainable_batchnorm=None,
    verbose=True,
):
    """ """
    if not trainable:
        model.train(trainable)
        trainable_params = []
    elif trainable_layers is None:
        model.train(trainable)
        trainable_params = list(model.parameters())
    else:
        trainable_layers_names = []
        trainable_params_names = []
        trainable_params = []
        model.train(False)
        if verbose:
            print(f"[set_trainable] {trainable_layers=}")
        for m_name, m in model.named_modules():
            msg = f"invalid trainable_layers ({m_name} -> multiple matches)"
            for pattern in trainable_layers:
                if pattern in m_name:
                    assert m_name not in trainable_layers_names, msg
                    trainable_layers_names.append(m_name)
                    m.train(trainable)
                    if verbose:
                        print(f"{m_name} ('{pattern}') -> {m.training}")
                    for p_basename, p in m.named_parameters():
                        p_name = f"{m_name}.{p_basename}"
                        assert p_name not in trainable_params_names, msg
                        trainable_params_names.append(p_name)
                        trainable_params.append(p)
                        if verbose:
                            print(f"|__ {p_name} {p.shape}")
            if trainable_batchnorm is not None:
                if "batchnorm" in str(type(m)).lower():
                    m.train(trainable_batchnorm)
                    if verbose:
                        print(f"{m_name} ({trainable_batchnorm=}) -> {m.training}")
        if verbose:
            print(f"[set_trainable] {len(trainable_layers_names)=}")
    if verbose:
        print(f"[set_trainable] {trainable} -> {len(trainable_params)=}")
    return trainable_params
