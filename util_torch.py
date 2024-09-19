import glob
import os
import numpy as np
import torch


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
