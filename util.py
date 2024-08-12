import glob
import json
import os
import resource
import time

import numpy as np
import pandas as pd
import scipy


def logistic_function(x, x0, k, chance=0.5):
    """ """
    return ((1 - chance) / (1 + np.exp(-k * (x - x0)))) + chance


def logistic_function_inv(y, x0, k, chance=0.5):
    """ """
    return (np.log(((1 - chance) / (y - chance)) - 1) / -k) + x0


def fit_logistic_function(x, y, method="trf", chance=0.5, p0=None, **kwargs):
    """ """
    if p0 is None:
        p0 = (x[np.argmin(np.abs(np.cumsum(y) / np.sum(y) - chance))], 1)
    popt, pcov = scipy.optimize.curve_fit(
        lambda _, x0, k: logistic_function(_, x0, k, chance=chance),
        xdata=x,
        ydata=y,
        p0=p0,
        method=method,
        **kwargs,
    )
    return np.squeeze(popt), np.squeeze(pcov)


class PsychoacousticExperiment:
    def __init__(
        self,
        read=True,
        write=True,
        overwrite=False,
        verbose=False,
        **kwargs,
    ):
        """
        Base class for running a psychoacoustic experiment on a
        model (converts evaluation output file to a results file)
        """
        self.read = read
        self.write = write
        self.overwrite = overwrite
        self.verbose = verbose
        for k, v in kwargs.items():
            setattr(self, k, v)

    def __call__(self, regex_dir_model):
        """ """
        for key in ["basename_eval", "basename_results", "run"]:
            msg = f"experiment subclass is missing attribute: {key}"
            assert hasattr(self, key), msg
        if isinstance(regex_dir_model, list):
            list_dir_model = regex_dir_model
        elif isinstance(regex_dir_model, str):
            list_dir_model = glob.glob(regex_dir_model)
        else:
            raise ValueError(f"unrecognized format: {regex_dir_model=}")
        list_df = []
        for dir_model in list_dir_model:
            fn_eval = os.path.join(dir_model, self.basename_eval)
            fn_results = os.path.join(dir_model, self.basename_results)
            df = None
            if self.read:
                if (os.path.exists(fn_results)) and (not self.overwrite):
                    df = pd.read_csv(fn_results)
                    if self.verbose:
                        print(f"[experiment] READ {fn_results=}")
            if df is None:
                df = self.run(fn_eval)
                if self.verbose:
                    print(f"[experiment] READ {fn_eval=}")
            if self.write:
                if (not os.path.exists(fn_results)) or (self.overwrite):
                    df.to_csv(fn_results, index=False)
                    if self.verbose:
                        print(f"[experiment] WROTE {fn_results=}")
            list_df.append(df.assign(dir_model=dir_model, fn_eval=fn_eval))
        return pd.concat(list_df)

    def __repr__(self):
        """ """
        d = {
            a: getattr(self, a)
            for a in dir(self)
            if (not a.startswith("__")) and (isinstance(getattr(self, a), str))
        }
        return json.dumps(d)


class ExperimentHeinz2001FrequencyDiscrimination(PsychoacousticExperiment):
    def __init__(
        self,
        basename_eval="eval.csv",
        basename_results="results.csv",
        **kwargs,
    ):
        """
        PsychoacousticExperiment class for pure tone frequency discrimination
        experiment from Heinz et al. (2001, Neural Computation)
        """
        super().__init__(
            basename_eval=basename_eval,
            basename_results=basename_results,
            **kwargs,
        )

    def octave_to_weber_fraction(self, interval):
        """ """
        return np.power(2, interval) - 1.0

    def run(self, fn_eval):
        """ """
        df = pd.concat([pd.read_csv(_) for _ in glob.glob(fn_eval)])
        df["correct"] = df["label"] == (df["logits"] > 0)
        df["interval"] = np.abs(df["interval"])
        df = df.groupby(["f0", "interval"]).agg({"correct": "mean"}).reset_index()
        df = df.sort_values(by=["f0", "interval"])
        df = df.groupby(["f0"]).agg({"interval": list, "correct": list}).reset_index()
        kw = {"method": "dogbox", "max_nfev": 10000}
        df["popt"] = df.apply(
            lambda _: fit_logistic_function(np.log(_["interval"]), _["correct"], **kw)[
                0
            ],
            axis=1,
        )
        df["log_threshold"] = df["popt"].map(lambda _: logistic_function_inv(0.75, *_))
        df["threshold"] = np.exp(df["log_threshold"])
        df["weber_fraction"] = self.octave_to_weber_fraction(df["threshold"])
        df = df.drop(columns=["interval", "correct", "popt"])
        return df


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


def format_axes(
    ax,
    str_title=None,
    str_xlabel=None,
    str_ylabel=None,
    fontsize_title=12,
    fontsize_labels=12,
    fontsize_ticks=12,
    fontweight_title=None,
    fontweight_labels=None,
    xscale="linear",
    yscale="linear",
    xlimits=None,
    ylimits=None,
    xticks=None,
    yticks=None,
    xticks_minor=None,
    yticks_minor=None,
    xticklabels=None,
    yticklabels=None,
    spines_to_hide=[],
    major_tick_params_kwargs_update={},
    minor_tick_params_kwargs_update={},
):
    """
    Helper function for setting axes-related formatting parameters.
    """
    ax.set_title(str_title, fontsize=fontsize_title, fontweight=fontweight_title)
    ax.set_xlabel(str_xlabel, fontsize=fontsize_labels, fontweight=fontweight_labels)
    ax.set_ylabel(str_ylabel, fontsize=fontsize_labels, fontweight=fontweight_labels)
    ax.set_xscale(xscale)
    ax.set_yscale(yscale)
    ax.set_xlim(xlimits)
    ax.set_ylim(ylimits)
    if xticks_minor is not None:
        ax.set_xticks(xticks_minor, minor=True)
    if yticks_minor is not None:
        ax.set_yticks(yticks_minor, minor=True)
    if xticks is not None:
        ax.set_xticks(xticks, minor=False)
    if yticks is not None:
        ax.set_yticks(yticks, minor=False)
    if xticklabels is not None:
        ax.set_xticklabels([], minor=True)
        ax.set_xticklabels(xticklabels, minor=False)
    if yticklabels is not None:
        ax.set_yticklabels([], minor=True)
        ax.set_yticklabels(yticklabels, minor=False)
    major_tick_params_kwargs = {
        "axis": "both",
        "which": "major",
        "labelsize": fontsize_ticks,
        "length": fontsize_ticks / 2,
        "direction": "out",
    }
    major_tick_params_kwargs.update(major_tick_params_kwargs_update)
    ax.tick_params(**major_tick_params_kwargs)
    minor_tick_params_kwargs = {
        "axis": "both",
        "which": "minor",
        "labelsize": fontsize_ticks,
        "length": fontsize_ticks / 4,
        "direction": "out",
    }
    minor_tick_params_kwargs.update(minor_tick_params_kwargs_update)
    ax.tick_params(**minor_tick_params_kwargs)
    for spine_key in spines_to_hide:
        ax.spines[spine_key].set_visible(False)
    return ax
