import copy
import json
import os

import torch

import perceptual_model
import peripheral_model


class Model(torch.nn.Module):
    def __init__(
        self,
        config_model={},
        architecture=[],
        input_shape=[2, 60000, 2],
        config_random_slice={"size": [50, 10000], "buffer": [0, 500]},
    ):
        """
        Construct torch model graph from `config_model` and `architecture`.
        These should be read from "config.json" and "arch.json" files within
        the model directory.

        Args
        ----
        config_model (dict): configures cochlear model and hyperparameters
        architecture (list): specifies deep neural network model layers
        input_shape (list): tensor input shape for model, should be one of:
            [batch=2, time=40000] for simplified spkr / word model
            [batch=2, time=60000, ear=2] for simplified localization model
            [batch=2, spont=3, freq=50, time=20000] for spkr/word model
            [batch=2, spont=3, freq=50, time=12000, ear=2] for loc model
        config_random_slice (dict): specifies how nervegram should be trimmed
            spkr / word models: {"size": [50, 20000], "buffer": [0, 0]}
            localization models: {"size": [50, 10000], "buffer": [0, 500]}
        """
        super().__init__()
        self.input_shape = input_shape
        self.config_random_slice = config_random_slice
        kwargs_peripheral_model = {
            "sr_input": config_model["kwargs_cochlea"].get("sr_input", None),
            "sr_output": config_model["kwargs_cochlea"].get("sr_output", None),
            "config_cochlear_filterbank": config_model["kwargs_cochlea"].get(
                "config_filterbank", {}
            ),
            "config_ihc_transduction": config_model["kwargs_cochlea"].get(
                "config_subband_processing", {}
            ),
            "config_ihc_lowpass_filter": config_model["kwargs_cochlea"].get(
                "kwargs_fir_lowpass_filter_output", {}
            ),
            "config_anf_rate_level": config_model["kwargs_cochlea"].get(
                "kwargs_sigmoid_rate_level_function", {}
            ),
            "config_anf_spike_generator": config_model["kwargs_cochlea"].get(
                "kwargs_spike_generator_binomial", {}
            ),
            "config_random_slice": config_random_slice,
        }
        assert kwargs_peripheral_model["config_ihc_lowpass_filter"].pop(
            "ihc_filter", True
        )
        self.peripheral_model = peripheral_model.PeripheralModel(
            **kwargs_peripheral_model,
        )
        self.perceptual_model = perceptual_model.PerceptualModel(
            architecture=architecture,
            input_shape=self.peripheral_model(torch.zeros(self.input_shape)).shape,
            heads=config_model["n_classes_dict"],
        )

    def forward(self, x):
        """ """
        return self.perceptual_model(self.peripheral_model(x))


def get_model(
    dir_model,
    fn_config="config.json",
    fn_arch="arch.json",
):
    """
    Helper function returns a torch model object from a directory
    containing config and architecture files.
    """
    # Load config_model and architecture
    with open(os.path.join(dir_model, fn_config), "r") as f:
        config_model = json.load(f)
    with open(os.path.join(os.path.join(dir_model, fn_arch)), "r") as f:
        architecture = json.load(f)
    # Determine input_shape and config_random_slice
    if "sound_localization" in dir_model:
        if config_model["kwargs_optimize"]["key_inputs"] == "signal":
            input_shape = [2, 60000, 2]
            config_random_slice = {"size": [50, 10000], "buffer": [0, 1000]}
        elif config_model["kwargs_optimize"]["key_inputs"] == "nervegram_meanrates":
            input_shape = [2, 3, 50, 12000, 2]
            config_random_slice = {"size": [50, 10000], "buffer": [0, 500]}
        else:
            raise ValueError(f"[get_model] could not infer input for {dir_model=}")
    else:
        if config_model["kwargs_optimize"]["key_inputs"] == "signal":
            input_shape = [2, 40000]
            config_random_slice = {"size": [50, 20000], "buffer": [0, 0]}
        elif config_model["kwargs_optimize"]["key_inputs"] == "nervegram_meanrates":
            input_shape = [2, 3, 50, 20000]
            config_random_slice = {"size": [50, 20000], "buffer": [0, 0]}
        else:
            raise ValueError(f"[get_model] could not infer input for {dir_model=}")
    print(f"[get_model] {dir_model=}")
    print(f"[get_model] |__ {input_shape=}")
    print(f"[get_model] |__ {config_random_slice=}")
    # Construct and return model object
    model = Model(
        config_model=copy.deepcopy(config_model),
        architecture=copy.deepcopy(architecture),
        input_shape=input_shape,
        config_random_slice=copy.deepcopy(config_random_slice),
    )
    return model, config_model
