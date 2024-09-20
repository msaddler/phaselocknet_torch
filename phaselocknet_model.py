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
