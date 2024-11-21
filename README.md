## phaselocknet_torch

Minimal and in-progress PyTorch implementation of the models from ["Models optimized for real-world tasks reveal the task-dependent necessity of precise temporal coding in hearing"](https://doi.org/10.1101/2024.04.21.590435) by Mark R. Saddler and Josh H. McDermott (in press, Nature Communications). The primary repository accompanying this work ([phaselocknet](https://github.com/msaddler/phaselocknet)) implements the models in TensorFlow and also includes code to run analyses and generate figures.

## Dependencies

This is a repository of Python (3.11.7) code. A complete list of Python dependencies is contained in [`requirements.txt`](requirements.txt). The models were developed in TensorFlow (`tensorflow-2.13.0`) on machines running CentOS Linux 7 with NVidia A100 GPUs. Here, we provide minimal code to load and run the models in PyTorch (`torch-2.2.1`).

## Model weights

Trained weights for each model configuration are too large to include here, but can be downloaded from our [Google Drive](https://drive.google.com/drive/folders/1YgC7x6Ot84XZInlSyHK-9NQ0jhhGUS2z?usp=share_link). Each model should have its own directory, containing `config.json` (cochlear model parameters and optimization hyperparameters), `arch.json` (artificial neural network architecture), and `ckpt_BEST` (TensorFlow checkpoint file with trained weights). Weights are provided as TensorFlow checkpoint files because models were trained in TensorFlow. [Here](util.py), we provide a function to load trained weights from the TensorFlow checkpoint files into equivalent PyTorch model objects.

## Contents

The [`DEMO.ipynb`](DEMO.ipynb) notebook shows how to load and run our TensorFlow-trained models in PyTorch. This notebook relies on:
- [`phaselocknet_model.py`](phaselocknet_model.py): High-level class for combined auditory nerve + deep neural network model
- [`peripheral_model.sh`](peripheral_model.py): PyTorch implementation of simplified auditory nerve model
- [`perceptual_model.py`](perceptual_model.py): PyTorch implementation of deep neural network model
- [`util.py`](util.py): Helper functions (e.g., for loading TensorFlow checkpoint weights into a PyTorch model obect)

Code to evaluate the PyTorch models at scale is provided in:
- [`phaselocknet_evaluate.py`](phaselocknet_evaluate.py): Python script to run model and write outputs to files
- [`phaselocknet_evaluate_job.sh`](phaselocknet_evaluate_job.sh): SLURM / bash script to evaluate model on stimuli from all experiments

## Contact

Note that all results in the paper were generated using the TensorFlow implementation of these models (https://github.com/msaddler/phaselocknet). The TensorFlow and PyTorch implementations are not guaranteed to produce identical behavior. Preliminary checks indicate that when stochastic model elements (auditory nerve spike sampling and random excerpting of auditory nerve representations) are removed, the two implementations produce softmax output distributions within numerical precision limits of one another. Please let the authors know if you find any discrepant behavior.

Mark R. Saddler (msaddler@mit.edu / marksa@dtu.dk)
