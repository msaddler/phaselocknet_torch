import argparse
import os
import time

import pandas as pd
import torch
import torchaudio

import phaselocknet_model
import util


def evaluate(
    model,
    dataset,
    dir_model,
    fn_eval_output="eval_output.csv",
    key_input="signal",
    key_input_sr="sr",
    sr=50000,
    overwrite=False,
    write_prob=1,
    batch_size=32,
    num_steps_per_display=20,
    device=None,
    num_workers=4,
):
    """ """
    if not key_input == "signal":
        msg = "[evaluate] function is only implemented for audio-input models"
        raise NotImplementedError(msg)
    fn_eval_output = os.path.join(dir_model, fn_eval_output)
    if os.path.exists(fn_eval_output):
        if overwrite:
            print(f"[evaluate] Overwriting pre-existing {fn_eval_output=}")
            os.remove(fn_eval_output)
        else:
            print(f"[complete] {fn_eval_output=} already exists!")
            return
    if os.path.exists(fn_eval_output + "~"):
        print(f"[evaluate] Deleting pre-existing tempfile: {fn_eval_output}~")
        os.remove(fn_eval_output + "~")
    if write_prob:
        df_prob = []
        fn_eval_prob = fn_eval_output.replace(".csv", "_prob.gz")
        if os.path.exists(fn_eval_prob):
            print(f"[evaluate] Deleted pre-existing {fn_eval_prob=}")
            os.remove(fn_eval_prob)
    print(f"[evaluate] {dir_model=}")
    print(f"[evaluate] {fn_eval_output=}")
    print(f"[evaluate] {write_prob=}")
    print(f"[evaluate] dataset: {len(dataset)} examples")
    print(f"[evaluate] dataset: {int(len(dataset) / batch_size)} batches")
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=False,
    )
    model.train(False)
    model.to(device)
    with torch.no_grad():
        t0 = time.time()
        for step, example in enumerate(dataloader):
            x = example.pop(key_input).type(torch.float32)
            sr_src = example[key_input_sr][0].item()
            if step == 0:
                resampler = torchaudio.transforms.Resample(
                    orig_freq=sr_src,
                    new_freq=sr,
                )
                print(f"[evaluate] resampling audio from {sr_src} to {sr} Hz")
            if x.ndim > 2:
                x = torch.stack(
                    [resampler(x[..., channel]) for channel in range(x.shape[-1])],
                    axis=-1,
                )
            else:
                x = resampler(x)
                if len(model.input_shape) > 2:
                    x = torch.stack([x for _ in range(model.input_shape[-1])], axis=-1)
            x = util.pad_or_trim_to_len(x, n=model.input_shape[1], dim=1)
            assert list(x.shape[1:]) == list(model.input_shape[1:])
            task_logits = model(x.to(device))
            task_preds = {
                k + ".pred": torch.argmax(v, dim=1) for k, v in task_logits.items()
            }
            example.update(task_preds)
            if write_prob:
                task_prob = {
                    k + ".prob": torch.nn.functional.softmax(v, dim=1)
                    for k, v in task_logits.items()
                }
                df_prob.append(
                    pd.DataFrame(
                        {
                            k: list(v.detach().cpu().numpy())
                            for k, v in sorted(task_prob.items())
                        }
                    )
                )
            example = {
                k: list(v.detach().cpu().numpy()) for k, v in sorted(example.items())
            }
            df = pd.DataFrame(example)
            df.to_csv(
                fn_eval_output + "~",
                mode="a",
                header=not os.path.exists(fn_eval_output + "~"),
                index=False,
            )
            if step % num_steps_per_display == num_steps_per_display - 1:
                display_str = util.get_model_progress_display_str(
                    epoch=0,
                    step=step + 1,
                    num_steps=step + 1,
                    t0=t0,
                    mem=True,
                )
                print(display_str)
    if write_prob:
        df_prob = pd.concat(df_prob).reset_index(drop=True)
        df_prob.to_pickle(fn_eval_prob + "~", compression="gzip")
        os.rename(fn_eval_prob + "~", fn_eval_prob)
        print(f"[complete] {fn_eval_prob=}")
    os.rename(fn_eval_output + "~", fn_eval_output)
    print(f"[complete] {fn_eval_output=}")


if __name__ == "__main__":
    """ """
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--dir_model", type=str, default=None)
    parser.add_argument("-e", "--regex_eval", type=str, default=None)
    parser.add_argument("-fe", "--fn_eval_output", type=str, default=None)
    parser.add_argument("-wp", "--write_prob", type=int, default=1)
    parser.add_argument("-o", "--overwrite", type=int, default=0)
    parser.add_argument("-b", "--batch_size", type=int, default=32)
    parser.add_argument("-n", "--num_workers", type=int, default=4)
    args = parser.parse_args()
    dataset = util.HDF5Dataset(regex_filenames=args.regex_eval)
    model, config_model = phaselocknet_model.get_model(
        dir_model=args.dir_model,
        fn_config="config.json",
        fn_arch="arch.json",
    )
    util.load_model_checkpoint(
        model=model.perceptual_model,
        dir_model=args.dir_model,
        step=None,
        fn_ckpt="ckpt_BEST.pt",
    )
    evaluate(
        model=model,
        dataset=dataset,
        dir_model=args.dir_model,
        fn_eval_output=args.fn_eval_output,
        key_input=config_model["kwargs_optimize"]["key_inputs"],
        key_input_sr="sr",
        sr=config_model["kwargs_cochlea"]["sr_input"],
        overwrite=args.overwrite,
        write_prob=args.write_prob,
        batch_size=args.batch_size,
        num_steps_per_display=20,
        device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
        num_workers=args.num_workers,
    )
