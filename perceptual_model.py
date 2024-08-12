import collections

import numpy as np
import torch
import torchvision


class PerceptualModel(torch.nn.Module):
    def __init__(
        self,
        architecture=[],
        input_shape=[2, 6, 50, 20000],
        heads={"label": 1},
        device=None,
    ):
        """
        Construct torch DNN model graph from architecture description
        """
        super().__init__()
        self.input_shape = input_shape
        self.body = collections.OrderedDict()
        self.heads = heads
        self.head = {k: collections.OrderedDict() for k in self.heads}
        self.construct_model(architecture, device)

    def get_layer_from_description(self, d, x):
        """ """
        layer_type = d["layer_type"].lower()
        if "conv" in layer_type:
            layer = CustomPaddedConv2d(
                in_channels=x.shape[1],
                out_channels=d["args"]["filters"],
                kernel_size=d["args"]["kernel_size"],
                stride=d["args"]["strides"],
                padding=d["args"]["padding"],
                dilation=d["args"].get("dilation", 1),
                groups=d["args"].get("groups", 1),
                bias=d["args"].get("bias", True),
                padding_mode="zeros",
            )
        elif "dense" in layer_type:
            layer = torch.nn.Linear(
                in_features=x.shape[1],
                out_features=d["args"]["units"],
                bias=d["args"].get("use_bias", True),
            )
        elif "dropout" in layer_type:
            layer = torch.nn.Dropout(p=d["args"]["rate"], inplace=False)
        elif "flatten" in layer_type:
            layer = CustomFlatten(
                start_dim=1,
                end_dim=-1,
                permute_dims=(0, 2, 3, 1),
            )
        elif "maxpool" in layer_type:
            layer = torch.nn.MaxPool2d(
                kernel_size=d["args"]["pool_size"],
                stride=d["args"]["strides"],
                padding=0,
            )
        elif "hpool" in layer_type:
            layer = HanningPooling(
                stride=d["args"]["strides"],
                kernel_size=d["args"]["pool_size"],
                padding=d["args"]["padding"],
                sqrt_window=d["args"].get("sqrt_window", False),
                normalize=d["args"].get("normalize", False),
            )
        elif "batchnorm" in layer_type.replace("_", ""):
            layer = torch.nn.SyncBatchNorm(
                num_features=x.shape[1] if d["args"].get("axis", -1) == -1 else None,
                eps=d["args"].get("eps", 1e-05),
                momentum=d["args"].get("momentum", 0.1),
                affine=d["args"].get("scale", True),
            )
        elif "layernorm" in layer_type.replace("_", ""):
            layer = CustomNorm(
                input_shape=x.shape,
                dim_affine=1 if d["args"].get("scale", True) else None,
                dim_norm=1 if d["args"]["axis"] == -1 else d["args"]["axis"],
                correction=1,
                eps=d["args"].get("eps", 1e-05),
            )
        elif "permute" in layer_type:
            layer = Permute(dims=d["args"]["dims"])
        elif "reshape" in layer_type:
            layer = Reshape(shape=d["args"]["shape"])
        elif "unsqueeze" in layer_type:
            layer = Unsqueeze(dim=d["args"]["dim"])
        elif "randomslice" in layer_type.replace("_", ""):
            layer = RandomSlice(
                size=d["args"]["size"],
                buffer=d["args"]["buffer"],
            )
        elif "relu" in layer_type:
            layer = torch.nn.ReLU(inplace=False)
        elif ("branch" in layer_type) or ("fc_top" in layer_type):
            layer = None
        else:
            raise ValueError(f"{layer_type=} not recognized")
        return layer

    def construct_model(self, architecture, device):
        """ """
        x = torch.zeros(self.input_shape).to(device)
        is_body_layer = True
        for d in architecture:
            if is_body_layer:
                layer = self.get_layer_from_description(d, x)
            else:
                layer = {
                    k: self.get_layer_from_description(d, x[k]) for k in self.heads
                }
            if (layer is None) or (
                isinstance(layer, dict) and list(layer.values())[0] is None
            ):
                is_body_layer = False
                if not isinstance(x, dict):
                    x = {k: torch.clone(x) for k in self.heads}
            else:
                if is_body_layer:
                    self.body[d["args"]["name"]] = layer
                    x = layer.to(x.device)(x)
                else:
                    for k in self.heads:
                        self.head[k][d["args"]["name"]] = layer[k]
                        x[k] = layer[k].to(x[k].device)(x[k])
        self.body = torch.nn.Sequential(self.body)
        if not isinstance(x, dict):
            x = {k: torch.clone(x) for k in self.heads}
        for k in self.heads:
            self.head[k]["fc_output"] = torch.nn.Linear(
                in_features=x[k].shape[1],
                out_features=self.heads[k],
                bias=True,
            )
            self.head[k] = torch.nn.Sequential(self.head[k])
        self.head = torch.nn.ModuleDict(self.head)

    def forward(self, x):
        """ """
        x = self.body(x)
        logits = {k: self.head[k](x) for k in self.heads}
        return logits


def calculate_same_pad(input_dim, kernel_dim, stride):
    """ """
    pad = (np.ceil(input_dim / stride) - 1) * stride + (kernel_dim - 1) + 1 - input_dim
    return int(max(pad, 0))


def custom_conv_pad(x, pad, weight=None, stride=None, **kwargs):
    """ """
    msg = f"Expected input shape [batch, channel, freq, time]: received {x.shape=}"
    assert x.ndim == 4, msg
    msg = f"Expected tuple or integers or a string: received {pad=}"
    assert isinstance(pad, (tuple, str)), msg
    if isinstance(pad, str):
        if pad.lower() == "same":
            pad_f = calculate_same_pad(x.shape[-2], weight.shape[-2], stride[-2])
            pad_t = calculate_same_pad(x.shape[-1], weight.shape[-1], stride[-1])
        elif pad.lower() in ["same_freq", "valid_time"]:
            pad_f = calculate_same_pad(x.shape[-2], weight.shape[-2], stride[-2])
            pad_t = 0
        elif pad.lower() in ["same_time", "valid_freq"]:
            pad_f = 0
            pad_t = calculate_same_pad(x.shape[-1], weight.shape[-1], stride[-1])
        elif pad.lower() == "valid":
            pad_f = 0
            pad_t = 0
        else:
            raise ValueError(f"Mode `{pad=}` is not recognized")
        pad = (pad_t // 2, pad_t - pad_t // 2, pad_f // 2, pad_f - pad_f // 2)
    return torch.nn.functional.pad(x, pad, **kwargs)


class ChannelwiseConv2d(torch.nn.Module):
    def __init__(self, kernel, pad=(0, 0), stride=(1, 1), dtype=torch.float32):
        """ """
        super().__init__()
        assert kernel.ndim == 2, "Expected kernel with shape [freq, time]"
        self.register_buffer(
            "weight",
            torch.tensor(kernel[None, None, :, :], dtype=dtype),
            persistent=True,
        )
        self.pad = pad
        self.stride = stride

    def forward(self, x):
        """ """
        y = custom_conv_pad(
            x,
            pad=self.pad,
            weight=self.weight,
            stride=self.stride,
            mode="constant",
            value=0,
        )
        y = y.view(-1, 1, *y.shape[-2:])
        y = torch.nn.functional.conv2d(
            input=y,
            weight=self.weight,
            bias=None,
            stride=self.stride,
            padding="valid",
            dilation=1,
            groups=1,
        )
        y = y.view(*x.shape[:-2], *y.shape[-2:])
        return y


class HanningPooling(ChannelwiseConv2d):
    def __init__(
        self,
        stride=[1, 1],
        kernel_size=[1, 1],
        padding="same",
        sqrt_window=False,
        normalize=False,
        dtype=torch.float32,
    ):
        """ """
        kernel = torch.ones(kernel_size, dtype=dtype)
        for dim, m in enumerate(kernel_size):
            shape = [-1 if _ == dim else 1 for _ in range(len(kernel_size))]
            kernel = kernel * torch.signal.windows.hann(
                m,
                sym=True,
                dtype=dtype,
            ).reshape(shape)
        if sqrt_window:
            kernel = torch.sqrt(kernel)
        if normalize:
            kernel = kernel / torch.sum(kernel)
        super().__init__(kernel.numpy(), pad=padding, stride=stride, dtype=dtype)


class CustomPaddedConv2d(torch.nn.Conv2d):
    def __init__(self, *args, **kwargs):
        """ """
        self.pad = kwargs.get("padding", 0)
        if isinstance(self.pad, int):
            self.pad = (self.pad, self.pad)
        if isinstance(self.pad, str):
            kwargs["padding"] = 0
        super().__init__(*args, **kwargs)

    def forward(self, x):
        """ """
        y = custom_conv_pad(
            x,
            pad=self.pad,
            weight=self.weight,
            stride=self.stride,
            mode="constant" if self.padding_mode == "zeros" else self.padding_mode,
            value=0,
        )
        y = torch.nn.functional.conv2d(
            input=y,
            weight=self.weight,
            bias=self.bias,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            groups=self.groups,
        )
        return y


class CustomFlatten(torch.nn.Module):
    def __init__(
        self,
        start_dim=0,
        end_dim=-1,
        permute_dims=None,
    ):
        """ """
        super().__init__()
        self.start_dim = start_dim
        self.end_dim = end_dim
        self.permute_dims = permute_dims

    def forward(self, x):
        """ """
        if self.permute_dims is not None:
            x = torch.permute(x, dims=self.permute_dims)
        return torch.flatten(x, start_dim=self.start_dim, end_dim=self.end_dim)


class CustomNorm(torch.nn.Module):
    def __init__(
        self,
        input_shape=[None, None, None, None],
        dim_affine=None,
        dim_norm=None,
        correction=1,
        eps=1e-05,
        dtype=torch.float32,
    ):
        """ """
        super().__init__()
        self.input_shape = input_shape
        self.dim_affine = dim_affine
        self.dim_norm = dim_norm
        self.correction = correction
        self.eps = eps
        self.dtype = dtype
        if self.dim_affine is not None:
            msg = "`input_shape` is required when `dim_affine` is not None"
            assert self.input_shape is not None, msg
            size = input_shape[self.dim_affine]
            self.shape = [1 for _ in self.input_shape]
            self.shape[self.dim_affine] = input_shape[self.dim_affine]
            self.weight = torch.nn.parameter.Parameter(
                data=torch.squeeze(torch.ones(size, dtype=self.dtype)),
                requires_grad=True,
            )
            self.bias = torch.nn.parameter.Parameter(
                data=torch.squeeze(torch.zeros(size, dtype=self.dtype)),
                requires_grad=True,
            )

    def forward(self, x):
        """ """
        x_var, x_mean = torch.var_mean(
            x, dim=self.dim_norm, correction=self.correction, keepdim=True
        )
        y = (x - x_mean) / torch.sqrt(x_var + self.eps)
        if self.dim_affine is not None:
            w = self.weight.view(self.shape)
            b = self.bias.view(self.shape)
            y = (y * w) + b
        return y


class Permute(torch.nn.Module):
    def __init__(self, dims=None):
        """ """
        super().__init__()
        self.dims = dims

    def forward(self, x):
        """ """
        return torch.permute(x, dims=self.dims)


class Reshape(torch.nn.Module):
    def __init__(self, shape=None):
        """ """
        super().__init__()
        self.shape = shape

    def forward(self, x):
        """ """
        return torch.reshape(x, shape=self.shape)


class Unsqueeze(torch.nn.Module):
    def __init__(self, dim=None):
        """ """
        super().__init__()
        self.dim = dim

    def forward(self, x):
        """ """
        return torch.unsqueeze(x, dim=self.dim)


class RandomSlice(torch.nn.Module):
    def __init__(self, size=[50, 20000], buffer=[0, 0], **kwargs):
        """ """
        super().__init__()
        self.size = size
        self.pre_crop_slice = []
        for b in buffer:
            if b is None:
                self.pre_crop_slice.append(slice(None))
            elif isinstance(b, int) and b > 0:
                self.pre_crop_slice.append(slice(b, -b))
            elif isinstance(b, int) and b == 0:
                self.pre_crop_slice.append(slice(None))
            elif isinstance(b, (tuple, list)):
                self.pre_crop_slice.append(slice(*b))
        self.crop = torchvision.transforms.RandomCrop(size=self.size, **kwargs)

    def forward(self, x):
        """ """
        return self.crop(x[..., *self.pre_crop_slice])
