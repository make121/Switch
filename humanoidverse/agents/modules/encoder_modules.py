from __future__ import annotations

import inspect

import torch.nn as nn

from .modules import BaseModule


class Estimator(nn.Module):
    def __init__(self, obs_dim_dict, module_config_dict):
        super(Estimator, self).__init__()
        self.module = BaseModule(obs_dim_dict, module_config_dict)

    # def estimate(self, obs_history):
    #     return self.module(obs_history)

    def forward(self, obs_history):
        return self.module(obs_history)


class ConvEncoder(nn.Module):
    def __init__(self, obs_dim_dict, module_config_dict, time_steps):
        super(ConvEncoder, self).__init__()
        self.obs_dim_dict = obs_dim_dict
        self.module_config_dict = module_config_dict
        self.time_steps = time_steps

        self._calculate_dim()
        self._build_network_layer(self.module_config_dict.layer_config)

    def _calculate_dim(self):
        input_dim = 0
        for each_input in self.module_config_dict["input_dim"]:
            if each_input in self.obs_dim_dict:
                # atomic observation type
                input_dim += self.obs_dim_dict[each_input]
            elif isinstance(each_input, (int, float)):
                # direct numeric input
                input_dim += each_input
            else:
                current_function_name = inspect.currentframe().f_code.co_name
                raise ValueError(f"{current_function_name} - Unknown input type: {each_input}")

        self.input_dim = input_dim
        self.output_dim = self.module_config_dict["output_dim"]
        self.hidden_dim = self.module_config_dict["hidden_dim"]

    def _build_network_layer(self, layer_config):
        layer_config = self._build_layer_config(layer_config, self.time_steps)

        self.encoder = nn.Sequential(nn.Linear(self.input_dim, self.hidden_dim), nn.ReLU())
        if layer_config["type"] == "Conv1d":
            self._build_conv_layer(layer_config)
        else:
            raise NotImplementedError(f"Unsupported layer type: {layer_config['type']}")

        self.output_layer = nn.Linear(layer_config["out_channels"][-1] * 3, self.output_dim)  ## dead

    def _build_layer_config(self, base_config, tsteps):
        if tsteps == 5:
            out_channels = [20, 10]
            kernel_sizes = [2, 2]
            strides = [1, 1]

        elif tsteps == 10:
            out_channels = [20, 10]
            kernel_sizes = [4, 2]
            strides = [2, 1]

        elif tsteps == 20:
            out_channels = [40, 20]
            kernel_sizes = [6, 4]
            strides = [2, 2]
        else:
            raise ValueError(f"Unsupported time_steps for now: {tsteps}")

        return dict(
            type=base_config["type"],
            out_channels=out_channels,
            kernel_sizes=kernel_sizes,
            strides=strides,
            activation=base_config["activation"],
        )

    def _build_conv_layer(self, layer_config):
        layers = []
        out_channels = layer_config["out_channels"]
        kernel_sizes = layer_config["kernel_sizes"]
        strides = layer_config["strides"]

        activation = getattr(nn, layer_config["activation"])()
        in_ch = self.hidden_dim

        for i in range(len(out_channels)):
            layers.append(nn.Conv1d(in_ch, out_channels[i], kernel_sizes[i], strides[i]))
            layers.append(activation)
            in_ch = out_channels[i]

        self.conv_module = nn.Sequential(*layers)

    def forward(self, x):
        x = x.view(-1, self.input_dim)
        x = self.encoder(x)
        x = x.view(-1, self.time_steps, self.hidden_dim).permute(0, 2, 1)
        x = self.conv_module(x).flatten(start_dim=1)
        return self.output_layer(x)
