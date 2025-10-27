from __future__ import annotations

import torch
import torch.nn as nn
from torch.distributions import Normal

from humanoidverse.agents.modules.encoder_modules import ConvEncoder
from .modules import BaseModule, MoEMLP


class Actor(nn.Module):
    def __init__(
        self,
        obs_dim_dict,
        module_config_dict,
        num_actions,
    ):
        super(Actor, self).__init__()

        actor_module_config_dict = self._process_module_config(module_config_dict, num_actions)

        self.actor_net_type = actor_module_config_dict.get("type", "MLP")

        if self.actor_net_type == "MLP":
            self.actor_module = BaseModule(obs_dim_dict, actor_module_config_dict)
        elif self.actor_net_type == "MoEMLP":
            self.actor_module = MoEMLP(obs_dim_dict, actor_module_config_dict)
        else:
            raise NotImplementedError

        new_obs_dim_dict = obs_dim_dict.copy()

        self.motion_encoder = ConvEncoder(
            obs_dim_dict,
            actor_module_config_dict.motion_encoder,
            actor_module_config_dict.motion_encoder.tsteps,
        )

        if getattr(actor_module_config_dict, "history_encoder", None) is not None:
            new_obs_dim_dict["prop_history"] //= actor_module_config_dict.history_encoder.tsteps
            self.history_encoder = ConvEncoder(
                new_obs_dim_dict,
                actor_module_config_dict.history_encoder,
                actor_module_config_dict.history_encoder.tsteps,
            )
        else:
            self.history_encoder = None

        if getattr(actor_module_config_dict, "priv_encoder", None) is not None:
            self.priv_encoder = BaseModule(obs_dim_dict, actor_module_config_dict.priv_encoder)
        else:
            self.priv_encoder = None

    def _process_module_config(self, module_config_dict, num_actions):
        for idx, output_dim in enumerate(module_config_dict["output_dim"]):
            if output_dim == "robot_action_dim":
                module_config_dict["output_dim"][idx] = num_actions
        return module_config_dict

    def motion_encoding(self, motion_obs):
        motion_embedding = self.motion_encoder(motion_obs)
        return motion_embedding

    def history_encoding(self, history_obs):
        history_embedding = self.history_encoder(history_obs)
        return history_embedding

    def priv_encoding(self, priv_obs):
        priv_embedding = self.priv_encoder(priv_obs)
        return priv_embedding

    def forward(self, obs_dict, hist_encoding: bool, obs_key="actor_obs", target_key="future_motion_targets"):
        motion_embedding = self.motion_encoding(obs_dict[target_key])

        if hist_encoding:
            latent = self.history_encoding(obs_dict["prop_history"])
        else:
            latent = self.priv_encoding(obs_dict["priv_obs"])

        actor_obs = torch.cat([obs_dict[obs_key], motion_embedding, latent], dim=-1)
        backbone_output = self.actor_module(actor_obs)
        return backbone_output


class ActorCritic(nn.Module):
    def __init__(
        self,
        obs_dim_dict,
        module_config_dict,
        num_actions,
        init_noise_std,
    ):
        super(ActorCritic, self).__init__()

        self.actor_module = Actor(obs_dim_dict, module_config_dict.actor, num_actions)

        critic_module_config_dict = module_config_dict.critic
        self.critic_net_type = critic_module_config_dict.get("type", "MLP")
        if self.critic_net_type == "MLP":
            self.critic_module = BaseModule(obs_dim_dict, critic_module_config_dict)
        elif self.critic_net_type == "MoEMLP":
            self.critic_module = MoEMLP(obs_dim_dict, critic_module_config_dict)
        else:
            raise NotImplementedError

        # Action noise
        self.std = nn.Parameter(init_noise_std * torch.ones(num_actions))
        self.fix_sigma = module_config_dict.actor.get("fix_sigma", False)
        self.max_sigma = module_config_dict.actor.get("max_sigma", 1.0)
        self.min_sigma = module_config_dict.actor.get("min_sigma", 0.1)

        if self.fix_sigma:
            self.std.requires_grad = False
        self.distribution = None
        # disable args validation for speedup
        Normal.set_default_validate_args = False

    @property
    def actor(self):
        return self.actor_module

    @property
    def critic(self):
        return self.critic_module

    @staticmethod
    # not used at the moment
    def init_weights(sequential, scales):
        [
            torch.nn.init.orthogonal_(module.weight, gain=scales[idx])
            for idx, module in enumerate(mod for mod in sequential if isinstance(mod, nn.Linear))
        ]

    def reset(self, dones=None):
        pass

    def forward(self):
        raise NotImplementedError

    @property
    def action_mean(self):
        return self.distribution.mean

    @property
    def action_std(self):
        return self.distribution.stddev

    @property
    def entropy(self):
        return self.distribution.entropy().sum(dim=-1)

    def update_distribution(self, obs, hist_encoding, obs_key):
        mean = self.actor(obs, hist_encoding, obs_key)
        self.distribution = Normal(mean, (mean * 0.0 + self.std).clamp(min=self.min_sigma, max=self.max_sigma))

    def act(self, obs, hist_encoding=False, obs_key="actor_obs", **kwargs):
        self.update_distribution(obs, hist_encoding, obs_key)
        return self.distribution.sample()

    def get_actions_log_prob(self, actions):
        return self.distribution.log_prob(actions).sum(dim=-1)

    def act_inference(self, obs, hist_encoding=True, **kwargs):
        actions_mean = self.actor(obs, hist_encoding)
        return actions_mean

    def evaluate(self, obs, obs_key="actor_obs", **kwargs):
        motion_embedding = self.actor.motion_encoding(obs["future_motion_targets"])
        critic_obs = torch.cat([obs[obs_key], obs["priv_obs"], motion_embedding], dim=-1)
        return self.critic(critic_obs)
