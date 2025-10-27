import math
import os
import statistics
import time
from collections import deque
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from hydra.utils import instantiate
from loguru import logger
from omegaconf import OmegaConf
from rich.console import Console
from rich.live import Live
from rich.panel import Panel
from torch.utils.tensorboard import SummaryWriter as TensorboardSummaryWriter
from tqdm import tqdm

from humanoidverse.agents.base_algo.base_algo import BaseAlgo
from humanoidverse.agents.callbacks.base_callback import RL_EvalCallback
from humanoidverse.agents.modules.agent_modules import Actor, ActorCritic
from humanoidverse.agents.modules.data_utils import RolloutStorage
from humanoidverse.envs.base_task.base_task import BaseTask
from humanoidverse.utils.average_meters import TensorAverageMeterDict
from humanoidverse.utils.helpers import determine_obs_dim
import glob

console = Console()


class PPO(BaseAlgo):
    def __init__(self, env: BaseTask, config, log_dir=None, device="cpu"):
        self.device = device
        self.env = env
        self.config = config
        self.log_dir = log_dir
        self.writer = TensorboardSummaryWriter(log_dir=self.log_dir, flush_secs=10)

        self.start_time = 0
        self.stop_time = 0
        self.collection_time = 0
        self.learn_time = 0

        self._init_config()

        self.tot_timesteps = 0
        self.tot_time = 0
        self.current_learning_iteration = 0

        # Book keeping
        self.ep_infos: List[Dict[str, Any]] = []
        self.rewbuffer = deque(maxlen=100)
        self.lenbuffer = deque(maxlen=100)
        self.cur_reward_sum = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)
        self.cur_episode_length = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)

        self.eval_callbacks: list[RL_EvalCallback] = []
        self.episode_env_tensors = TensorAverageMeterDict()
        _ = self.env.reset_all()
        self.learn = self.learn_RL if not self.train_distill else self.learn_distill

    def _init_config(self):
        # Env related Config
        self.num_envs: int = self.env.config.num_envs
        self.algo_obs_dim_dict = self.env.config.robot.algo_obs_dim_dict
        self.num_act = self.env.config.robot.actions_dim

        # Logging related Config
        self.save_interval = self.config.save_interval
        self.logging_interval = self.config.get("logging_interval", 10)

        # Training related Config
        self.num_steps_per_env = self.config.num_steps_per_env
        self.load_optimizer = self.config.load_optimizer
        self.num_learning_iterations = self.config.num_learning_iterations
        self.init_at_random_ep_len = self.config.init_at_random_ep_len

        # Algorithm related Config

        self.desired_kl = self.config.desired_kl
        self.schedule = self.config.schedule
        self.learning_rate = self.config.learning_rate

        self.clip_param = self.config.clip_param
        self.num_learning_epochs = self.config.num_learning_epochs
        self.num_mini_batches = self.config.num_mini_batches
        self.gamma = self.config.gamma
        self.lam = self.config.lam
        self.value_loss_coef = self.config.value_loss_coef
        self.entropy_coef = self.config.entropy_coef
        self.max_grad_norm = self.config.max_grad_norm
        self.use_clipped_value_loss = self.config.use_clipped_value_loss
        self.num_rew_fn = self.env.num_rew_fn
        self.priv_reg_coef_schedual = self.config.priv_reg_coef_schedual
        self.counter = 0

        # Teacher-student related config
        if self.config.teacher_model_path is not None:
            self.train_distill = True
        else:
            self.train_distill = False

        self.dagger_only = self.config.dagger_only
        if self.train_distill:
            self._preprocess_teacher_config()

        self.actor_type = self.config.module_dict.get("actor", {}).get("type", "MLP")
        if not self.dagger_only:
            self.critic_type = self.config.module_dict.get("critic", {}).get("type", "MLP")
        self.dagger_update_freq = self.config.get("dagger_update_freq", 20)

    def setup(self):
        logger.info("Setting up PPO")
        self._setup_models_and_optimizer()
        logger.info(f"Setting up Storage")
        self._setup_storage()

    def _preprocess_teacher_config(self):
        """Preprocess the teacher config to match MLP size && obs dict && obs dim."""

        if self.config.teacher_model_path is None:
            raise ValueError("Teacher load path is not set. Please provide a valid path to the teacher config.")

        config_path = Path(self.config.teacher_model_path).parent / "config.yaml"
        teacher_config = OmegaConf.load(config_path)
        teacher_obs_key = "teacher_actor_obs"
        obs_dim_dict, _, _ = determine_obs_dim(teacher_config)
        self.algo_obs_dim_dict[teacher_obs_key] = obs_dim_dict["actor_obs"]
        self.algo_obs_dim_dict["teacher_future_motion_targets"] = obs_dim_dict["future_motion_targets"]
        self.env.config.obs.obs_dict[teacher_obs_key] = teacher_config.obs.obs_dict["actor_obs"]
        self.env.config.obs.obs_dict["teacher_future_motion_targets"] = teacher_config.obs.obs_dict["future_motion_targets"]

        def replace_actor_obs(input_dim):
            return ["teacher_actor_obs" if x == "actor_obs" else x for x in input_dim]

        module_dict = teacher_config.algo.config.module_dict
        module_dict.actor.input_dim = replace_actor_obs(list(module_dict.actor.input_dim))
        module_dict.critic.input_dim = replace_actor_obs(list(module_dict.critic.input_dim))
        module_dict.actor.motion_encoder.input_dim = ["teacher_future_motion_targets"]

        OmegaConf.resolve(teacher_config)
        self.config.teacher_module_dict = module_dict

    def _setup_models_and_optimizer(self):
        if self.env.config.use_vec_reward:
            self.config.module_dict.critic["output_dim"][-1] = self.num_rew_fn

        if self.train_distill:
            self.teacher_actor = Actor(
                self.algo_obs_dim_dict,
                self.config.teacher_module_dict.actor,
                self.num_act,
            ).to(self.device)
            logger.info(f"Loading teacher actor from {self.config.teacher_model_path}")
            state_dict = torch.load(self.config.teacher_model_path, map_location=self.device)
            actor_state_dict = {k[len("actor_module.") :]: v for k, v in state_dict["model_state_dict"].items() if k.startswith("actor_module.")}
            self.teacher_actor.load_state_dict(actor_state_dict, strict=True)
            for param in self.teacher_actor.parameters():
                param.requires_grad = False
            self.teacher_actor.eval()

        self.alg = ActorCritic(
            self.algo_obs_dim_dict,
            self.config.module_dict,
            self.num_act,
            self.config.init_noise_std,
        ).to(self.device)

        # if self.load_critic_when_distill:
        #     self.alg.critic_module.load_state_dict(self.teacher.critic_module.state_dict())

        if self.train_distill:
            self.alg.actor.history_encoder.load_state_dict(self.teacher_actor.history_encoder.state_dict())
            for p in self.alg.actor.history_encoder.parameters():
                p.requires_grad_(False)
        else:
            self.hist_encoder_optimizer = optim.AdamW(self.alg.actor.history_encoder.parameters(), lr=self.learning_rate)

        if self.dagger_only:
            self.optimizer = optim.AdamW(self.alg.actor.parameters(), lr=self.learning_rate)
        else:
            self.optimizer = optim.AdamW(self.alg.parameters(), lr=self.learning_rate)

    def _setup_storage(self):
        self.storage = RolloutStorage(self.env.num_envs, self.num_steps_per_env, self.device)
        ## Register obs keys
        for obs_key, obs_dim in self.algo_obs_dim_dict.items():
            if obs_key == "future_motion_targets" or obs_key == "teacher_future_motion_targets":
                self.storage.register_key(
                    obs_key,
                    shape=(obs_dim * len(self.env.tar_obs_steps),),
                    dtype=torch.float,
                )
                self.storage.register_key(
                    "next_" + obs_key,
                    shape=(obs_dim * len(self.env.tar_obs_steps),),
                    dtype=torch.float,
                )
            else:
                self.storage.register_key(obs_key, shape=(obs_dim,), dtype=torch.float)
                self.storage.register_key("next_" + obs_key, shape=(obs_dim,), dtype=torch.float)

        ## Register others
        self.storage.register_key("actions", shape=(self.num_act,), dtype=torch.float)
        self.storage.register_key("rewards", shape=(self.num_rew_fn,), dtype=torch.float)
        self.storage.register_key("dones", shape=(1,), dtype=torch.bool)
        self.storage.register_key("values", shape=(self.num_rew_fn,), dtype=torch.float)
        self.storage.register_key("returns", shape=(self.num_rew_fn,), dtype=torch.float)
        self.storage.register_key("advantages", shape=(1,), dtype=torch.float)
        self.storage.register_key("actions_log_prob", shape=(1,), dtype=torch.float)
        self.storage.register_key("action_mean", shape=(self.num_act,), dtype=torch.float)
        self.storage.register_key("action_sigma", shape=(self.num_act,), dtype=torch.float)
        if self.train_distill:
            self.storage.register_key("teacher_actions", shape=(self.num_act,), dtype=torch.float)

    def _eval_mode(self):
        self.alg.eval()

    def _train_mode(self):
        self.alg.train()

    def load(self, ckpt_path):
        if ckpt_path is not None:
            logger.info(f"Loading checkpoint from {ckpt_path}")
            loaded_dict = torch.load(ckpt_path, map_location=self.device)

            self.alg.load_state_dict(loaded_dict["model_state_dict"])
            if self.load_optimizer:
                self.optimizer.load_state_dict(loaded_dict["optimizer_state_dict"])
                self.learning_rate = loaded_dict["optimizer_state_dict"]["param_groups"][0]["lr"]
                self.set_learning_rate(self.learning_rate)
                logger.info(f"Optimizer loaded from checkpoint")
                logger.info(f"Learning rate: {self.learning_rate}")
            self.current_learning_iteration = loaded_dict["iter"]
            return loaded_dict["infos"]

    def save(self, path, infos=None):
        logger.info(f"Saving checkpoint to {path}")
        torch.save(
            {
                "model_state_dict": self.alg.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "iter": self.current_learning_iteration,
                "infos": infos,
            },
            path,
        )

    def learn_RL(self):
        if self.init_at_random_ep_len:
            self.env.episode_length_buf = torch.randint_like(self.env.episode_length_buf, high=int(self.env.max_episode_length))

        obs_dict = self.env.reset_all()
        for obs_key in obs_dict.keys():
            obs_dict[obs_key] = obs_dict[obs_key].to(self.device)

        self._train_mode()

        num_learning_iterations = self.num_learning_iterations

        tot_iter = self.current_learning_iteration + num_learning_iterations

        for it in range(self.current_learning_iteration, tot_iter):
            self.hist_encoding = it % self.dagger_update_freq == 0
            self.start_time = time.time()

            obs_dict = self._rollout_step(obs_dict)

            loss_dict = self._training_step()

            if self.hist_encoding:
                loss_dict = self._training_step_dagger()

            self.stop_time = time.time()
            self.learn_time = self.stop_time - self.start_time

            # Logging
            log_dict = {
                "it": it,
                "loss_dict": loss_dict,
                "collection_time": self.collection_time,
                "learn_time": self.learn_time,
                "ep_infos": self.ep_infos,
                "rewbuffer": self.rewbuffer,
                "lenbuffer": self.lenbuffer,
                "num_learning_iterations": num_learning_iterations,
            }
            self._post_epoch_logging(log_dict)
            if it % self.save_interval == 0:
                self.current_learning_iteration = it
                self.save(os.path.join(self.log_dir, "model_{}.pt".format(it)))
            self.ep_infos.clear()

        self.save(os.path.join(self.log_dir, "model_{}.pt".format(self.current_learning_iteration)))

    def learn_distill(self):
        if self.init_at_random_ep_len:
            self.env.episode_length_buf = torch.randint_like(self.env.episode_length_buf, high=int(self.env.max_episode_length))

        obs_dict = self.env.reset_all()
        for obs_key in obs_dict.keys():
            obs_dict[obs_key] = obs_dict[obs_key].to(self.device)

        self._train_mode()

        num_learning_iterations = self.num_learning_iterations

        tot_iter = self.current_learning_iteration + num_learning_iterations

        for it in range(self.current_learning_iteration, tot_iter):
            self.hist_encoding = True
            self.start_time = time.time()

            obs_dict = self._rollout_step(obs_dict)

            loss_dict = self._training_step_distill()

            self.stop_time = time.time()
            self.learn_time = self.stop_time - self.start_time

            # Logging
            log_dict = {
                "it": it,
                "loss_dict": loss_dict,
                "collection_time": self.collection_time,
                "learn_time": self.learn_time,
                "ep_infos": self.ep_infos,
                "rewbuffer": self.rewbuffer,
                "lenbuffer": self.lenbuffer,
                "num_learning_iterations": num_learning_iterations,
            }
            self._post_epoch_logging(log_dict)
            if it % self.save_interval == 0:
                self.current_learning_iteration = it
                self.save(os.path.join(self.log_dir, "model_{}.pt".format(it)))
            self.ep_infos.clear()

        self.save(os.path.join(self.log_dir, "model_{}.pt".format(self.current_learning_iteration)))

    def _actor_rollout_step(self, obs_dict, policy_state_dict):
        if self.train_distill:
            with torch.inference_mode():
                policy_state_dict["teacher_actions"] = self.teacher_actor_act_step(obs_dict, hist_encoding=True).detach()
        if self.dagger_only:
            actions = self.alg.act_inference(obs_dict, hist_encoding=True)
            policy_state_dict["actions"] = actions
            assert len(actions.shape) == 2
            return policy_state_dict
        else:
            actions = self._actor_act_step(obs_dict, self.hist_encoding)
            policy_state_dict["actions"] = actions
            action_mean = self.alg.action_mean.detach()
            action_sigma = self.alg.action_std.detach()
            actions_log_prob = self.alg.get_actions_log_prob(actions).detach().unsqueeze(1)
            policy_state_dict["action_mean"] = action_mean
            policy_state_dict["action_sigma"] = action_sigma
            policy_state_dict["actions_log_prob"] = actions_log_prob

            assert len(actions.shape) == 2
            assert len(actions_log_prob.shape) == 2
            assert len(action_mean.shape) == 2
            assert len(action_sigma.shape) == 2

            return policy_state_dict

    def _rollout_step(self, obs_dict):
        with torch.inference_mode():
            for i in range(self.num_steps_per_env):
                # Compute the actions and teacher_actions
                policy_state_dict = {}
                policy_state_dict = self._actor_rollout_step(obs_dict, policy_state_dict)
                if not self.dagger_only:
                    values = self._critic_eval_step(obs_dict).detach()  # (num_rew_fn, 1)
                    policy_state_dict["values"] = values

                ## Append states to storage
                for obs_key in obs_dict.keys():
                    self.storage.update_key(obs_key, obs_dict[obs_key])

                for obs_ in policy_state_dict.keys():
                    self.storage.update_key(obs_, policy_state_dict[obs_])
                actions = policy_state_dict["actions"]
                actor_state = {}
                actor_state["actions"] = actions
                obs_dict, rewards, dones, infos = self.env.step(actor_state)

                for obs_key in obs_dict.keys():
                    obs_dict[obs_key] = obs_dict[obs_key].to(self.device)
                    self.storage.update_key("next_" + obs_key, obs_dict[obs_key])

                rewards, dones = rewards.to(self.device), dones.to(self.device)

                self.episode_env_tensors.add(infos["to_log"])
                if not self.env.config.use_vec_reward:
                    rewards_stored = rewards.clone().unsqueeze(1)
                else:
                    rewards_stored = rewards.clone().reshape(self.env.num_envs, self.env.num_rew_fn)
                if not self.dagger_only:
                    if "time_outs" in infos:
                        rewards_stored += self.gamma * policy_state_dict["values"] * infos["time_outs"].unsqueeze(1).to(self.device)
                    assert len(rewards_stored.shape) == 2
                self.storage.update_key("rewards", rewards_stored)
                self.storage.update_key("dones", dones.unsqueeze(1))
                self.storage.increment_step()

                self._process_env_step(rewards, dones, infos)

                if self.log_dir is not None:
                    # Book keeping
                    if "episode" in infos:
                        self.ep_infos.append(infos["episode"])
                    self.cur_reward_sum += rewards.view(self.env.num_envs, self.env.num_rew_fn).sum(dim=-1)
                    self.cur_episode_length += 1
                    new_ids = (dones > 0).nonzero(as_tuple=False)
                    self.rewbuffer.extend(self.cur_reward_sum[new_ids][:, 0].cpu().numpy().tolist())
                    self.lenbuffer.extend(self.cur_episode_length[new_ids][:, 0].cpu().numpy().tolist())
                    self.cur_reward_sum[new_ids] = 0
                    self.cur_episode_length[new_ids] = 0

            self.stop_time = time.time()
            self.collection_time = self.stop_time - self.start_time
            self.start_time = self.stop_time

            if not self.dagger_only:
                returns, advantages = self._compute_returns(
                    last_obs_dict=obs_dict,
                    policy_state_dict=dict(
                        values=self.storage.query_key("values"),
                        dones=self.storage.query_key("dones"),
                        rewards=self.storage.query_key("rewards"),
                    ),
                )

                self.storage.batch_update_data("returns", returns)
                self.storage.batch_update_data("advantages", advantages)

        return obs_dict

    def _process_env_step(self, rewards, dones, infos):
        self.alg.reset(dones)

    def _compute_returns(self, last_obs_dict, policy_state_dict):
        """Compute the returns and advantages for the given policy state.
        This function calculates the returns and advantages for each step in the
        environment based on the provided observations and policy state. It uses
        Generalized Advantage Estimation (GAE) to compute the advantages, which
        helps in reducing the variance of the policy gradient estimates.
        Args:
            last_obs_dict (dict): The last observation dictionary containing the
                      final state of the environment.
            policy_state_dict (dict): A dictionary containing the policy state
                          information, including 'values', 'dones',
                          and 'rewards'.
        Returns:
            tuple: A tuple containing:
            - returns (torch.Tensor): The computed returns for each step.
            - advantages (torch.Tensor): The normalized advantages for each step.
        """
        last_values = self.alg.evaluate(last_obs_dict).detach()

        values = policy_state_dict["values"]
        dones = policy_state_dict["dones"]
        rewards = policy_state_dict["rewards"]

        last_values = last_values.to(self.device)
        values = values.to(self.device)
        dones = dones.to(self.device)
        rewards = rewards.to(self.device)

        returns = torch.zeros_like(values)
        # advantages = torch.zeros_like(dones)  # not vec, it must be a scalar

        num_steps = returns.shape[0]
        advantage = 0
        for step in reversed(range(num_steps)):
            if step == num_steps - 1:
                next_values = last_values
            else:
                next_values = values[step + 1]
            next_is_not_terminal = 1.0 - dones[step].float()
            delta = rewards[step] + next_is_not_terminal * self.gamma * next_values - values[step]
            advantage = delta + next_is_not_terminal * self.gamma * self.lam * advantage
            returns[step] = advantage + values[step]

        # Compute and normalize the advantages
        tot_advantages = returns - values
        if not self.env.config.use_vec_reward:
            advantages = (tot_advantages - tot_advantages.mean()) / (tot_advantages.std() + 1e-8)
        else:
            aggr_tot_advantages = tot_advantages.sum(dim=-1)
            advantages = ((aggr_tot_advantages - aggr_tot_advantages.mean()) / (aggr_tot_advantages.std() + 1e-8)).unsqueeze(-1)
        return returns, advantages

    def _training_step(self):
        loss_dict = self._init_loss_dict_at_training_step()

        generator = self.storage.mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)

        for policy_state_dict in generator:
            # Move everything to the device
            for policy_state_key in policy_state_dict.keys():
                policy_state_dict[policy_state_key] = policy_state_dict[policy_state_key].to(self.device)
            loss_dict = self._update_algo_step(policy_state_dict, loss_dict)

        num_updates = self.num_learning_epochs * self.num_mini_batches
        for key in loss_dict.keys():
            loss_dict[key] /= num_updates
        self.storage.clear()
        self.update_counter()
        return loss_dict

    def _training_step_dagger(self):
        loss_dict = self._init_hist_latent_loss_dict_at_dagger_step()

        generator = self.storage.mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)

        for policy_state_dict in generator:
            # Move everything to the device
            for policy_state_key in policy_state_dict.keys():
                policy_state_dict[policy_state_key] = policy_state_dict[policy_state_key].to(self.device)
            loss_dict = self._update_dagger_step(policy_state_dict, loss_dict)

        num_updates = self.num_learning_epochs * self.num_mini_batches
        for key in loss_dict.keys():
            loss_dict[key] /= num_updates
        self.storage.clear()
        self.update_counter()
        return loss_dict

    def _training_step_distill(self):
        loss_dict = self._init_hist_latent_loss_dict_at_distill_step()

        generator = self.storage.mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)

        for policy_state_dict in generator:
            # Move everything to the device
            for policy_state_key in policy_state_dict.keys():
                policy_state_dict[policy_state_key] = policy_state_dict[policy_state_key].to(self.device)
            loss_dict = self._update_distill_step(policy_state_dict, loss_dict)

        num_updates = self.num_learning_epochs * self.num_mini_batches
        for key in loss_dict.keys():
            loss_dict[key] /= num_updates
        self.storage.clear()
        return loss_dict

    def _init_hist_latent_loss_dict_at_dagger_step(self):
        loss_dict = {}
        loss_dict["hist_latent_loss"] = 0
        return loss_dict

    def _init_loss_dict_at_training_step(self):
        loss_dict = {}
        loss_dict["Value"] = 0
        loss_dict["Entropy"] = 0
        loss_dict["Surrogate"] = 0
        loss_dict["priv_reg_loss"] = 0
        loss_dict["Actor_Load_Balancing_Loss"] = 0
        loss_dict["Critic_Load_Balancing_Loss"] = 0
        return loss_dict

    def _init_hist_latent_loss_dict_at_distill_step(self):
        loss_dict = {}
        loss_dict["Value"] = 0
        loss_dict["Entropy"] = 0
        loss_dict["Surrogate"] = 0
        loss_dict["bc_loss"] = 0
        loss_dict["Actor_Load_Balancing_Loss"] = 0
        loss_dict["Critic_Load_Balancing_Loss"] = 0
        return loss_dict

    def _update_algo_step(self, policy_state_dict, loss_dict):
        loss_dict = self._update_ppo(policy_state_dict, loss_dict)
        return loss_dict

    def _update_dagger_step(self, policy_state_dict, loss_dict):
        loss_dict = self._update_dagger(policy_state_dict, loss_dict)
        return loss_dict

    def _update_distill_step(self, policy_state_dict, loss_dict):
        loss_dict = self._update_distill(policy_state_dict, loss_dict)
        return loss_dict

    def _actor_act_step(self, obs_dict, hist_encoding=False):
        return self.alg.act(obs_dict, hist_encoding)

    def teacher_actor_act_step(self, obs_dict, hist_encoding=True):
        return self.teacher_actor(obs_dict, hist_encoding, obs_key="teacher_actor_obs", target_key="teacher_future_motion_targets")

    def _critic_eval_step(self, obs_dict):
        return self.alg.evaluate(obs_dict)

    def _update_ppo(self, policy_state_dict, loss_dict):
        actions_batch = policy_state_dict["actions"]
        old_actions_log_prob_batch = policy_state_dict["actions_log_prob"]
        old_mu_batch = policy_state_dict["action_mean"]
        old_sigma_batch = policy_state_dict["action_sigma"]
        target_values_batch = policy_state_dict["values"]
        advantages_batch = policy_state_dict["advantages"]
        returns_batch = policy_state_dict["returns"]

        self._actor_act_step(policy_state_dict, hist_encoding=False)
        actions_log_prob_batch = self.alg.get_actions_log_prob(actions_batch)
        value_batch = self._critic_eval_step(policy_state_dict)
        mu_batch = self.alg.action_mean
        sigma_batch = self.alg.action_std
        entropy_batch = self.alg.entropy

        # Adaptation module update
        priv_latent_batch = self.alg.actor.priv_encoding(policy_state_dict["priv_obs"])
        with torch.inference_mode():
            hist_latent_batch = self.alg.actor.history_encoding(policy_state_dict["prop_history"])
        priv_reg_loss = (priv_latent_batch - hist_latent_batch.detach()).norm(p=2, dim=1).mean()
        priv_reg_stage = min(
            max((self.counter - self.priv_reg_coef_schedual[2]), 0) / self.priv_reg_coef_schedual[3],
            1,
        )
        priv_reg_coef = priv_reg_stage * (self.priv_reg_coef_schedual[1] - self.priv_reg_coef_schedual[0]) + self.priv_reg_coef_schedual[0]

        # KL
        if self.desired_kl is not None and self.schedule == "adaptive":
            with torch.inference_mode():
                kl = torch.sum(
                    torch.log(sigma_batch / (old_sigma_batch + 1e-5))
                    + (old_sigma_batch**2 + (old_mu_batch - mu_batch) ** 2) / (2.0 * sigma_batch**2)
                    - 0.5,
                    axis=-1,
                )
                kl_mean = kl.mean()
                if kl_mean > self.desired_kl * 2.0:
                    self.learning_rate = max(1e-5, self.learning_rate / 1.5)

                elif kl_mean < self.desired_kl / 2.0 and kl_mean > 0.0:
                    self.learning_rate = min(1e-2, self.learning_rate * 1.5)

                for param_group in self.optimizer.param_groups:
                    param_group["lr"] = self.learning_rate

        # Surrogate loss
        ratio = torch.exp(actions_log_prob_batch - torch.squeeze(old_actions_log_prob_batch))
        surrogate = -torch.squeeze(advantages_batch) * ratio
        surrogate_clipped = -torch.squeeze(advantages_batch) * torch.clamp(ratio, 1.0 - self.clip_param, 1.0 + self.clip_param)
        surrogate_loss = torch.max(surrogate, surrogate_clipped).mean()

        # Value function loss
        if self.use_clipped_value_loss:
            value_clipped = target_values_batch + (value_batch - target_values_batch).clamp(-self.clip_param, self.clip_param)
            value_losses = (value_batch - returns_batch).pow(2)
            value_losses_clipped = (value_clipped - returns_batch).pow(2)
            value_loss = torch.max(value_losses, value_losses_clipped).sum(dim=-1).mean()
        else:
            value_loss = (returns_batch - value_batch).pow(2).sum(dim=-1).mean()

        if self.critic_type == "MoEMLP":
            critic_load_balancing_loss = self.alg.critic.compute_load_balancing_loss() * self.config.get("load_balancing_loss_alpha", 1e-2)
            loss_dict["Critic_Load_Balancing_Loss"] += critic_load_balancing_loss.item()
            value_loss = value_loss + critic_load_balancing_loss

            loss_dict["Critic_Load_Balancing_Loss"] += critic_load_balancing_loss.item()

        entropy_loss = entropy_batch.mean()

        actor_loss = surrogate_loss - self.entropy_coef * entropy_loss

        # Load balancing loss (only for MoE-based actors)
        if self.actor_type == "MoEMLP":
            load_balancing_loss = self.alg.actor.actor_module.compute_load_balancing_loss() * self.config.get("load_balancing_loss_alpha", 1e-2)
            actor_loss = actor_loss + load_balancing_loss
            loss_dict["Actor_Load_Balancing_Loss"] += load_balancing_loss.item()

        critic_loss = self.value_loss_coef * value_loss

        total_loss = actor_loss + critic_loss + priv_reg_coef * priv_reg_loss

        self.optimizer.zero_grad()

        total_loss.backward()

        nn.utils.clip_grad_norm_(self.alg.parameters(), self.max_grad_norm)

        self.optimizer.step()

        loss_dict["Surrogate"] += surrogate_loss.item()
        loss_dict["Value"] += value_loss.item()
        loss_dict["Entropy"] += entropy_loss.item()
        loss_dict["priv_reg_loss"] += priv_reg_loss.item()

        return loss_dict

    def _update_dagger(self, policy_state_dict, loss_dict):
        with torch.inference_mode():
            self.alg.act(policy_state_dict, hist_encoding=True)

        # Adaptation module update
        with torch.inference_mode():
            priv_latent_batch = self.alg.actor.priv_encoding(policy_state_dict["priv_obs"])
        hist_latent_batch = self.alg.actor.history_encoding(policy_state_dict["prop_history"])
        hist_latent_loss = (priv_latent_batch.detach() - hist_latent_batch).norm(p=2, dim=1).mean()
        self.hist_encoder_optimizer.zero_grad()
        hist_latent_loss.backward()
        nn.utils.clip_grad_norm_(self.alg.actor.history_encoder.parameters(), self.max_grad_norm)
        self.hist_encoder_optimizer.step()

        loss_dict["hist_latent_loss"] += hist_latent_loss.item()

        return loss_dict

    def _update_distill(self, policy_state_dict, loss_dict):
        actions_teacher_batch = policy_state_dict["teacher_actions"]

        if self.dagger_only:
            actions_student_batch = self.alg.act_inference(policy_state_dict, hist_encoding=True)
            bc_loss = (actions_teacher_batch.detach() - actions_student_batch).norm(p=2, dim=1).mean()
            loss_dict["bc_loss"] += bc_loss.item()
            self.optimizer.zero_grad()
            bc_loss.backward()
            nn.utils.clip_grad_norm_(self.alg.actor.parameters(), self.max_grad_norm)
            self.optimizer.step()
            return loss_dict
        else:
            raise NotImplementedError("Distillation with PPO not implemented in non-DAGGER-only mode yet.")

    def update_counter(self):
        self.counter += 1

    def set_learning_rate(self, learning_rate):
        self.learning_rate = learning_rate

    @property
    def inference_model(self):
        return {"actor": self.alg.actor}

    def _post_epoch_logging(self, log_dict, width=80, pad=40):
        # Update total timesteps and total time
        self.tot_timesteps += self.num_steps_per_env * self.env.num_envs
        self.tot_time += log_dict["collection_time"] + log_dict["learn_time"]
        iteration_time = log_dict["collection_time"] + log_dict["learn_time"]

        if log_dict["it"] % self.logging_interval != 0:  # Check report frequency
            return

        # Closure functions to generate log strings
        def generate_computation_log():
            # Calculate mean standard deviation and frames per second (FPS)
            mean_std = self.alg.std.mean()
            fps = int(self.num_steps_per_env * self.env.num_envs / iteration_time)
            str = f" \033[1m Learning iteration {log_dict['it']}/{self.current_learning_iteration + log_dict['num_learning_iterations']} \033[0m "

            return (
                f"""{str.center(width, " ")}\n\n"""
                f"""{"Computation:":>{pad}} {fps:.0f} steps/s\n"""
                f"""{"Mean action noise std:":>{pad}} {mean_std:>10.4f}\n"""
            )

        def generate_reward_length_log():
            # Generate log for mean reward and mean episode length
            reward_length_string = ""

            if len(log_dict["rewbuffer"]) > 0:
                reward_length_string += (
                    f"""{"Mean reward:":>{pad}} {statistics.mean(log_dict["rewbuffer"]):>10.4f}\n"""
                    f"""{"Mean episode length:":>{pad}} {statistics.mean(log_dict["lenbuffer"]):>10.4f}\n"""
                )

                self.writer.add_scalar(
                    "Train/mean_reward",
                    statistics.mean(log_dict["rewbuffer"]),
                    log_dict["it"],
                )
                self.writer.add_scalar(
                    "Train/mean_episode_length",
                    statistics.mean(log_dict["lenbuffer"]),
                    log_dict["it"],
                )
            return reward_length_string

        def generate_env_log():
            # Generate log for environment metrics
            env_log_string = ""
            env_log_dict = self.episode_env_tensors.mean_and_clear()
            env_log_dict = {f"{k}": v for k, v in env_log_dict.items()}

            for k, v in env_log_dict.items():
                entry = f"{f'{k}:':>{pad}} {v:>10.4f}"
                env_log_string += f"{entry}\n"
                self.writer.add_scalar("Env/" + k, v, log_dict["it"])

            for loss_key, loss_value in log_dict["loss_dict"].items():
                self.writer.add_scalar(f"Learn/{loss_key}", loss_value, log_dict["it"])
            self.writer.add_scalar("Learn/actor_learning_rate", self.learning_rate, log_dict["it"])
            self.writer.add_scalar("Learn/mean_noise_std", self.alg.std.mean().item(), log_dict["it"])
            return env_log_string

        def generate_episode_log():
            # Generate log for episode information
            ep_string = f"{'-' * width}\n"  # Add a separator line before episode info

            if log_dict["ep_infos"]:
                # Initialize a dictionary to hold the sum and count for mean calculation
                mean_values = {key: 0.0 for key in log_dict["ep_infos"][0].keys()}
                total_episodes = 0

                for ep_info in log_dict["ep_infos"]:
                    # Sum the values for mean calculation
                    for key in mean_values.keys():
                        # Check if the key is 'end_epis_length' and handle it accordingly
                        if key == "end_epis_length":
                            # Sum the lengths of episodes
                            mean_values[key] += ep_info[key].sum().item()  # Convert tensor to scalar
                            total_episodes += ep_info[key].numel()  # Count the number of episodes
                        else:
                            mean_values[key] += (
                                (ep_info[key] / ep_info["end_epis_length"] * self.env.max_episode_length).sum().item()
                            )  # Average for other keys

                rew_total = 0
                for key, value in mean_values.items():
                    if key.startswith("rew_"):
                        rew_total += value

                mean_values["rew_total"] = rew_total

                # Calculate the mean for each key
                for key in mean_values.keys():
                    mean_values[key] /= total_episodes  # Mean over all episode lengths

                    self.writer.add_scalar("Env/" + key, mean_values[key], log_dict["it"])

                # Prepare the string for logging
                for key, value in mean_values.items():
                    if key == "end_epis_length":
                        continue
                    ep_string += f"""{f"{key}:":>{pad}} {value:>10.4f} \n"""  # Print mean values with 4 decimal places
            ep_string += f"Note: reward computed per step\n"

            return ep_string

        def generate_total_time_log():
            # Calculate ETA and generate total time log
            fps = int(self.num_steps_per_env * self.env.num_envs / iteration_time)
            eta = self.tot_time / (log_dict["it"] + 1) * (log_dict["num_learning_iterations"] - log_dict["it"])

            self.writer.add_scalar("Perf/total_fps", fps, log_dict["it"])
            self.writer.add_scalar("Perf/collection_time", log_dict["collection_time"], log_dict["it"])
            self.writer.add_scalar("Perf/learning_time", log_dict["learn_time"], log_dict["it"])
            self.writer.add_scalar("Perf/iter_time", iteration_time, log_dict["it"])
            self.writer.add_scalar("Perf/total_time", self.tot_time, log_dict["it"])  # Log total time

            return (
                f"""{"-" * width}\n"""
                f"""{"Total timesteps:":>{pad}} {self.tot_timesteps:.0f}\n"""  # Integer without decimal
                f"""{"Collection time:":>{pad}} {log_dict["collection_time"]:>10.4f}s\n"""  # Four decimal places
                f"""{"Learning time:":>{pad}} {log_dict["learn_time"]:>10.4f}s\n"""  # Four decimal places
                f"""{"Iteration time:":>{pad}} {iteration_time:>10.4f}s\n"""  # Four decimal places
                f"""{"Total time:":>{pad}} {self.tot_time:>10.4f}s\n"""  # Four decimal places
                f"""{"ETA:":>{pad}} {eta:>10.4f}s\n"""
                f"""{"Time Now:":>{pad}} {time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())}\n"""
            )  # Four decimal places

        # Generate all log strings
        log_string = (
            generate_computation_log()
            + generate_reward_length_log()
            + generate_env_log()
            + generate_episode_log()
            + generate_total_time_log()
            + f"Logging Directory: {self.log_dir}"
        )

        # Use rich Live to update a specific section of the console
        with Live(
            Panel(log_string, title="Training Log"),
            refresh_per_second=4,
            console=console,
        ):
            pass

    ##########################################################################################
    # Code for Evaluation
    ##########################################################################################

    def env_step(self, actor_state):
        obs_dict, rewards, dones, extras = self.env.step(actor_state)
        actor_state.update({"obs": obs_dict, "rewards": rewards, "dones": dones, "extras": extras})
        return actor_state

    @torch.no_grad()
    def get_example_obs(self):
        obs_dict = self.env.reset_all()
        for obs_key in obs_dict.keys():
            logger.info(f"{obs_key}, {sorted(self.env.config.obs.obs_dict[obs_key])}")
        # move to cpu
        for k in obs_dict:
            obs_dict[k] = obs_dict[k].cpu()
        return obs_dict

    @torch.no_grad()
    def evaluate_policy(self):
        self._create_eval_callbacks()
        self._pre_evaluate_policy()
        actor_state = self._create_actor_state()
        step = 0
        self.eval_policy = self._get_inference_policy()
        obs_dict = self.env.reset_all()
        init_actions = torch.zeros(self.env.num_envs, self.num_act, device=self.device)
        actor_state.update({"obs": obs_dict, "actions": init_actions})
        actor_state = self._pre_eval_env_step(actor_state)
        while True:
            actor_state["step"] = step
            actor_state = self._pre_eval_env_step(actor_state)
            actor_state = self.env_step(actor_state)
            actor_state = self._post_eval_env_step(actor_state)
            step += 1
        self._post_evaluate_policy()

    @torch.no_grad()
    def evaluate_policy_steps(self, Nsteps: int):
        self._create_eval_callbacks()
        self._pre_evaluate_policy()
        actor_state = self._create_actor_state()
        step = 0
        self.eval_policy = self._get_inference_policy()
        obs_dict = self.env.reset_all()
        init_actions = torch.zeros(self.env.num_envs, self.num_act, device=self.device)
        actor_state.update({"obs": obs_dict, "actions": init_actions})
        actor_state = self._pre_eval_env_step(actor_state)
        for step in tqdm(range(int(Nsteps)), desc="Evaluating Policy", unit="step"):
            actor_state["step"] = step
            actor_state = self._pre_eval_env_step(actor_state)
            actor_state = self.env_step(actor_state)
            actor_state = self._post_eval_env_step(actor_state)
            step += 1
        self._post_evaluate_policy()

    def _create_actor_state(self):
        return {"done_indices": [], "stop": False}

    def _create_eval_callbacks(self):
        if self.config.eval_callbacks is not None:
            for cb in self.config.eval_callbacks:
                self.eval_callbacks.append(instantiate(self.config.eval_callbacks[cb], training_loop=self))

    def _pre_evaluate_policy(self, reset_env=True):
        self._eval_mode()
        self.env.set_is_evaluating()
        if reset_env:
            _ = self.env.reset_all()

        for c in self.eval_callbacks:
            c.on_pre_evaluate_policy()

    def _post_evaluate_policy(self):
        for c in self.eval_callbacks:
            c.on_post_evaluate_policy()

    def _pre_eval_env_step(self, actor_state: dict):
        actions = self.eval_policy(actor_state["obs"])
        actor_state.update({"actions": actions})
        for c in self.eval_callbacks:
            actor_state = c.on_pre_eval_env_step(actor_state)
        return actor_state

    def _post_eval_env_step(self, actor_state):
        for c in self.eval_callbacks:
            actor_state = c.on_post_eval_env_step(actor_state)
        return actor_state

    def _get_inference_policy(self, device=None):
        self.alg.eval()  # switch to evaluation mode (dropout for example)
        if device is not None:
            self.alg.to(device)
        return self.alg.act_inference
