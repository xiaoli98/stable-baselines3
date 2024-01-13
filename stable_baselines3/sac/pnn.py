import os
from typing import Any, Dict, List, Optional, Tuple, Type, Union

from gymnasium import spaces
import numpy as np
from torch import nn
import torch as th
import torch.nn.functional as F

from stable_baselines3.common.policies import BasePolicy
from stable_baselines3.common.torch_layers import (
    BaseFeaturesExtractor,
    CombinedExtractor,
    FlattenExtractor,
    NatureCNN,
)
from avalanche.benchmarks.utils.flat_data import ConstantSequence
from avalanche.models import MultiTaskModule
from avalanche.models import MultiHeadClassifier
from avalanche.benchmarks.scenarios import CLExperience

from stable_baselines3.common.type_aliases import PyTorchObs, Schedule
from stable_baselines3.common.preprocessing import get_action_dim
from stable_baselines3.sac.policies import SACPolicy, LOG_STD_MAX, LOG_STD_MIN
from stable_baselines3.sac.sub_policies import SubSACPolicy


class LinearAdapter(nn.Module):
    """
    Linear adapter for Progressive Neural Networks.
    """

    def __init__(self, in_features, out_features_per_column, num_prev_modules):
        """
        :param in_features: size of each input sample
        :param out_features_per_column: size of each output sample
        :param num_prev_modules: number of previous modules
        """
        super().__init__()
        # Eq. 1 - lateral connections
        # one layer for each previous column. Empty for the first task.
        self.lat_layers = nn.ModuleList([])
        for _ in range(num_prev_modules):
            m = nn.Linear(in_features, out_features_per_column)
            self.lat_layers.append(m)

    def forward(self, x):
        assert len(x) == self.num_prev_modules
        hs = []
        for ii, lat in enumerate(self.lat_layers):
            hs.append(lat(x[ii]))
        return sum(hs)


class MLPAdapter(nn.Module):
    """
    MLP adapter for Progressive Neural Networks.
    """

    def __init__(
        self,
        in_features,
        out_features_per_column,
        num_prev_modules,
        activation=F.relu,
    ):
        """
        :param in_features: size of each input sample
        :param out_features_per_column: size of each output sample
        :param num_prev_modules: number of previous modules
        :param activation: activation function (default=ReLU)
        """
        super().__init__()
        self.num_prev_modules = num_prev_modules
        self.activation = activation

        if num_prev_modules == 0:
            return  # first adapter is empty

        # Eq. 2 - MLP adapter. Not needed for the first task.
        self.V = nn.Linear(in_features * num_prev_modules, out_features_per_column)
        self.alphas = nn.Parameter(th.randn(num_prev_modules))
        self.U = nn.Linear(out_features_per_column, out_features_per_column)

    def forward(self, x):
        if self.num_prev_modules == 0:
            return 0  # first adapter is empty

        assert len(x) == self.num_prev_modules
        assert len(x[0].shape) == 2, (
            "Inputs to MLPAdapter should have two dimensions: "
            "<batch_size, num_features>."
        )
        for i, el in enumerate(x):
            x[i] = self.alphas[i] * el
        x = th.cat(x, dim=1)
        x = self.U(self.activation(self.V(x)))
        return x


class PNNColumn(SubSACPolicy):
    """
    Progressive Neural Network column.
    """

    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Box,
        lr_schedule: Schedule,
        num_prev_modules,
        net_arch: Optional[Union[List[int], Dict[str, List[int]]]] = None,
        activation_fn: Type[nn.Module] = nn.ReLU,
        use_sde: bool = False,
        log_std_init: float = -3,
        use_expln: bool = False,
        clip_mean: float = 2.0,
        features_extractor_class: Type[BaseFeaturesExtractor] = FlattenExtractor,
        features_extractor_kwargs: Optional[Dict[str, Any]] = None,
        normalize_images: bool = True,
        optimizer_class: Type[th.optim.Optimizer] = th.optim.Adam,
        optimizer_kwargs: Optional[Dict[str, Any]] = None,
        n_critics: int = 2,
        share_features_extractor: bool = False,
        truncate_obs: bool = False,
        sub_policy_name: str = None,
        adapter="mlp",
    ):

        super().__init__(
                observation_space,
                action_space,
                lr_schedule,
                net_arch,
                activation_fn,
                use_sde,
                log_std_init,
                use_expln,
                clip_mean,
                features_extractor_class,
                features_extractor_kwargs,
                normalize_images,
                optimizer_class,
                optimizer_kwargs,
                n_critics,
                share_features_extractor,
                truncate_obs,
                sub_policy_name,
        )
        self.in_features = self.net_arch[-1] if len(self.net_arch) > 0 else self.actor.features_dim
        self.out_features_per_column = get_action_dim(self.action_space)
        self.num_prev_modules = num_prev_modules
        
        # print(f"\tstats: \n\tin_features: {self.in_features}")
        # print(f"\tout_features: {self.out_features_per_column}")
        # print(f"\tnum  prev modules: {self.num_prev_modules}")
        # print(f"\tfeature extractor: {self.actor.features_extractor}")
        # print(f"\ttruncate obs : {truncate_obs}")
        # self.itoh = policy(policy_kwargs)
        if adapter == "linear":
            self.adapter = LinearAdapter(
                self.in_features, self.out_features_per_column, num_prev_modules
            )
        elif adapter == "mlp":
            self.adapter = MLPAdapter(
                self.in_features, self.out_features_per_column, num_prev_modules
            )
        else:
            raise ValueError("`adapter` must be one of: {'mlp', `linear'}.")
        # print(f"\tadapter: {self.adapter if self.adapter is not None else None}")
        
    def freeze(self):
        for param in self.parameters():
            param.requires_grad = False
            
    def _get_latent(self, obs:PyTorchObs):
        # print(f"{self.sub_policy_name}: {self.truncate_obs}")
        # print(f"obs shape: {obs.shape}")
        if self.truncate_obs:
            # truncate the observation to only the fist row
            obs = th.narrow(obs, 1, 0, 1)
        features = self.extract_features(obs, self.actor.features_extractor)
        return self.actor.latent_pi(features)
        
    def get_action_dist_params(self, obs: PyTorchObs, previous_latent) -> Tuple[th.Tensor, th.Tensor, Dict[str, th.Tensor]]:
        """
        Get the parameters for the action distribution.

        :param obs:
        :return:
            Mean, standard deviation and optional keyword arguments.
        """
        # features = self.extract_features(obs, self.actor.features_extractor)
        # latent_pi = self.actor.latent_pi(features)
        latent_pi = self._get_latent(obs)
        adapter_out = self.adapter(previous_latent)
        mean_actions = self.actor.mu(latent_pi) + adapter_out

        if self.actor.use_sde:
            return mean_actions, self.actor.log_std, dict(latent_sde=latent_pi)
        # Unstructured exploration (Original implementation)
        log_std = self.actor.log_std(latent_pi)  # type: ignore[operator]
        # Original Implementation to cap the standard deviation
        log_std = th.clamp(log_std, LOG_STD_MIN, LOG_STD_MAX)
        return mean_actions, log_std, {}
    
    def forward(self, obs: PyTorchObs, previous_latent, deterministic: bool = False) -> th.Tensor:
        mean_actions, log_std, kwargs = self.get_action_dist_params(obs, previous_latent)
        return self.actor.action_dist.actions_from_params(mean_actions, log_std, deterministic=deterministic, **kwargs)

    def action_log_prob(self, obs: PyTorchObs, previous_latent) -> Tuple[th.Tensor, th.Tensor]:
        mean_actions, log_std, kwargs = self.get_action_dist_params(obs, previous_latent)
        return self.actor.action_dist.log_prob_from_params(mean_actions, log_std, **kwargs)


class PNN_Policy(BasePolicy):
    """
    SAC Policy with adaptation from previous SAC policym like the PNN.
    adaptation only for latent
    """
    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Box,
        lr_schedule: Schedule,
        net_arch: Optional[Union[List[int], Dict[str, List[int]]]] = None,
        activation_fn: Type[nn.Module] = nn.ReLU,
        use_sde: bool = False,
        log_std_init: float = -3,
        use_expln: bool = False,
        clip_mean: float = 2.0,
        features_extractor_class: Type[BaseFeaturesExtractor] = FlattenExtractor,
        features_extractor_kwargs: Optional[Dict[str, Any]] = None,
        normalize_images: bool = True,
        optimizer_class: Type[th.optim.Optimizer] = th.optim.Adam,
        optimizer_kwargs: Optional[Dict[str, Any]] = None,
        n_critics: int = 2,
        share_features_extractor: bool = False,
        sub_policies_path: str = "/home/boy/stable-baselines3/checkpoint/subpolicies_5vehicles_MIRINL/",
        adapter="mlp",
    ):

        super().__init__(            
            observation_space,
            action_space,
            features_extractor_class,
            features_extractor_kwargs,
            optimizer_class=optimizer_class,
            optimizer_kwargs=optimizer_kwargs,
            squash_output=True,
            normalize_images=normalize_images,
        )
        checkpoints = os.listdir(sub_policies_path)
        assert len(checkpoints) >= 1
        self.num_columns = len(checkpoints) + 1

        self.columns = nn.ModuleList()
        for i in range(self.num_columns - 1):
            print(f"loading sub policy: {checkpoints[i]}...")
            obs_space = observation_space
            truncate_obs = False
            if "lane_centering" in checkpoints[i]:
                obs_space = spaces.Box(-np.inf, np.inf, (1, 9), np.float32)
                truncate_obs = True
            col = PNNColumn(
                obs_space,
                action_space,
                lr_schedule,
                0, #num_prev_modules
                net_arch,
                activation_fn,
                use_sde,
                log_std_init,
                use_expln,
                clip_mean,
                features_extractor_class,
                features_extractor_kwargs,
                normalize_images,
                optimizer_class,
                optimizer_kwargs,
                n_critics,
                share_features_extractor,
                truncate_obs,
                checkpoints[i], #subpolicy name
                adapter=adapter,
            )
            load_weights = th.load(sub_policies_path+checkpoints[i], map_location=th.device('cpu'))
            col.load_state_dict(load_weights)
            for params in col.parameters():
                params.requires_grad = False
            self.columns.append(col)
            print("DONE")
        
        # create last col
        last_col = PNNColumn(
                observation_space,
                action_space,
                lr_schedule,
                self.num_columns - 1, #num_prev_modules
                net_arch,
                activation_fn,
                use_sde,
                log_std_init,
                use_expln,
                clip_mean,
                features_extractor_class,
                features_extractor_kwargs,
                normalize_images,
                optimizer_class,
                optimizer_kwargs,
                n_critics,
                share_features_extractor,
                False,# truncate_obs,
                "racetrack",#sub_policy_name,
                adapter=adapter,
            )
        self.columns.append(last_col)
        self._create_alias()
        # self.classifier = MultiHeadClassifier(hidden_features_per_column)

    def _create_alias(self):
        # refer actor and critic only for the last column
        self.actor = self.columns[-1].actor
        self.critic = self.columns[-1].critic
        self.critic_target = self.columns[-1].critic_target
        
    def set_training_mode(self, mode: bool) -> None:
        """
        Put the policy in either training or evaluation mode.

        This affects certain modules, such as batch normalisation and dropout.

        :param mode: if true, set to training mode, else set to evaluation mode
        """
        self.actor.set_training_mode(mode)
        self.critic.set_training_mode(mode)
        self.training = mode
        
    def _predict(self, obs: PyTorchObs, deterministic: bool = False) -> th.Tensor:
        # obs = obs.flatten(1)
        previous_latent = [ col._get_latent(obs) for col in self.columns[:-1]]
        # for idx, prev_latent in enumerate(previous_latent):
        #     print(f"{self.columns[idx].sub_policy_name}: {prev_latent}\nshape:{prev_latent.shape}")
        return self.columns[-1](obs, previous_latent, deterministic)
    
__all__ = ["PNN", "PNNLayer", "PNNColumn", "MLPAdapter", "LinearAdapter"]
