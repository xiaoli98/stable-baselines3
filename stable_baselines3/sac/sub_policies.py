from typing import Any, Dict, List, Optional, Tuple, Type, Union

import torch as th
from gymnasium import spaces
from torch import nn
from torch._tensor import Tensor

from stable_baselines3.common.torch_layers import (
    BaseFeaturesExtractor,
    CombinedExtractor,
    FlattenExtractor,
    NatureCNN,
)
from stable_baselines3.common.type_aliases import PyTorchObs, Schedule
from stable_baselines3.sac.policies import Actor, SACPolicy

class SubActor(Actor):
    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Box,
        net_arch: List[int],
        features_extractor: nn.Module,
        features_dim: int,
        activation_fn: Type[nn.Module] = nn.ReLU,
        use_sde: bool = False,
        log_std_init: float = -3,
        full_std: bool = True,
        use_expln: bool = False,
        clip_mean: float = 2.0,
        normalize_images: bool = True,
        truncate_obs: bool=False,
    ):
        super().__init__(
            observation_space,
            action_space,
            net_arch,
            features_extractor,
            features_dim,
            activation_fn,
            use_sde,
            log_std_init,
            full_std,
            use_expln,
            clip_mean,
            normalize_images,
        )
        self.truncate_obs = truncate_obs
        
    def forward(self, obs: PyTorchObs, deterministic: bool = False) -> th.Tensor:
        if self.truncate_obs:
            # truncate the observation to only the fist row
            obs = th.narrow(obs, 1, 0, 1)
        return super().forward(obs, deterministic)
    
    def action_log_prob(self, obs: PyTorchObs) -> Tuple[Tensor, Tensor]:
        if self.truncate_obs:
            # truncate the observation to only the fist row
            obs = th.narrow(obs, 1, 0, 1)
        return super().action_log_prob(obs)
    
class SubSACPolicy(SACPolicy):
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
        truncate_obs: bool = False,
        sub_policy_name: str = None,
    ):
        self.truncate_obs = truncate_obs
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
        )
        if sub_policy_name:
            self.sub_policy_name = sub_policy_name
        
    def make_actor(self, features_extractor: Optional[BaseFeaturesExtractor] = None) -> Actor:
        actor_kwargs = self._update_features_extractor(self.actor_kwargs, features_extractor)
        actor_kwargs.update({
                "truncate_obs": self.truncate_obs,
            })
        return SubActor(**actor_kwargs).to(self.device)
    
    def _predict(self, observation: PyTorchObs, deterministic: bool = False) -> Tensor:
        return super()._predict(observation.unsqueeze(0), deterministic)