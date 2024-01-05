import os
from typing import Any, Dict, List, Optional, Tuple, Type, TypeVar, Union

from gymnasium import spaces
import numpy as np
import torch as th
from torch import Tensor, nn


from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.torch_layers import (
    BaseFeaturesExtractor,
    CombinedExtractor,
    FlattenExtractor,
    MlpExtractor,
    NatureCNN,
    create_mlp,
)
from stable_baselines3.common.type_aliases import PyTorchObs, Schedule

class PPOMaster_Policy(ActorCriticPolicy):
    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        lr_schedule: Schedule,
        net_arch: Optional[Union[List[int], Dict[str, List[int]]]] = None,
        activation_fn: Type[nn.Module] = nn.Tanh,
        ortho_init: bool = True,
        use_sde: bool = False,
        log_std_init: float = 0.0,
        full_std: bool = True,
        use_expln: bool = False,
        squash_output: bool = False,
        features_extractor_class: Type[BaseFeaturesExtractor] = FlattenExtractor,
        features_extractor_kwargs: Optional[Dict[str, Any]] = None,
        share_features_extractor: bool = True,
        normalize_images: bool = True,
        optimizer_class: Type[th.optim.Optimizer] = th.optim.Adam,
        optimizer_kwargs: Optional[Dict[str, Any]] = None,
        
        load_subpolicies: bool = True,
        sub_policies_path: str = None,
        master_action_space: Optional[spaces.Box] = None,
        weighting_scheme: str = "classic",
    ):
        self.n_subpolicies = master_action_space
        self.env_action_space = action_space
        self.n_actions = np.prod(master_action_space.shape)
        self.weighting_scheme = weighting_scheme
        super().__init__(
            observation_space,
            self.n_subpolicies,
            lr_schedule,
            net_arch,
            activation_fn,
            ortho_init,
            use_sde,
            log_std_init,
            full_std,
            use_expln,
            squash_output,
            features_extractor_class,
            features_extractor_kwargs,
            share_features_extractor,
            normalize_images,
            optimizer_class,
            optimizer_kwargs,
        )
        
        self.subpolicies = th.nn.ModuleList()
        if load_subpolicies:
            assert sub_policies_path is not None, "please specify where to load subpolicies"
            checkpoints = os.listdir(sub_policies_path)
            n_subpolicies = np.prod(master_action_space.shape)
            for i in range(n_subpolicies):
                subpolicy = ActorCriticPolicy(observation_space,
                                      action_space,
                                      lr_schedule)
                print(f"loading from: {sub_policies_path+checkpoints[i]}...")
                subpolicy.load_state_dict(th.load(sub_policies_path+checkpoints[i], map_location=th.device('cpu')))
                self.subpolicies.append(subpolicy)
        for params in self.subpolicies.parameters():
            params.requires_grad = False
    
    def get_pool_out(self, observation: PyTorchObs, deterministic: bool =True):
        if isinstance(observation, tuple) and len(observation) == 2 and isinstance(observation[1], dict):
            raise ValueError(
                "You have passed a tuple to the predict() function instead of a Numpy array or a Dict. "
                "You are probably mixing Gym API with SB3 VecEnv API: `obs, info = env.reset()` (Gym) "
                "vs `obs = vec_env.reset()` (SB3 VecEnv). "
                "See related issue https://github.com/DLR-RM/stable-baselines3/issues/1694 "
                "and documentation for more information: https://stable-baselines3.readthedocs.io/en/master/guide/vec_envs.html#vecenv-api-vs-gym-api"
            )
        if not th.is_tensor(observation):
            observation, vectorized_env = self.obs_to_tensor(observation)
        pool_output = []
        with th.no_grad():
            for subpolicy in self.subpolicies:
                pool_output.append(subpolicy.actor(observation, deterministic))
        return th.stack(pool_output)
    
    def weighted_action(self, actions, weights):
        if not th.is_tensor(weights):
            weights = th.tensor(weights, device=self.device)
        out = weights.unsqueeze(-1) * actions
        out = th.sum(out, dim=0)
        out = out / th.sum(weights, dim=0).unsqueeze(-1)
        return  out
    
    def forward(self, obs: Tensor, deterministic: bool = False) -> Tuple[Tensor, Tensor, Tensor]:
        weights, values, log_prob = super().forward(obs, deterministic)
        pool_output = self.get_pool_out(obs, deterministic=True)
        weighted_action = self.weighted_action(pool_output, weights)
        
        return weighted_action, weights, values, log_prob
    
    def _predict(self, observation: PyTorchObs, deterministic: bool = False) -> th.Tensor:
        weights = self.get_distribution(observation).get_actions(deterministic=deterministic)
        pool_output = self.get_pool_out(observation, deterministic=True)
        weighted_action = self.weighted_action(pool_output, weights)
        return weighted_action