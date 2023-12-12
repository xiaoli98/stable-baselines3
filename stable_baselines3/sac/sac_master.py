from typing import Any, Dict, List, Optional, Tuple, Type, Union

import numpy as np
import os
import torch as th
from gymnasium import spaces
from torch import nn
from torch.nn import ModuleList

from stable_baselines3.common.distributions import SquashedDiagGaussianDistribution, StateDependentNoiseDistribution
from stable_baselines3.common.policies import BasePolicy, ContinuousCritic, ContinuousCriticPool
from stable_baselines3.common.preprocessing import get_action_dim
from stable_baselines3.common.torch_layers import (
    BaseFeaturesExtractor,
    CombinedExtractor,
    FlattenExtractor,
    NatureCNN,
    create_mlp,
    get_actor_critic_arch,
)
from stable_baselines3.common.noise import OrnsteinUhlenbeckActionNoise
from stable_baselines3.common.type_aliases import PyTorchObs, Schedule
from stable_baselines3.sac.policies import SACPolicy

from stable_baselines3.common.utils import is_vectorized_observation, obs_as_tensor

class SACMasterPolicy(SACPolicy):
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
        load_subpolicies: bool = True,
        sub_policies_path: str = None,
        master_action_space: Optional[spaces.Box] = None,
        weighting_scheme: str = "classic",
    ):
        self.n_subpolicies = master_action_space
        self.env_action_space = action_space
        self.n_actions = np.prod(master_action_space.shape)
        self.weighting_scheme = weighting_scheme
        self.weights_noise = OrnsteinUhlenbeckActionNoise(mean=np.zeros(1), sigma=0.1*np.ones(1), dtype=np.float32)

        # we modify the action space to match the number of subpolicies
        master_ob_dim = np.prod(observation_space.shape) + self.n_actions * np.prod(self.env_action_space.shape)
        self.master_ob_space = spaces.Box(-np.inf, np.inf, [master_ob_dim])
        
        super().__init__(
            self.master_ob_space,
            # observation_space,
            self.n_subpolicies, #action_space
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
        self.subpolicies = th.nn.ModuleList()
        if load_subpolicies:
            assert sub_policies_path is not None, "please specify where to load subpolicies"
            checkpoints = os.listdir(sub_policies_path)
            checkpoints = sorted(checkpoints)
            n_subpolicies = np.prod(master_action_space.shape)
            for i in range(n_subpolicies):
                subpolicy = SACPolicy(observation_space,
                                      action_space,
                                      lr_schedule)
                print(f"loading from: {sub_policies_path+checkpoints[i]}...")
                subpolicy.load_state_dict(th.load(sub_policies_path+checkpoints[i], map_location=th.device("cpu")))
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
            # observation, vectorized_env = self.obs_to_tensor(observation)
            observation = th.as_tensor(observation)
        pool_output = []
        with th.no_grad():
            for subpolicy in self.subpolicies:
                pool_output.append(subpolicy.actor(observation, deterministic))
        return th.stack(pool_output)
    
    def get_weights(self, observation: PyTorchObs, deterministic: bool= False):
        if not th.is_tensor(observation):
            observation = th.as_tensor(observation)
        return self.actor(observation, deterministic)

    def weighted_action(self, actions, weights):
        if not th.is_tensor(weights):
            weights = th.tensor(weights, device=self.device)
        out = weights.unsqueeze(-1) * actions
        out = th.sum(out, dim=0)
        sum = th.sum(weights, dim=0).unsqueeze(-1)
        out = out / sum
        return  out
    
    def _predict(self, observation: PyTorchObs, pool_output, deterministic: bool = False) -> th.Tensor:
        # pool_output = self.get_pool_out(observation, deterministic=True)
        # [BS x NP]
        
        weights = self.actor(observation, deterministic).transpose(0,1)
        if not deterministic:
            noise = th.as_tensor(self.weights_noise(), device=self.device)
            weights = th.clip(weights+noise, -0.999, 1)
        # scale into [0,1] 
        if self.weighting_scheme == "classic":
            weights += 1
            weights /= 2
        elif self.weighting_scheme == "minmax":
            weights -= weights.min()
            weights /= weights.max() 

        out = self.weighted_action(pool_output, weights)
        return out, weights
        
    def predict(
        self,
        observation: Union[np.ndarray, Dict[str, np.ndarray]],
        state: Optional[Tuple[np.ndarray, ...]] = None,
        episode_start: Optional[np.ndarray] = None,
        deterministic: bool = False,
    ) -> Tuple[np.ndarray, Optional[Tuple[np.ndarray, ...]]]:
        """
        Get the policy action from an observation (and optional hidden state).
        Includes sugar-coating to handle different observations (e.g. normalizing images).

        :param observation: the input observation
        :param state: The last hidden states (can be None, used in recurrent policies)
        :param episode_start: The last masks (can be None, used in recurrent policies)
            this correspond to beginning of episodes,
            where the hidden states of the RNN must be reset.
        :param deterministic: Whether or not to return deterministic actions.
        :return: the model's action and the next hidden state
            (used in recurrent policies)
        """
        # Switch to eval mode (this affects batch norm / dropout)
        self.set_training_mode(False)

        # Check for common mistake that the user does not mix Gym/VecEnv API
        # Tuple obs are not supported by SB3, so we can safely do that check
        if isinstance(observation, tuple) and len(observation) == 2 and isinstance(observation[1], dict):
            raise ValueError(
                "You have passed a tuple to the predict() function instead of a Numpy array or a Dict. "
                "You are probably mixing Gym API with SB3 VecEnv API: `obs, info = env.reset()` (Gym) "
                "vs `obs = vec_env.reset()` (SB3 VecEnv). "
                "See related issue https://github.com/DLR-RM/stable-baselines3/issues/1694 "
                "and documentation for more information: https://stable-baselines3.readthedocs.io/en/master/guide/vec_envs.html#vecenv-api-vs-gym-api"
            )
        pool_output = self.get_pool_out(observation, deterministic=True)
        pool_output_cpu = pool_output.cpu().numpy()
        master_ob = observation.reshape(observation.shape[0], -1) #aka flatten on dim=1
        # print(f"master ob before: {master_ob.shape}")
        master_ob = np.concatenate([master_ob, pool_output_cpu.reshape(pool_output_cpu.shape[0], -1)], axis=1)
        # print(f"master_ob after: {master_ob.shape}")
        
        obs_tensor, vectorized_env = self.obs_to_tensor(master_ob)
        # obs_tensor = th.as_tensor(observation)

        with th.no_grad():
            actions, weights = self._predict(obs_tensor, pool_output, deterministic=deterministic)
        # Convert to numpy, and reshape to the original action shape
        actions = actions.cpu().numpy().reshape((-1, *self.env_action_space.shape))  # type: ignore[misc]
        weights = weights.cpu().numpy().reshape((-1, self.n_actions))
        if isinstance(self.action_space, spaces.Box):
            if self.squash_output:
                # Rescale to proper domain when using squashing
                actions = self.unscale_action(actions)  # type: ignore[assignment, arg-type]
            else:
                # Actions could be on arbitrary scale, so clip the actions to avoid
                # out of bound error (e.g. if sampling from a Gaussian distribution)
                actions = np.clip(actions, self.action_space.low, self.action_space.high)  # type: ignore[assignment, arg-type]
                # probably  not needed
                weights = np.clip(weights, self.action_space.low, self.action_space.high ) 
        # Remove batch dimension if needed
        if not vectorized_env:
            assert isinstance(actions, np.ndarray)
            actions = actions.squeeze(axis=0)
            weights = weights.squeeze(axis=0)
        return actions, weights, state  # type: ignore[return-value]
    
    def scale_action(self, action: np.ndarray) -> np.ndarray:
        """
        Rescale the action from [low, high] to [-1, 1]
        (no need for symmetric action space)

        :param action: Action to scale
        :return: Scaled action
        """
        assert isinstance(
            self.action_space, spaces.Box
        ), f"Trying to scale an action using an action space that is not a Box(): {self.action_space}"
        low, high = self.env_action_space.low, self.env_action_space.high
        return 2.0 * ((action - low) / (high - low)) - 1.0

    def unscale_action(self, scaled_action: np.ndarray) -> np.ndarray:
        """
        Rescale the action from [-1, 1] to [low, high]
        (no need for symmetric action space)

        :param scaled_action: Action to un-scale
        """
        assert isinstance(
            self.action_space, spaces.Box
        ), f"Trying to unscale an action using an action space that is not a Box(): {self.action_space}"
        low, high = self.env_action_space.low, self.env_action_space.high
        return low + (0.5 * (scaled_action + 1.0) * (high - low))