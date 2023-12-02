from typing import Any, Dict, List, Optional, Tuple, Type, Union

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
from stable_baselines3.common.type_aliases import PyTorchObs, Schedule
from stable_baselines3.sac.sac import SACPolicy
from stable_baselines3.sac.policies import Actor
class ActorPool(BasePolicy):
    actor_pool: ModuleList
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
        pool_size: int = 6,
        out_type: str = "mean", #if output is the mean of each subpolicy's outpout
    ):
        super().__init__(
            observation_space,
            action_space,
            features_extractor=features_extractor,
            normalize_images=normalize_images,
            squash_output=True,
        )
        self.pool_size = pool_size
        
        assert  out_type in ["mean", "separate", "single"], "out_type must be ont of ['mean', 'separate', 'single']"
            
        self.out_type = out_type
        self.actor_pool = ModuleList()
        self.action_dim = get_action_dim(self.action_space)
        for i in range(pool_size):
            self.actor_pool.append(Actor(
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
            ))
    
    def _get_constructor_parameters(self) -> Dict[str, Any]:
        data = super()._get_constructor_parameters()

        data.update(
            dict(
                net_arch=self.net_arch,
                features_dim=self.features_dim,
                activation_fn=self.activation_fn,
                use_sde=self.use_sde,
                log_std_init=self.log_std_init,
                full_std=self.full_std,
                use_expln=self.use_expln,
                features_extractor=self.features_extractor,
                clip_mean=self.clip_mean,
                pool_size=self.pool_size,
            )
        )
        return data
    
    def get_std(self) -> th.Tensor:
        std = []
        for idx, actor in enumerate(self.actor_pool):
            msg = f" actor[{idx}]: get_std() is only available when using gSDE"
            assert isinstance(actor.action_dist, StateDependentNoiseDistribution), msg
            std.append(actor.action_dist.get_std(actor.log_std))
        return th.tensor(std)
    
    def reset_noise(self, batch_size: int = 1) -> None:
        """
        Sample new weights for the exploration matrix, when using gSDE.

        :param batch_size:
        """
        for idx, actor in enumerate(self.actor_pool):
            msg = f"actor[{idx}]: reset_noise() is only available when using gSDE"
            assert isinstance(actor.action_dist, StateDependentNoiseDistribution), msg
            actor.action_dist.sample_weights(actor.log_std, batch_size=batch_size)
    
    def forward(self, obs: PyTorchObs, deterministic: bool = False) -> th.Tensor:
        if self.out_type == "mean":
            bs = obs.shape[0]
            actions = th.zeros([bs, self.action_dim], device=self.device)
            for actor in self.actor_pool:
                actions += actor(obs, deterministic)
            return actions/self.pool_size
        elif self.out_type == "single":
            #random actor is selected to perform the rollout
            i = th.randint(0, self.pool_size, [1])
            return self.actor_pool[i](obs, deterministic)

        elif self.out_type == "separate":
            pass
        
    def action_log_prob(self, obs: PyTorchObs) -> Tuple[th.Tensor, th.Tensor]:
        if self.out_type == "mean":
            bs = obs.shape[0]
            actions = th.zeros([bs, self.action_dim], device=self.device)
            log_probs = th.zeros([bs], device=self.device)
            for actor in self.actor_pool:
                action, log_prob = actor.action_log_prob(obs)
                # mean_actions, log_std, kwargs = actor.get_action_dist_params(obs)
                # action, log_prob = actor.action_dist.log_prob_from_params(mean_actions, log_std, **kwargs)
                # print(f"log_prob shape: {log_prob.shape}")
                actions += action
                log_probs += log_prob
            return actions/self.pool_size, log_probs/self.pool_size
        elif self.out_type == "single":
            bs = obs.shape[0]
            actions = th.zeros([self.pool_size, bs, self.action_dim], device=self.device)
            log_probs = th.zeros([self.pool_size, bs], device=self.device)
            for idx, actor in enumerate(self.actor_pool):
                action, log_prob = actor.action_log_prob(obs)
                actions[idx] = action
                log_probs[idx] = log_prob
            return tuple([actions, log_probs])
                
    def _predict(self, observation: PyTorchObs, deterministic: bool = False) -> th.Tensor:
        return self(observation, deterministic)
    
    
class SACPolicyPool(SACPolicy):

    # policyPool:ModuleList
    actor_pool:ActorPool
    critic_pool:ContinuousCriticPool
    critic_target_pool:ContinuousCriticPool
    
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
        pool_size: int = 6,
        out_type: str = "mean",
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
            share_features_extractor
        )
        self.action_dim = get_action_dim(self.action_space)
        self.pool_size = pool_size
        self.actor_kwargs.update({"out_type": out_type})
        self._build(lr_schedule)
                
    def _get_constructor_parameters(self) -> Dict[str, Any]:
        data = super()._get_constructor_parameters()

        data.update(
            dict(pool_size=self.pool_size,)
        )
        return data
    
    def make_actor(self, features_extractor: Optional[BaseFeaturesExtractor] = None) -> Actor:
        actor_kwargs = self._update_features_extractor(self.actor_kwargs, features_extractor)
        return ActorPool(**actor_kwargs).to(self.device)

    def make_critic(self, features_extractor: Optional[BaseFeaturesExtractor] = None) -> ContinuousCritic:
        critic_kwargs = self._update_features_extractor(self.critic_kwargs, features_extractor)
        return ContinuousCriticPool(**critic_kwargs).to(self.device)
    
    def forward(self, obs: PyTorchObs, deterministic: bool = False) -> th.Tensor:
        return self._predict(obs, deterministic=deterministic)

    # TODO returns only the mean action, no pool action
    def _predict(self, observation: PyTorchObs, deterministic: bool = False) -> th.Tensor:
        return self.actor._predict(observation, deterministic)
    
    def load(self, path="~/stable_baselines3/checkpoint/sub_policies/"):
        checkpoints = os.listdir(path)
        num_subpolicies = len(checkpoints)
        for i in range(num_subpolicies):
            self.actor[i].load_state_dict(th.load(path+checkpoints[0]))
        
         
MlpPolicy = SACPolicyPool

class CnnPolicy(SACPolicyPool):
    """
    Policy class (with both actor and critic) for SAC.

    :param observation_space: Observation space
    :param action_space: Action space
    :param lr_schedule: Learning rate schedule (could be constant)
    :param net_arch: The specification of the policy and value networks.
    :param activation_fn: Activation function
    :param use_sde: Whether to use State Dependent Exploration or not
    :param log_std_init: Initial value for the log standard deviation
    :param use_expln: Use ``expln()`` function instead of ``exp()`` when using gSDE to ensure
        a positive standard deviation (cf paper). It allows to keep variance
        above zero and prevent it from growing too fast. In practice, ``exp()`` is usually enough.
    :param clip_mean: Clip the mean output when using gSDE to avoid numerical instability.
    :param features_extractor_class: Features extractor to use.
    :param normalize_images: Whether to normalize images or not,
         dividing by 255.0 (True by default)
    :param optimizer_class: The optimizer to use,
        ``th.optim.Adam`` by default
    :param optimizer_kwargs: Additional keyword arguments,
        excluding the learning rate, to pass to the optimizer
    :param n_critics: Number of critic networks to create.
    :param share_features_extractor: Whether to share or not the features extractor
        between the actor and the critic (this saves computation time)
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
        features_extractor_class: Type[BaseFeaturesExtractor] = NatureCNN,
        features_extractor_kwargs: Optional[Dict[str, Any]] = None,
        normalize_images: bool = True,
        optimizer_class: Type[th.optim.Optimizer] = th.optim.Adam,
        optimizer_kwargs: Optional[Dict[str, Any]] = None,
        n_critics: int = 2,
        share_features_extractor: bool = False,
        pool_size: int = 6,
        out_type: str = "mean",
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
            pool_size,
            out_type,
        )