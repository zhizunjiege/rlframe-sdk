from typing import Iterable, List, Literal, Optional, Union

from ..base import ConfigBase


class DDPG(ConfigBase):
    """Deterministic Policy Gradient model config."""

    def __init__(
        self,
        *,
        obs_dim: int = 4,
        act_dim: int = 2,
        hidden_layers_actor: List[int] = [64, 64],
        hidden_layers_critic: List[int] = [64, 64],
        lr_actor: float = 0.0001,
        lr_critic: float = 0.001,
        gamma: float = 0.99,
        tau: float = 0.001,
        replay_size: int = 1000000,
        batch_size: int = 64,
        noise_type: Literal['normal', 'ou'] = 'ou',
        noise_sigma: Union[float, Iterable[float]] = 0.2,
        noise_theta: Union[float, Iterable[float]] = 0.15,
        noise_dt: float = 0.01,
        noise_max: float = 1.0,
        noise_min: float = 1.0,
        noise_decay: float = 1.0,
        update_after: int = 64,
        update_online_every: int = 1,
        dtype: str = 'float32',
        seed: Optional[int] = None,
    ):
        """Init config.

        Args:
            obs_dim: Dimension of observation.
            act_dim: Dimension of actions.
            hidden_layers_actor: Units of actor hidden layers.
            hidden_layers_critic: Units of critic hidden layers.
            lr_actor: Learning rate of actor network.
            lr_critic: Learning rate of critic network.
            gamma: Discount factor.
            tau: Soft update factor.
            replay_size: Maximum size of replay buffer.
            batch_size: Size of batch.
            noise_type: Type of noise, `normal` or `ou`.
            noise_sigma: Sigma of noise.
            noise_theta: Theta of noise, `ou` only.
            noise_dt: Delta time of noise, `ou` only.
            noise_max: Maximum value of noise.
            noise_min: Minimum value of noise.
            noise_decay: Decay rate of noise.
                Note: Noise decayed exponentially, so always between 0 and 1.
            update_after: Number of env interactions to collect before starting to do gradient descent updates.
                Note: Ensures replay buffer is full enough for useful updates.
            update_online_every: Number of env interactions that should elapse between gradient descent updates.
                Note: Regardless of how long you wait between updates, the ratio of env steps to gradient steps is locked to 1.
            dtype: Data type of model.
            seed: Seed for random number generators.
        """
        if obs_dim < 1:
            raise ValueError('obs_dim must be greater than 0')
        if act_dim < 2:
            raise ValueError('act_dim must be greater than 1')
        if len(hidden_layers_actor) < 1:
            raise ValueError('hidden_layers_actor must have at least 1 element')
        if len(hidden_layers_critic) < 1:
            raise ValueError('hidden_layers_critic must have at least 1 element')
        if lr_actor <= 0 or lr_actor > 1:
            raise ValueError('lr_actor must be in (0, 1]')
        if lr_critic <= 0 or lr_critic > 1:
            raise ValueError('lr_critic must be in (0, 1]')
        if gamma <= 0 or gamma > 1:
            raise ValueError('gamma must be in (0, 1]')
        if tau <= 0 or tau > 1:
            raise ValueError('tau must be in (0, 1]')
        if replay_size < 1:
            raise ValueError('replay_size must be greater than 0')
        if batch_size < 1:
            raise ValueError('batch_size must be greater than 0')
        if noise_type not in ['normal', 'ou']:
            raise ValueError('noise_type must be `normal` or `ou`')
        if isinstance(noise_sigma, Iterable):
            if len(noise_sigma) != act_dim:
                raise ValueError('noise_sigma must have the same length as act_dim')
            for sigma in noise_sigma:
                if sigma < 0:
                    raise ValueError('noise_sigma must be greater than or equal to 0')
        else:
            if noise_sigma < 0:
                raise ValueError('noise_sigma must be greater than or equal to 0')
        if isinstance(noise_theta, Iterable):
            if len(noise_theta) != act_dim:
                raise ValueError('noise_theta must have the same length as act_dim')
            for theta in noise_theta:
                if theta < 0:
                    raise ValueError('noise_theta must be greater than or equal to 0')
        else:
            if noise_theta < 0:
                raise ValueError('noise_theta must be greater than or equal to 0')
        if noise_dt <= 0:
            raise ValueError('noise_dt must be greater than 0')
        if noise_max < 0 or noise_max > 1:
            raise ValueError('noise_max must be in [0, 1]')
        if noise_min < 0 or noise_min > 1:
            raise ValueError('noise_min must be in [0, 1]')
        if noise_decay <= 0 or noise_decay > 1:
            raise ValueError('noise_decay must be in (0, 1]')
        if update_after < batch_size:
            raise ValueError('update_after must be greater than or equal to batch_size')
        if update_online_every < 1:
            raise ValueError('update_online_every must be greater than 0')
        if dtype not in ['float32', 'float64']:
            raise ValueError('dtype must be float32 or float64')
        if seed is not None and (seed < 0 or seed > 2**32 - 1):
            raise ValueError('seed must be in [0, 2**32 - 1] or None for random seed')

        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.hidden_layers_actor = hidden_layers_actor
        self.hidden_layers_critic = hidden_layers_critic
        self.lr_actor = lr_actor
        self.lr_critic = lr_critic
        self.gamma = gamma
        self.tau = tau
        self.replay_size = replay_size
        self.batch_size = batch_size
        self.noise_type = noise_type
        self.noise_sigma = noise_sigma
        self.noise_theta = noise_theta
        self.noise_dt = noise_dt
        self.noise_max = noise_max
        self.noise_min = noise_min
        self.noise_decay = noise_decay
        self.update_after = update_after
        self.update_online_every = update_online_every
        self.dtype = dtype
        self.seed = seed
