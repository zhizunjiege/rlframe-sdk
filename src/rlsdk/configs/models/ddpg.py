from typing import Iterable, List, Literal, Optional, Union

from ..base import ConfigBase


class DDPG(ConfigBase):
    """Deterministic Policy Gradient model config."""

    def __init__(
        self,
        *,
        obs_dim: int,
        act_dim: int,
        hidden_layers_actor: List[int] = [64, 64],
        hidden_layers_critic: List[int] = [64, 64],
        lr_actor=0.0001,
        lr_critic=0.001,
        gamma=0.99,
        tau=0.001,
        buffer_size=1000000,
        batch_size=64,
        noise_type: Literal['ou', 'normal'] = 'ou',
        noise_sigma: Union[float, Iterable[float]] = 0.2,
        noise_theta: Union[float, Iterable[float]] = 0.15,
        noise_dt=0.01,
        noise_max=1.0,
        noise_min=1.0,
        noise_decay=1.0,
        update_after=64,
        update_every=1,
        seed: Optional[int] = None,
    ):
        """Init DDPG config.

        Args:
            obs_dim: Dimension of observation.
            act_dim: Dimension of actions.
            hidden_layers_actor: Units of actor hidden layers.
            hidden_layers_critic: Units of critic hidden layers.
            lr_actor: Learning rate of actor network.
            lr_critic: Learning rate of critic network.
            gamma: Discount factor.
            tau: Soft update factor.
            buffer_size: Maximum size of buffer.
            batch_size: Size of batch.
            noise_type: Type of noise, `ou` or `normal`.
            noise_sigma: Sigma of noise.
            noise_theta: Theta of noise, `ou` only.
            noise_dt: Delta time of noise, `ou` only.
            noise_max: Maximum value of noise.
            noise_min: Minimum value of noise.
            noise_decay: Decay rate of noise.
                Note: Noise decayed exponentially, so always between 0 and 1.
            update_after: Number of env interactions to collect before starting to do gradient descent updates.
                Note: Ensures buffer is full enough for useful updates.
            update_every: Number of env interactions that should elapse between gradient descent updates.
                Note: Regardless of how long you wait between updates, the ratio of env steps to gradient steps is locked to 1.
            seed: Seed for random number generators.
        """
        if obs_dim < 1:
            raise ValueError('obs_dim must be greater than 0')
        if act_dim < 1:
            raise ValueError('act_dim must be greater than 0')
        if len(hidden_layers_actor) < 1:
            raise ValueError('hidden_layers_actor must have at least 1 element')
        if len(hidden_layers_critic) < 1:
            raise ValueError('hidden_layers_critic must have at least 1 element')
        if lr_actor <= 0 or lr_actor > 1:
            raise ValueError('lr_actor must be in (0, 1]')
        if lr_critic <= 0 or lr_critic > 1:
            raise ValueError('lr_critic must be in (0, 1]')
        if gamma <= 0 or gamma >= 1:
            raise ValueError('gamma must be in (0, 1)')
        if tau <= 0 or tau > 1:
            raise ValueError('tau must be in (0, 1]')
        if buffer_size < 1:
            raise ValueError('buffer_size must be greater than 0')
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
        if update_every < 1:
            raise ValueError('update_every must be greater than 0')
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
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.noise_type = noise_type
        self.noise_sigma = noise_sigma
        self.noise_theta = noise_theta
        self.noise_dt = noise_dt
        self.noise_max = noise_max
        self.noise_min = noise_min
        self.noise_decay = noise_decay
        self.update_after = update_after
        self.update_every = update_every
        self.seed = seed
