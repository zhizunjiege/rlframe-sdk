from typing import List, Optional

from ..base import ConfigBase


class DQN(ConfigBase):
    """Deep Q-learning Network model config."""

    def __init__(
        self,
        *,
        obs_dim=4,
        act_num=2,
        hidden_layers: List[int] = [64, 64],
        lr=0.001,
        gamma=0.95,
        replay_size=1000000,
        batch_size=32,
        epsilon_max=1.0,
        epsilon_min=0.1,
        epsilon_decay=0.9,
        start_steps=0,
        update_after=32,
        update_online_every=1,
        update_target_every=200,
        dtype='float32',
        seed: Optional[int] = None,
    ):
        """Init config.

        Args:
            obs_dim: Dimension of observation.
            act_num: Number of actions.
            hidden_layers: Units of hidden layers.
            lr: Learning rate.
            gamma: Discount factor.
            replay_size: Maximum size of replay buffer.
            batch_size: Size of batch.
            epsilon_max: Maximum value of epsilon.
            epsilon_min: Minimum value of epsilon.
            epsilon_decay: Decay rate of epsilon.
                Note: Epsilon decayed exponentially, so always between 0 and 1.
            start_steps: Number of steps for uniform-random action selection before running real policy.
                Note: Helps exploration.
            update_after: Number of env interactions to collect before starting to do gradient descent updates.
                Note: Ensures replay buffer is full enough for useful updates.
            update_online_every: Number of env interactions that should elapse between gradient descent updates.
                Note: Regardless of how long you wait between updates, the ratio of env steps to gradient steps is locked to 1.
            update_target_every: Number of gradient updations that should elapse between target network updates.
            dtype: Data type of model.
            seed: Seed for random number generators.
        """
        if obs_dim < 1:
            raise ValueError('obs_dim must be greater than 0')
        if act_num < 2:
            raise ValueError('act_num must be greater than 1')
        if len(hidden_layers) < 1:
            raise ValueError('hidden_layers must have at least 1 element')
        if lr <= 0 or lr > 1:
            raise ValueError('lr must be in (0, 1]')
        if gamma <= 0 or gamma > 1:
            raise ValueError('gamma must be in (0, 1]')
        if replay_size < 1:
            raise ValueError('replay_size must be greater than 0')
        if batch_size < 1:
            raise ValueError('batch_size must be greater than 0')
        if epsilon_max < 0 or epsilon_max > 1:
            raise ValueError('epsilon_max must be in [0, 1]')
        if epsilon_min < 0 or epsilon_min > 1:
            raise ValueError('epsilon_min must be in [0, 1]')
        if epsilon_decay <= 0 or epsilon_decay > 1:
            raise ValueError('epsilon_decay must be in (0, 1]')
        if start_steps < 0:
            raise ValueError('start_steps must be greater than or equal to 0')
        if update_after < batch_size:
            raise ValueError('update_after must be greater than or equal to batch_size')
        if update_online_every < 1:
            raise ValueError('update_online_every must be greater than 0')
        if update_target_every < 1:
            raise ValueError('update_target_every must be greater than 0')
        if dtype not in ['float32', 'float64']:
            raise ValueError('dtype must be float32 or float64')
        if seed is not None and (seed < 0 or seed > 2**32 - 1):
            raise ValueError('seed must be in [0, 2**32 - 1] or None for random seed')

        self.obs_dim = obs_dim
        self.act_num = act_num
        self.hidden_layers = hidden_layers
        self.lr = lr
        self.gamma = gamma
        self.replay_size = replay_size
        self.batch_size = batch_size
        self.epsilon_max = epsilon_max
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.start_steps = start_steps
        self.update_after = update_after
        self.update_online_every = update_online_every
        self.update_target_every = update_target_every
        self.dtype = dtype
        self.seed = seed
