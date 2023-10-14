from typing import List, Literal, Optional, Union

from ..base import ConfigBase


class PPO(ConfigBase):
    """Proximal Policy Optimization model config."""

    def __init__(
        self,
        *,
        policy: Literal['discrete', 'continuous', 'multi-discrete', 'hybrid'],
        obs_dim: int,
        act_dim: Union[int, List[int], List[List[int]]],
        hidden_layers_pi: List[int] = [64, 64],
        hidden_layers_vf: List[int] = [64, 64],
        lr_pi=0.0003,
        lr_vf=0.001,
        gamma=0.99,
        lam=0.97,
        epsilon=0.2,
        buffer_size=4000,
        update_pi_iter=80,
        update_vf_iter=80,
        max_kl=0.01,
        seed: Optional[int] = None,
    ):
        """Init PPO config.

        Args:
            policy: Type of Policy network.
                Note: one of `discrete`, `continuous`, `multi-discrete` and `hybrid`.
            obs_dim: Dimension of observation.
            act_dim: Dimension of actions.
                Note: it should be a list if policy is `multi-discrete` or a list of list if policy is `hybrid`.
            hidden_layers_pi: Units of hidden layers for policy network.
            hidden_layers_vf: Units of hidden layers for value network.
            lr_pi: Learning rate for policy network.
            lr_vf: Learning rate for value network.
            gamma: Discount factor.
            lam: Lambda for Generalized Advantage Estimation.
            epsilon: Clip ratio for PPO-cilp version.
            buffer_size: Size of buffer.
            update_pi_iter: Number of iterations for updating policy network.
            update_vf_iter: Number of iterations for updating value network.
            max_kl: Maximum value of kl divergence.
            seed: Seed for random number generators.
        """
        if policy not in ['discrete', 'continuous', 'multi-discrete', 'hybrid']:
            raise ValueError('policy must be one of `discrete`, `continuous`, `multi-discrete` and `hybrid`')
        if obs_dim < 1:
            raise ValueError('obs_dim must be greater than 0')

        if policy == 'discrete' or policy == 'continuous':
            if not isinstance(act_dim, int) or act_dim < 1:
                raise ValueError('act_dim must be greater than 0 and int if policy is `discrete` or `continuous`')
        elif policy == 'multi-discrete':
            if not isinstance(act_dim, list) or len(act_dim) < 1:
                raise ValueError('act_dim must be a list and have at least 1 element if policy is `multi-discrete`')
        elif policy == 'hybrid':
            valid = True
            if not isinstance(act_dim, list) or len(act_dim) < 1:
                valid = False
            else:
                n = len(act_dim[0])
                for i in range(len(act_dim)):
                    if not isinstance(act_dim[i], list) or len(act_dim[i]) < 1 or len(act_dim[i]) != n:
                        valid = False
                        break
            if not valid:
                raise ValueError('act_dim must be a list of list and have shape (m, n) if policy is `hybrid`')

        if len(hidden_layers_pi) < 1:
            raise ValueError('hidden_layers_pi must have at least 1 element')
        if len(hidden_layers_vf) < 1:
            raise ValueError('hidden_layers_vf must have at least 1 element')
        if lr_pi <= 0:
            raise ValueError('lr_pi must be greater than 0')
        if lr_vf <= 0:
            raise ValueError('lr_vf must be greater than 0')
        if gamma <= 0 or gamma >= 1:
            raise ValueError('gamma must be in (0, 1)')
        if lam <= 0 or lam >= 1:
            raise ValueError('lam must be in (0, 1)')
        if epsilon <= 0 or epsilon >= 1:
            raise ValueError('epsilon must be in (0, 1)')
        if buffer_size < 1:
            raise ValueError('buffer_size must be greater than 0')
        if update_pi_iter < 1:
            raise ValueError('update_pi_iter must be greater than 0')
        if update_vf_iter < 1:
            raise ValueError('update_vf_iter must be greater than 0')
        if max_kl <= 0:
            raise ValueError('max_kl must be greater than 0')
        if seed is not None and (seed < 0 or seed > 2**32 - 1):
            raise ValueError('seed must be in [0, 2**32 - 1] or None for random seed')

        self.policy = policy
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.hidden_layers_pi = hidden_layers_pi
        self.hidden_layers_vf = hidden_layers_vf
        self.lr_pi = lr_pi
        self.lr_vf = lr_vf
        self.gamma = gamma
        self.lam = lam
        self.epsilon = epsilon
        self.buffer_size = buffer_size
        self.update_pi_iter = update_pi_iter
        self.update_vf_iter = update_vf_iter
        self.max_kl = max_kl
        self.seed = seed
