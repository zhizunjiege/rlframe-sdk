from typing import Dict, List, Optional, Union

import numpy as np

from .base import RLModelBase


class Custom(RLModelBase):

    def __init__(
        self,
        training: bool,
        *,
        obs_dim: int = 4,
        act_num: int = 2,
    ):
        super().__init__(training)

        self.obs_dim = obs_dim
        self.act_num = act_num

    def react(self, states: np.ndarray) -> int:
        return np.random.randint(0, self.act_num)

    def store(
        self,
        states: np.ndarray,
        actions: int,
        next_states: np.ndarray,
        reward: float,
        terminated: bool,
        truncated: bool,
    ):
        ...

    def train(self) -> Dict[str, List[float]]:
        return {}

    def get_weights(self) -> Dict[str, List[np.ndarray]]:
        return {}

    def set_weights(self, weights: Dict[str, List[np.ndarray]]):
        ...

    def get_buffer(self) -> Optional[Dict[str, Union[int, str, Dict[str, np.ndarray]]]]:
        return None

    def set_buffer(self, buffer: Optional[Dict[str, Union[int, str, Dict[str, np.ndarray]]]]):
        ...
