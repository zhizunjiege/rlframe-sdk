# Python 3.8.10
from typing import Any, Dict, List, Union

import numpy as np


def func(
    states: Dict[str, List[Dict[str, Any]]],
    inputs: Union[np.ndarray, Dict[Union[str, int], np.ndarray]],
    actions: Dict[str, List[Dict[str, Any]]],
    outputs: Union[np.ndarray, Dict[Union[str, int], np.ndarray]],
    next_states: Dict[str, List[Dict[str, Any]]],
    next_inputs: Union[np.ndarray, Dict[Union[str, int], np.ndarray]],
    terminated: bool,
    truncated: bool,
    reward: Union[float, Dict[Union[str, int], float]],
) -> Union[float, Dict[Union[str, int], float]]:
    """Calculate the reward for the current step."""
    global caches
    if terminated:
        return reward + 1
    else:
        return reward + 0
