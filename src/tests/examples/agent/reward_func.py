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
) -> Union[float, Dict[Union[str, int], float]]:
    """Calculate the reward for the current step."""
    if terminated:
        return 1.0
    else:
        return 0.0
