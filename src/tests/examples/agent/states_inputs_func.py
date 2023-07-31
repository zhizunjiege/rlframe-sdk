# Python 3.8.10
import math
from typing import Any, Dict, List, Union

import numpy as np


def func(states: Dict[str, List[Dict[str, Any]]]) -> Union[np.ndarray, Dict[Union[str, int], np.ndarray]]:
    """Convert `states` to `inputs` for model inferecing."""
    global caches
    uav, sub = states['example_uav'][0], states['example_sub'][0]
    return np.array([
        (uav['longitude'] - 122.25) / 0.25,
        (uav['latitude'] - 26.75) / 0.25,
        (uav['altitude'] - 1000) / 100.0,
        (uav['speed'] - 200) / 20.0,
        math.sin(uav['azimuth'] / 180 * math.pi),
        math.cos(uav['azimuth'] / 180 * math.pi),
        (sub['longitude'] - 122.75) / 0.25,
        (sub['latitude'] - 26.75) / 0.25,
        (sub['altitude'] - 0) / 100.0,
        (sub['speed'] - 20) / 2.0,
        math.sin(sub['azimuth'] / 180 * math.pi),
        math.cos(sub['azimuth'] / 180 * math.pi),
    ])
