from typing import Any, Dict, List, Union

import numpy as np


def func(outputs: Union[np.ndarray, Dict[Union[str, int], np.ndarray]]) -> Dict[str, List[Dict[str, Any]]]:
    """Convert `outputs` to `actions` for model simulation."""
    return {
        'example_uav': [{
            'azimuth': float(45 * outputs),
        }],
    }
