from typing import List

import numpy as np
from components import docking_components, qip_components, rdkit_components

all_info = {**rdkit_components, **qip_components, **docking_components}


def get_sigmoid_func(sigmoid_range: List[float], decrease: bool = False):
    min_range = np.float64(min(sigmoid_range))
    max_range = np.float64(max(sigmoid_range))
    center = np.float64(0.5 * (min_range + max_range))
    alpha = np.log(0.05 / 0.95) / (center - max_range)

    if decrease:

        def dec_func(x):
            return np.float64(1.0 / (1.0 + np.exp(-alpha * (x - center))))

        return dec_func

    else:

        def inc_func(x):
            return 1.0 - np.float64(1.0 / (1.0 + np.exp(-alpha * (x - center))))

        return inc_func


sigmoid_funcs = [
    get_sigmoid_func(x["range"], x.get("minimize", False)) for x in all_info.values()
]
obj_weights = [x.get("weight", 1) for x in all_info.values()]
