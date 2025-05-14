from typing import List

import numpy as np

from typing import Any, Dict

# RDKit components
rdkit_components: Dict[str, Dict[str, Any]] = {
    "SAS": {"range": [3.5, 5], "minimize": True},
    "QED": {"range": [0.3, 0.7]},
    "ALogP": {"range": [2, 5], "minimize": True},
}

# QIP-ADMET components
qip_components: Dict[str, Dict[str, Any]] = {
    "caco2_wang": {"range": [-7, -5.15]},
    "solubility_aqsoldb": {"range": [-6, -4]},
    "ppbr_az": {"range": [90, 100], "minimize": True},
    "cyp2d6_veith": {"range": [0.3, 0.5]},
    "cyp3a4_veith": {"range": [0.3, 0.5]},
    "cyp2c9_veith": {"range": [0.3, 0.5]},
    "cyp3a4_substrate_carbonmangels": {"range": [0.3, 0.5]},
    "cyp2c9_substrate_carbonmangels": {"range": [0.3, 0.5]},
    "cyp2d6_substrate_carbonmangels": {"range": [0.3, 0.5]},
    "clearance_hepatocyte_az": {"range": [20, 80], "minimize": True},
    "herg": {"range": [0.3, 0.5]},
}

# ROCS/Gold components
docking_components: Dict[str, Dict[str, Any]] = {
    "rocs": {"range": [0.96, 1, 1], "prop": "ROCS_Tanimoto"},
    "gold": {"range": [70, 100], "weight": 4, "prop": "Gold.PLP.Fitness"},
}


all_info = {**rdkit_components, **qip_components, **docking_components}


# TODO: For some reason this does not work in vectorized form
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


sigmoid_funcs = {
    name: get_sigmoid_func(x["range"], x.get("minimize", False))
    for name, x in all_info.items()
}
obj_weights = {name: x.get("weight", 1) for name, x in all_info.items()}
