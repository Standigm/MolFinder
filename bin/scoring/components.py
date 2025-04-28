from typing import Any, Dict

# RDKit components
rdkit_components: Dict[str, Dict[str, Any]] = {
    "SAS": {"range": [3.5, 5], "minimize": True},
    "QED": {"range": [0.3, 0.7]},
    "AlogP": {"range": [2, 5], "minimize": True},
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
