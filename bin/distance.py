from rdkit.DataStructs import TanimotoSimilarity
from rdkit import Chem
import time
import numpy as np


def _get_fp(x):
    return Chem.RDKFingerprint(x)


def get_fp(mol_or_smi):
    if isinstance(mol_or_smi, (Chem.rdchem.Mol, Chem.rdchem.RWMol)):
        _mol = mol_or_smi
    elif isinstance(mol_or_smi, str):
        _mol = Chem.MolFromSmiles(mol_or_smi)
    else:
        raise ValueError("This type is not allowed.")
    return _get_fp(_mol)


def cal_avg_dist(solutions):
    dist_sum = 0
    min_dist = 10
    max_dist = 0
    _n = len(solutions)

    for i in range(_n - 1):
        for j in range(i + 1, _n):
            fps1 = get_fp(solutions[i, 1])
            fps2 = get_fp(solutions[j, 1])
            dist = TanimotoSimilarity(fps1, fps2)
            dist_sum += dist
            if dist < min_dist:
                min_dist = dist
            if dist > max_dist:
                max_dist = dist

    return dist_sum / (_n * (_n - 1) / 2)  # , min_dist, max_dist


def cal_rnd_avg_dist(solutions, nrnd=400000):
    dist_sum = 0
    min_dist = 10
    max_dist = 0
    tmp_chk = 0

    start_chk = time.time()
    for _ in range(nrnd):
        if _ == 0:
            tmp_chk = start_chk

        mol1, mol2 = np.random.choice(solutions[:, 1], size=2)
        fps1 = get_fp(mol1)
        fps2 = get_fp(mol2)
        dist = TanimotoSimilarity(fps1, fps2)
        dist_sum += dist

        if dist < min_dist:
            min_dist = dist
        if dist > max_dist:
            max_dist = dist
        if _ % int(nrnd / 10) == 0:
            print(
                f"{_ / nrnd * 100:.1f}% complete {(time.time() - tmp_chk) / 60} min/10%\r"
            )
            tmp_chk = time.time()

    print(f"calc. Dist total {(time.time() - start_chk) / 60} min")

    return dist_sum / nrnd  # , min_dist, max_dist


def cal_array_dist(solutions1, solutions2):
    """
    numpy
    :param solutions1:
    :param solutions2:
    :return:
    """

    n1 = len(solutions1)
    n2 = len(solutions2)
    dist = np.zeros([n1, n2])

    for i in range(n1):
        for j in range(n2):
            fps1 = get_fp(solutions1[i, 1])
            fps2 = get_fp(solutions2[j, 1])
            dist[n1, n2] = TanimotoSimilarity(fps1, fps2)

    return dist
