if fp_method == "rdkit":

    def _get_fp(x):
        return Chem.RDKFingerprint(x)
elif fp_method == "morgan":

    def _get_fp(x):
        return AllChem.GetMorganFingerprintAsBitVect(x, 2)
elif fp_method == "morgan1024":

    def _get_fp(x):
        return AllChem.GetMorganFingerprintAsBitVect(x, 2, nBits=1024)
elif fp_method == "morgan2048":

    def _get_fp(x):
        return AllChem.GetMorganFingerprintAsBitVect(x, 2, nBits=2048)


def get_fp(mol_or_smi):
    if isinstance(mol_or_smi, (Chem.rdchem.Mol, Chem.rdchem.RWMol)):
        _mol = mol_or_smi
    elif isinstance(mol_or_smi, str):
        _mol = Chem.MolFromSmiles(mol_or_smi)
    else:
        raise ValueError("This type is not allowed.")
    return _get_fp(_mol)
