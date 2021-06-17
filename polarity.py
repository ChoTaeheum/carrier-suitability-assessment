from rdkit import Chem
from rdkit.Chem import Descriptors

def Polarity(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if Descriptors.MolLogP(mol) < 0:
        return 3
    else:
        return 2