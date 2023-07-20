import numpy as np
import pandas as pd
import torch
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import MolFromSmiles
from rdkit.Chem import rdmolops
from torch_geometric.data import Data
from itertools import permutations
import pickle
from tqdm import tqdm

class Parser(object):
    def __init__(self, useHs=True):
        self.data = pd.read_excel("./data/dipole_moments_10071mols.xlsx")
        self.useHs = useHs
        if self.useHs:
            self.atom_type =  ['H', 'C', 'O', 'N', 'S', 'Cl', 'F', 'Br', 'P', 'I', 'Si', 'B', 'Na', 'Sn', 'Se', 'other']
        else:
            self.atom_type =  ['C', 'O', 'N', 'S', 'Cl', 'F', 'Br', 'P', 'I', 'Si', 'B', 'Na', 'Sn', 'Se', 'other']            

    def parse_dataset(self):
        total_data = []
        def _one_hot(x, allowable_set):
            if x not in allowable_set:
                x = allowable_set[-1]
            temp = list(map(lambda s: x == s, allowable_set))
            return [1 if i else 0 for i in temp]
        hydrogen_donor = Chem.MolFromSmarts("[$([N;!H0;v3,v4&+1]),$([O,S;H1;+0]),n&H1&+0]")
        hydrogen_acceptor = Chem.MolFromSmarts(
            "[$([O,S;H1;v2;!$(*-*=[O,N,P,S])]),$([O,S;H0;v2]),$([O,S;-]),$([N;v3;!$(N-*=[O,N,P,S])]),n&H0&+0,$([o,s;+0;!$([o,s]:n);!$([o,s]:c:n)])]")
        acidic = Chem.MolFromSmarts("[$([C,S](=[O,S,P])-[O;H1,-1])]")
        basic = Chem.MolFromSmarts(
            "[#7;+,$([N;H2&+0][$([C,a]);!$([C,a](=O))]),$([N;H1&+0]([$([C,a]);!$([C,a](=O))])[$([C,a]);!$([C,a](=O))]),$([N;H0&+0]([C;!$(C(=O))])([C;!$(C(=O))])[C;!$(C(=O))])]")

        for kdx in tqdm(range(len(self.data))):
            try:
                name = self.data.iloc[kdx, 0]
                if self.useHs:
                    sdf = Chem.SDMolSupplier("./dipole/{}".format(name), removeHs=False)
                else:
                    sdf = Chem.SDMolSupplier("./dipole/{}".format(name))
                mol = sdf[0]
                scalar_feature = []
                pos = []
                N = mol.GetNumAtoms()
                ring = mol.GetRingInfo()
                hydrogen_donor_match = sum(mol.GetSubstructMatches(hydrogen_donor), ())
                hydrogen_acceptor_match = sum(mol.GetSubstructMatches(hydrogen_acceptor), ())
                acidic_match = sum(mol.GetSubstructMatches(acidic), ())
                basic_match = sum(mol.GetSubstructMatches(basic), ())
                adj = np.array(rdmolops.GetAdjacencyMatrix(mol), dtype=float)
                conf = mol.GetConformer()
                dist = np.array(rdmolops.Get3DDistanceMatrix(mol))
                for atom_idx in range(N):
                    # make atom feature
                    atom = mol.GetAtomWithIdx(atom_idx)
                    atom_feature = []
                    atom_feature += _one_hot(atom.GetSymbol(), self.atom_type)
                    # atom_feature += [atom.GetAtomicNum()]
                    atom_feature += _one_hot(atom.GetDegree(), [0, 1, 2, 3, 4, 5, 6])
                    atom_feature += _one_hot(atom.GetFormalCharge(), [-3, -2, -1, 0, 1, 2, 3])
                    atom_feature += _one_hot(atom.GetImplicitValence(), [0, 1, 2, 3, 4, 5, 6])
                    atom_feature += [atom.GetIsAromatic()]
                    atom_feature += _one_hot(atom.GetHybridization(), [Chem.rdchem.HybridizationType.SP,
                                                                Chem.rdchem.HybridizationType.SP2,
                                                                Chem.rdchem.HybridizationType.SP3,
                                                                Chem.rdchem.HybridizationType.SP3D,
                                                                Chem.rdchem.HybridizationType.SP3D2])
                    atom_feature += [ring.IsAtomInRingOfSize(atom_idx, 3),
                                ring.IsAtomInRingOfSize(atom_idx, 4),
                                ring.IsAtomInRingOfSize(atom_idx, 5),
                                ring.IsAtomInRingOfSize(atom_idx, 6),
                                ring.IsAtomInRingOfSize(atom_idx, 7),
                                ring.IsAtomInRingOfSize(atom_idx, 8)]
                    atom_feature += _one_hot(atom.GetTotalNumHs(), [0, 1, 2, 3, 4])
                    atom_feature += [atom_idx in acidic_match, atom_idx in basic_match]
                    atom_feature += [atom_idx in hydrogen_donor_match, atom_idx in hydrogen_acceptor_match]
                    scalar_feature.append(atom_feature)

                    # get atom position
                    atom_pos = conf.GetAtomPosition(atom_idx)
                    pos += [list(atom_pos)]

                # make vector feature from shape of scalar feature
                # vector_feature = np.tile(np.expand_dims(np.zeros_like(scalar_feature), axis=-1), (1, 1, 3))

                # make edge index for B matrix
                row_B, col_B, edge_attr = [], [], []
                for i in range(N):
                    for j in range(N):
                        bond = mol.GetBondBetweenAtoms(i, j)
                        if bond is not None:
                            row_B += [i]
                            col_B += [j]
                            bond_type = bond.GetBondType()
                            if bond_type == Chem.rdchem.BondType.SINGLE:
                                edge_attr += [1]
                            elif bond_type == Chem.rdchem.BondType.DOUBLE:
                                edge_attr += [2]
                            elif bond_type == Chem.rdchem.BondType.TRIPLE:
                                edge_attr += [3]
                            elif bond_type == Chem.rdchem.BondType.AROMATIC:
                                edge_attr += [1.5]
                # add selfloop for B
                    row_B += [i]
                    col_B += [i]
                    edge_attr += [1]
                  
                # make edge index for A matrix
                row_A_, col_A_ = [], []
                for i in range(N):
                    for j in range(N):
                        if dist[i][j] < 5 and adj[i][j] == 0 and i != j:
                            row_A_ += [i]
                            col_A_ += [j]
                          
                edge_index_A_ = torch.tensor([row_A_, col_A_], dtype=torch.long)          
                edge_index_B = torch.tensor([row_B, col_B], dtype=torch.long)
                edge_attr = torch.tensor(edge_attr, dtype=torch.float)
                y = torch.tensor(float(self.data.iloc[kdx, 1]), dtype=torch.float).view(-1, 1)
                pos = torch.tensor(pos).to(torch.float).to(torch.float)
                scalar = torch.tensor(scalar_feature).to(torch.float)
                data = Data(scalar=scalar, pos=pos, edge_index_A_= edge_index_A_, edge_index_B=edge_index_B, edge_attr=edge_attr, y=y)
                total_data.append(data)
            except:
                pass
        print(len(total_data))
        if self.useHs:
            torch.save(total_data, "./data/10071_dipole_useHs.pt")
        else:
            torch.save(total_data, "./data/10071_dipole.pt")

temp = Parser()
temp.parse_dataset()
