import os
import numpy as np
import pandas as pd

from .molecule import MoleculeData
import torch
from torch_geometric.transforms.point_pair_features import PointPairFeatures

from rdkit import Chem


ALL_BONDS = ["SINGLE", "DOUBLE", "TRIPLE", "AROMATIC", "IONIC", "HYDROGEN", "UNSPECIFIED"]
BOND_TYPE_MAP = {k: v  for k, v in zip(ALL_BONDS, range(len(ALL_BONDS)))}
N_BOND_TYPES = len(ALL_BONDS)

ATOM_TYPE_MAP = {
    "H": 0,
    "C": 1, 
    "N": 2,
    "O": 3,
    "S": 4,
    "Cl": 5,
    "Br": 6,
    "F": 7,
    "<UNK>": 8,
}

VALUES_LIST = [
    -6.02097332e-01, 
    -3.81008055e+01, 
    -5.47375864e+01,
    -7.52024502e+01, 
    -3.98139817e+02, 
    -4.60185141e+02,
    -2.57377264e+03, 
    -9.98234301e+01
]

SELF_ENERGIES_PRED = {
    "H" : -6.02097332e-01,
    "C" : -3.81008055e+01,
    "N" : -5.47375864e+01,
    "O" : -7.52024502e+01,
    "S" : -3.98139817e+02,
    "Cl" : -4.60185141e+02,
    "Br" : -2.57377264e+03,
    "F" : -9.98234301e+01
}

CURRENT_PATH = os.path.dirname(os.path.realpath(__file__))
atom_to_atom_df = pd.read_csv(os.path.join(CURRENT_PATH, "atom_to_atom_bond_map.csv"))
ATOM_TYPE__ATOM_TYPE_MAP = dict(zip(atom_to_atom_df['key'], atom_to_atom_df['value']))


def smiles_to_mol(smiles):
    mol = Chem.MolFromSmiles(smiles)
    mol = Chem.AddHs(mol)
    return mol


def mol_from_mol_file(mol_file):
    mol = Chem.MolFromMolFile(mol_file, removeHs=False)
    return mol


def mol_from_sdf(sdf_file):
    mol = list(Chem.SDMolSupplier(sdf_file, removeHs=False))[0]
    return mol


def get_mol_coordinates(molecule):
    coordinates = []
    for i in range(molecule.GetNumAtoms()):
            positions = molecule.GetConformer().GetAtomPosition(i)
            coordinates.append([positions.x, positions.y, positions.z])
    
    return torch.tensor(coordinates, dtype=torch.float32)


def get_energy_value_from_mol(mol):
    energy = mol.GetPropsAsDict()['energy']
    
    return energy


def get_energy_value(mol_file):
    energy = None
    with open(mol_file, "r") as mol_reader:
        for line in mol_reader:
            if line.startswith("Energy:"):
                energy = float(line.split()[1])
                break
    
    return energy


def remove_self_energies(mol, y):
    sum_elements = 0
    atom_sybols = []

    for atom in mol.GetAtoms():
        try:
            sum_elements += SELF_ENERGIES_PRED[atom.GetSymbol()]
            atom_sybols.append(atom.GetSymbol())
        except:
            raise Exception(f"found atom which have no pre-difined self energy: atom type is {atom.GetSymbol()}")

    scaled_energy = (y - sum_elements) * 627.509 #to be in kcal/mol
    
    if scaled_energy > 5e3:
        raise Exception(f"Too big target value: {scaled_energy}")
    
    return scaled_energy


def get_pairwise_distances(coords):
    """
    Calculate pairwise distances between all pairs of points, preserving gradients.

    Args:
        coords (torch.Tensor): Tensor of shape (N, 3) representing coordinates of N points.

    Returns:
        distances (torch.Tensor): Tensor of shape (N, N) containing pairwise distances.
    """
    # Calculate squared differences using broadcasting
    coord_diff = coords.unsqueeze(1) - coords.unsqueeze(0)  # (N, 1, 3) - (1, N, 3) -> (N, N, 3)
    squared_diff = torch.sum(coord_diff ** 2, dim=-1)  # (N, N)

    # Create a mask to ignore diagonal elements (self-distances)
    mask = torch.eye(coords.size(0), dtype=torch.bool, device=coords.device)  # (N, N)
    squared_diff.masked_fill_(mask, float('inf'))  # Set diagonal to infinity

    # Check for overlapping points
    if (squared_diff < 0.64).any():  # Check for distances less than 0.8 (squared)
        raise Exception("Overlap in a molecule!")

    # Check for 2D coordinates
    z_coords = coords[:, 2]
    if torch.allclose(z_coords, torch.zeros_like(z_coords), atol=1e-1):
        raise Exception("2D coordinates detected!")

    # Calculate distances (square root of squared differences)
    distances = torch.sqrt(squared_diff)

    return distances


def get_edge_index(mol, pos, cutoff):
    pairwise_distances = get_pairwise_distances(coords=pos)
    num_atoms = mol.GetNumAtoms()
    edge_index = [[], []]
    for i in range(num_atoms):
        for j in range(i+1, num_atoms):
            if pairwise_distances[i,j] < cutoff:
                edge_index[0].append(i)
                edge_index[0].append(j)
                
                edge_index[1].append(j)
                edge_index[1].append(i)
    return torch.tensor(edge_index, dtype=torch.int64)


def construct_graph(mol, pos, cutoff):
    x = get_atom_types(mol=mol)
    edge_index = get_edge_index(mol=mol, pos=pos, cutoff=cutoff)
    
    return MoleculeData(x=(x, ), edge_index=edge_index, pos=pos)
    
    
def edge_featurizer(mol, pos, cutoff):
    pairwise_distances = get_pairwise_distances(coords=pos)
    num_atoms = mol.GetNumAtoms()
    bond_type_list = []
    atom_to_atom_list = []

    for i in range(num_atoms):
        for j in range(i+1, num_atoms):
            if pairwise_distances[i, j] < cutoff:
                bond = mol.GetBondBetweenAtoms(i, j)
                bond_type = str(bond.GetBondType()) if bond is not None else 'UNSPECIFIED'

                bond_type_list.append(BOND_TYPE_MAP.get((bond_type), max(BOND_TYPE_MAP.values())))
                bond_type_list.append(BOND_TYPE_MAP.get((bond_type), max(BOND_TYPE_MAP.values())))

                i_symbol = mol.GetAtomWithIdx(i).GetSymbol()         
                j_symbol = mol.GetAtomWithIdx(j).GetSymbol()

                begin_atom_type = i_symbol if i_symbol in ATOM_TYPE_MAP.keys() else '<UNK>'
                end_atom_type = j_symbol if j_symbol in ATOM_TYPE_MAP.keys() else '<UNK>'
                
                atom_to_atom_list.append(ATOM_TYPE__ATOM_TYPE_MAP[f'{begin_atom_type}_{end_atom_type}'])
                atom_to_atom_list.append(ATOM_TYPE__ATOM_TYPE_MAP[f'{end_atom_type}_{begin_atom_type}'])
    
    edge_attr = (
        torch.tensor(atom_to_atom_list, dtype=torch.int64),
        torch.tensor(bond_type_list, dtype=torch.int64)
    )
    
    return edge_attr


def get_atom_types(mol):
    node_features = []
    for atom in mol.GetAtoms():
        node_features.append(ATOM_TYPE_MAP.get(atom.GetSymbol(), max(ATOM_TYPE_MAP.values())))
    node_features = torch.tensor(node_features, dtype=torch.int64).reshape(-1, 1)
    
    return node_features


def bond_featurizer(mol):
    bonds = list(mol.GetBonds())
    edge_index_list = []
    bond_type_list = []
    atom_type_to_atom_type_list = []
    for bond in bonds:
        begin_atom_idx = bond.GetBeginAtomIdx()
        end_atom_idx = bond.GetEndAtomIdx()
        
        edge_index_list.append([begin_atom_idx, end_atom_idx])
        edge_index_list.append([end_atom_idx, begin_atom_idx])

        bond_type_list.append(BOND_TYPE_MAP.get(str(bond.GetBondType()), max(BOND_TYPE_MAP.values())))
        bond_type_list.append(BOND_TYPE_MAP.get(str(bond.GetBondType()), max(BOND_TYPE_MAP.values())))
        
        begin_atom_type = mol.GetAtomWithIdx(begin_atom_idx).GetSymbol()
        end_atom_type = mol.GetAtomWithIdx(end_atom_idx).GetSymbol()
        
        if not begin_atom_type in ATOM_TYPE_MAP.keys():
            begin_atom_type = "<UNK>"
        
        if not end_atom_type in ATOM_TYPE_MAP.keys():
            end_atom_type = "<UNK>"
        
        atom_type_to_atom_type_list.append(ATOM_TYPE__ATOM_TYPE_MAP[f'{begin_atom_type}_{end_atom_type}'])
        atom_type_to_atom_type_list.append(ATOM_TYPE__ATOM_TYPE_MAP[f'{end_atom_type}_{begin_atom_type}'])
        
    return bond_type_list, edge_index_list, atom_type_to_atom_type_list


def get_edge_weights(mol_graph):
    distance_map = get_pairwise_distances(coords=mol_graph.pos)
    index_1 = mol_graph.edge_index[0, :]
    index_2 = mol_graph.edge_index[1, :]
    return 1/distance_map[index_1, index_2]


def get_bond_weights(distance_map, edge_index_of_bonds):
    index_1 = np.array(edge_index_of_bonds)[:, 0]
    index_2 = np.array(edge_index_of_bonds)[:, 1]
    return list(distance_map[index_1, index_2])


def global_edge_featurizer(mol, coords):
    distance_map = get_pairwise_distances(coords=coords)
    bond_features_list, edge_index_list, atom_to_atom_list = bond_featurizer(mol)
    edge_weights = get_bond_weights(distance_map=distance_map, edge_index_of_bonds=edge_index_list)

    for i in range(distance_map.shape[0]):
        for j in range(i + 1, distance_map.shape[1]):
            if i == j:
                continue
            distance_i_j = distance_map[i, j]
            if  distance_i_j <= 4:
                if [i, j] not in edge_index_list:
                    edge_index_list.append([i, j])
                    edge_index_list.append([j, i])

                    bond_features_list.append(BOND_TYPE_MAP['UNSPECIFIED'])
                    bond_features_list.append(BOND_TYPE_MAP['UNSPECIFIED'])

                    edge_weights.append(distance_i_j)
                    edge_weights.append(distance_i_j)

                    begin_atom_type = mol.GetAtomWithIdx(i).GetSymbol()
                    end_atom_type = mol.GetAtomWithIdx(j).GetSymbol()
                    
                    if not begin_atom_type in ATOM_TYPE_MAP.keys():
                        begin_atom_type = "<UNK>"
                    
                    if not end_atom_type in ATOM_TYPE_MAP.keys():
                        end_atom_type = "<UNK>"
                    
                    atom_to_atom_list.append(ATOM_TYPE__ATOM_TYPE_MAP[f'{begin_atom_type}_{end_atom_type}'])
                    atom_to_atom_list.append(ATOM_TYPE__ATOM_TYPE_MAP[f'{end_atom_type}_{begin_atom_type}'])
                                    
    edge_attr = (torch.tensor(atom_to_atom_list, dtype=torch.int64), torch.tensor(bond_features_list, dtype=torch.int64))
    edge_weights = 1/torch.tensor(edge_weights, dtype=torch.float32)
    edge_index = torch.tensor(edge_index_list, dtype=torch.int64).T
    return edge_index, edge_attr, edge_weights


def get_point_pair_features(molecule_data):
    pp_featurizer = PointPairFeatures(cat=False)
    mol_data_with_ppf_fts = pp_featurizer(molecule_data)
    return mol_data_with_ppf_fts



def get_normals_retain_grad(coords, k_neighbors):
    pseudo_normal = torch.zeros_like(coords)
    for coords_i in coords:
        pseudo_normal += 1/k_neighbors * (coords_i - coords) 
    pseudo_normal = torch.nn.functional.normalize(pseudo_normal)
    return pseudo_normal


def get_unimol_input(molecule):
    coordinates = []
    atom_types = []
    for i, atom in enumerate(molecule.GetAtoms()):
        positions = molecule.GetConformer().GetAtomPosition(i)
        coordinates.append([positions.x, positions.y, positions.z])
        atom_types.append(atom.GetSymbol())
    
    assert len(coordinates) == len(atom_types)
    assert len(atom_types) == molecule.GetNumAtoms()

    return atom_types, torch.tensor(coordinates, dtype=torch.float32)


def get_unimol_features(clf, atom_types, coordinates):
    """
    return mol_feature and atomic features for given molecule
    """
    data = {
        'atoms': [atom_types],
        'coordinates': [coordinates],
        'target': [-1]
    }
    out = clf.get_repr(data=data, return_atomic_reprs=True)

    return torch.tensor(out["cls_repr"][0], dtype=torch.float32), torch.tensor(out['atomic_reprs'][0], dtype=torch.float32)
