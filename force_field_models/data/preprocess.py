import torch
from rdkit import Chem
from torch_geometric.transforms.center import Center

from .molecule import MoleculeData
from .utils import (mol_from_mol_file, get_atom_types, get_edge_weights, mol_from_sdf,
                    get_energy_value, remove_self_energies, get_mol_coordinates, edge_featurizer, get_energy_value_from_mol,
                    global_edge_featurizer, get_point_pair_features, get_normals_retain_grad, construct_graph)


def preprocess_mol(mol_file, node_featurizer=get_atom_types, edge_featurizer=global_edge_featurizer):

    if not isinstance(mol_file, str):
        raise Exception(f"Preprocess input takes as argument mol files, given: {mol_file}")
    
    max_nodes = 200
    molecule_data = None
    try:
        if mol_file.endswith(".sdf"):
            mol = mol_from_sdf(mol_file)
            y = get_energy_value_from_mol(mol)
        elif mol_file.endswith(".mol"):
            mol = mol_from_mol_file(mol_file=mol_file)
            y = get_energy_value(mol_file=mol_file)
        else:    
            raise Exception("File format not supported")
        
        smiles = Chem.MolToSmiles(Chem.RemoveHs(mol), canonical=True)
        atom_types_encoded = node_featurizer(mol)
        
        coordinates = get_mol_coordinates(mol)

        edge_index, edge_attr, edge_weights = edge_featurizer(mol, coordinates)

        n_atoms = mol.GetNumAtoms()        
        if n_atoms >  max_nodes:
            raise Exception("molecule too big with %i atoms"%n_atoms)
        
        molecule_data = MoleculeData(x=(atom_types_encoded, ), edge_index=edge_index, edge_attr=None,
                                     pos=coordinates, file_name=mol_file, edge_weights=edge_weights,
                                     y=remove_self_energies(mol, y) if y is not None else None,
                                     smiles=smiles, initial_target=y,
                                    )

        molecule_data = Center()(molecule_data)

        normal = get_normals_retain_grad(molecule_data.pos, k_neighbors=10)
        molecule_data.norm = normal

        molecule_data_with_ppf = get_point_pair_features(molecule_data=molecule_data)
        ppf_fts = molecule_data_with_ppf.edge_attr
        molecule_data.edge_attr = (*edge_attr, ppf_fts)
        
        del molecule_data.norm

    except Exception as exception:
        print("\nCouldn't process molecule : {}".format(mol_file), "\nmolecule exception - {}\n\n".format(exception))
        return None

    return molecule_data


def preprocess_inference(mol_file, input_type="mol"):
     
    if not isinstance(mol_file, str):
        raise Exception(f"Preprocess input takes as argument mol files, given: {mol_file}")
    
    mol_graph = None
    try:
        if input_type == "sdf":
            mol = mol_from_sdf(mol_file)
        elif input_type == "mol":
            mol = mol_from_mol_file(mol_file=mol_file)
        else:
            raise Exception("File format not supported. Supported formats are sdf and mol.") 
        coords = get_mol_coordinates(mol)
        mol_graph = construct_graph(mol=mol, pos=coords,cutoff=4)
        mol_graph.file_name = mol_file

    except Exception as exception:
        print("\nCouldn't process molecule : {}".format(mol_file), "\nmolecule exception - {}\n\n".format(exception))

    return mol_graph


def get_graph_edge_fts(mol_graph):
    mol = mol_from_mol_file(mol_file=mol_graph.file_name)

    mol_graph = Center()(mol_graph)
    mol_graph.norm = get_normals_retain_grad(mol_graph.pos, k_neighbors=10)

    
    edge_weights = get_edge_weights(mol_graph)
    atom_to_atom_fts, bond_types = edge_featurizer(mol=mol, pos=mol_graph.pos, cutoff=4)

    molecule_data_with_ppf = get_point_pair_features(molecule_data=mol_graph)
    ppf_fts = molecule_data_with_ppf.edge_attr
    edge_attr = (atom_to_atom_fts, bond_types, ppf_fts)
    edge_weights = edge_weights

    return edge_attr, edge_weights


def featurize_batch(batch):
    molecule_data_list = batch.to_data_list()
    atom_to_atom_list = []
    bond_types_list = []
    ppf_fts_list = []
    edge_weights_list = []
    for mol_graph in molecule_data_list:
        (atom_to_atom_fts, bond_types, ppf_fts), edge_weights = get_graph_edge_fts(mol_graph=mol_graph)

        atom_to_atom_list.append(atom_to_atom_fts)
        bond_types_list.append(bond_types)
        ppf_fts_list.append(ppf_fts)
        edge_weights_list.append(edge_weights)
    
    atom_to_atom = torch.cat(atom_to_atom_list)
    bond_types = torch.cat(bond_types_list)
    ppf_fts = torch.cat(ppf_fts_list)
    edge_weights = torch.cat(edge_weights_list)
    
    batch.edge_weights = edge_weights
    batch.edge_attr = (atom_to_atom, bond_types, ppf_fts)
