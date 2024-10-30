import torch
from rdkit import Chem
from typing import List
from torch_geometric.data import Data

class MoleculeData(Data):
    def __init__(self, x: torch.Tensor | None = None, 
                edge_index: torch.Tensor | None = None, edge_attr: torch.Tensor | None = None,
                y: torch.Tensor | None = None, pos: torch.Tensor | None = None, mol_features: torch.Tensor | None = None,
                initial_target: torch.Tensor | None = None, norm: torch.Tensor | None = None, file_name: str | None = None, smiles: str | None = None, **kwargs):
       
        super().__init__(x, edge_index, edge_attr, y, pos, **kwargs)
        self.mol_features= mol_features
        self.norm = norm
        self.file_name = file_name
        self.smiles = smiles
        self.initial_target = initial_target


class Molecule:

    def __init__(self, id: str, mol=None, atoms=None, atom_cords=None) -> None:
        self.id = id
        self.mol: Chem.Mol = mol
        self.smiles: str = Chem.MolToSmiles(mol) if mol else None
        if atoms is None:
            self.atoms: List[str] = self._get_elements_list(self.mol) if mol else None
        else:
            self.atoms: List[str] = atoms
        
        if atom_cords is None:
            self.atom_coords: torch.Tensor = self._get_atom_coords_from_mol(self.mol) if mol else None
        else:
            self.atom_coords: torch.Tensor = atom_cords

    def to(self, device):
        if self.atom_coords is not None:
            self.atoms_coords = self.atoms_coords.to(device)
        
        return self
    
    @classmethod
    def _get_elements_list(cls, mol):
        atoms = []
        for atom in mol.GetAtoms():
            atoms.append(atom.GetSymbol)
        return atoms
    
    @classmethod
    def _get_atom_coords_from_mol(cls, mol):
        mol_conformer = mol.GetConformer()

        atom_count = mol.GetNumAtoms()
        atom_coords = torch.zeros((atom_count, 3))
        for i in range(atom_count):
            positions = mol_conformer.GetAtomPosition(i)
            atom_coords[i] = torch.Tensor([positions.x, positions.y, positions.z])

        return atom_coords
