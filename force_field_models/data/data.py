import os
import h5py
import torch
import numpy as np
from tqdm import tqdm
from joblib import Parallel, delayed
from torch_geometric.data import Dataset

from .molecule import MoleculeData


class EnergyDataset(Dataset):
    def __init__(self, root, transform=None, pre_transform=None, pre_filter=None, log: bool = True, prefix: str=""):
        self.prefix = prefix
        super().__init__(root, transform, pre_transform, pre_filter, log)
        
    @property
    def raw_file_names(self):
        return os.listdir(self.raw_dir)
    
    @property
    def processed_file_names(self):
        all_processed_files = os.listdir(self.processed_dir)
        all_processed_files = [f for f in all_processed_files if f.startswith(self.prefix) and f.endswith('.pt')]

        def extract_index(file_name: str) -> int:
            return int(file_name.split('_')[-1].split('.')[0])
        
        sorted_processed_files = sorted(all_processed_files, key=extract_index)
        
        return sorted_processed_files
    
    def process(self):
        n = 0
        def process_file(raw_path):
            molecule_data = self.transform(raw_path)
            return molecule_data
        
        with Parallel(n_jobs=-1) as parallel:
            molecule_data_list = parallel(delayed(process_file)(raw_path) for raw_path in self.raw_paths)
        
        for mol_data in molecule_data_list:
            if mol_data is not None:
                torch.save(mol_data, os.path.join(self.processed_dir, f"{self.prefix}_data_{n}.pt"))
                n += 1
            else:
                continue

    def len(self):
        all_data = os.listdir(self.processed_dir)
        all_processed_files = [f for f in all_data if f.startswith(self.prefix) and f.endswith('.pt')]
        return len(all_processed_files)

    def get(self, idx):
        data_idx = torch.load(os.path.join(self.processed_dir, f"{self.prefix}_data_{idx}.pt"))
        
        return data_idx


class EnergyDatasetChunks(Dataset):
    def __init__(self, root, transform=None, pre_transform=None, pre_filter=None
                , log: bool = True, prefix: str="", chunk_size=10000, device='cpu'):
        self.prefix = prefix
        self.chunk_size = chunk_size
        self.device = device
        super().__init__(root, transform, pre_transform, pre_filter, log)
        
    @property
    def raw_file_names(self):
        return os.listdir(self.raw_dir)
    
    @property
    def processed_file_names(self):
        all_processed_files = os.listdir(self.processed_dir)
        all_processed_files = [f for f in all_processed_files if f.startswith(self.prefix) and f.endswith('.pt')]

        def extract_index(file_name: str) -> int:
            return int(file_name.split('_')[-1].split('.')[0])
        
        sorted_processed_files = sorted(all_processed_files, key=extract_index)
        
        return sorted_processed_files
    
    def process(self):
        def process_file(raw_path):
            molecule_data = self.transform(raw_path)
            return molecule_data
        
        with Parallel(n_jobs=-1) as parallel:
            molecule_data_list = parallel(delayed(process_file)(raw_path) for raw_path in self.raw_paths)
        
        molecule_data_list = [mol_data for mol_data in molecule_data_list if mol_data is not None]
        
        for i in range(0, len(molecule_data_list), self.chunk_size):
            chunk = molecule_data_list[i: i + self.chunk_size]
            torch.save(chunk, chunk, os.path.join(self.processed_dir, f"energy_{self.prefix}_{i//self.chunk_size}.pt"))


    def len(self):
        num_chunks = len([filename for filename in os.listdir(self.processed_dir) if filename.startswith(f"energy_{self.prefix}_")])
        return num_chunks * self.chunk_size

    def get(self, idx):
        chunk_idx = idx // self.chunk_size
        chunk_file = os.path.join(self.processed_dir, f"energy_{self.prefix}_{chunk_idx}.pt")
        chunk_data = torch.load(chunk_file, map_location=self.device)
        
        return chunk_data[idx % self.chunk_size]


class EnergyDatasetH5(Dataset):
    def __init__(self, root, transform=None, pre_transform=None, pre_filter=None
                , log: bool = True, prefix: str=""):
        self.prefix = prefix
        super().__init__(root, transform, pre_transform, pre_filter, log)
        self.dataset = h5py.File(os.path.join(self.processed_dir, f'energy_{self.prefix}.h5'), mode='r')
        
    @property
    def raw_file_names(self):
        return os.listdir(self.raw_dir)
    
    @property
    def processed_file_names(self):
        return [f'energy_{self.prefix}.h5']
    
    def process(self):
        def process_file(raw_path):
            molecule_data = self.transform(raw_path)
            return molecule_data
        
        processed_path = os.path.join(self.processed_dir, f'energy_{self.prefix}.h5')
        with Parallel(n_jobs=-1) as parallel:
            molecule_data_list = parallel(delayed(process_file)(raw_path) for raw_path in self.raw_paths)
        
        molecule_data_list = [mol_data for mol_data in molecule_data_list if mol_data is not None]

        with h5py.File(processed_path, 'w') as f:
            
            for i, mol_data in enumerate(molecule_data_list):
                group = f.create_group(str(i))
                group.create_dataset('x', data=mol_data.x[0])
                group.create_dataset('edge_index', data=mol_data.edge_index)
                
                group.create_dataset('edge_attr_0', data=mol_data.edge_attr[0])
                group.create_dataset('edge_attr_1', data=mol_data.edge_attr[1])
                group.create_dataset('edge_attr_2', data=mol_data.edge_attr[2])
                group.create_dataset('edge_weights', data=mol_data.edge_weights)

                group.create_dataset('initial_target', data=mol_data.initial_target)
                group.create_dataset('pos', data=mol_data.pos)

                group.create_dataset('smiles', data=mol_data.smiles)                
                group.create_dataset('file_name', data=mol_data.file_name)

                
    def len(self):
        return len(self.dataset)

    def get(self, idx):
        group = self.dataset[str(idx)]
        
        mol_idx = MoleculeData(
            x=(torch.tensor(group['x'][()], dtype=torch.int64), ),
            edge_index=torch.tensor(group['edge_index'][()], dtype=torch.int64),
            edge_attr=(
                torch.tensor(group['edge_attr_0'][()], dtype=torch.int64),
                torch.tensor(group['edge_attr_1'][()], dtype=torch.int64),
                torch.tensor(group['edge_attr_2'][()], dtype=torch.float32)
            ),
            edge_weights=torch.tensor(group['edge_weights'][()], dtype=torch.float32),
            initial_target=torch.tensor(group['initial_target'][()], dtype=torch.float32),
            pos=torch.tensor(group['pos'][()], dtype=torch.float32),
            file_name=group['file_name'][()],
            smiles=group['smiles'][()]
        )

        return mol_idx


class EnergyDatasetChunksH5(Dataset):

    def __init__(self, root, transform=None, pre_transform=None, pre_filter=None,
                log: bool = True, prefix: str="", chunk_size=10000):
        self.prefix = prefix
        self.chunk_size = chunk_size
        super().__init__(root, transform, pre_transform, pre_filter, log)
        num_chunks = len([filename for filename in os.listdir(self.processed_dir) if filename.startswith(f"energy_{self.prefix}_")])
        chunk_paths = [os.path.join(self.processed_dir, f'energy_{self.prefix}_{n}.h5') for n in range(num_chunks)]
        self.dataset_list = {str(i): h5py.File(chunk_paths[i], mode='r') for i in range(num_chunks)}

    @property
    def raw_file_names(self):
        return os.listdir(self.raw_dir)
    
    @property
    def processed_file_names(self):
        all_processed_files = os.listdir(self.processed_dir)
        all_processed_files = [f for f in all_processed_files if f.startswith(self.prefix) and f.endswith('.pt')]

        def extract_index(file_name: str) -> int:
            return int(file_name.split('_')[-1].split('.')[0])
        
        sorted_processed_files = sorted(all_processed_files, key=extract_index)
        
        return sorted_processed_files
    
    def process(self):
        def process_file(raw_path):
            molecule_data = self.transform(raw_path)
            return molecule_data
        
        with Parallel(n_jobs=-1) as parallel:
            molecule_data_list = parallel(delayed(process_file)(raw_path) for raw_path in self.raw_paths)
        
        molecule_data_list = [mol_data for mol_data in molecule_data_list if mol_data is not None]
        n = 0
        for i in range(0, len(molecule_data_list), self.chunk_size):
            chunk = molecule_data_list[i: i + self.chunk_size]
            processed_path = os.path.join(self.processed_dir, f'energy_{self.prefix}_{n}.h5')
            with h5py.File(processed_path, 'w') as f:
                for j, mol_data in enumerate(chunk):
                    group = f.create_group(str(j))
                    group.create_dataset('x_pos', data=torch.cat((mol_data.x[0], mol_data.pos), axis=1))
                    edge_fts = torch.cat(
                        (
                            mol_data.edge_index.T,
                            mol_data.edge_weights.unsqueeze(1),
                            mol_data.edge_attr[0].unsqueeze(1),
                            mol_data.edge_attr[1].unsqueeze(1),
                            mol_data.edge_attr[2]
                        ), axis=1)
                    
                    group.create_dataset('ei_ew_ea', data=edge_fts)
                    group.create_dataset('initial_target', data=mol_data.initial_target)
                    group.create_dataset('Smi_FileName', data=mol_data.smiles +'{}'+ mol_data.file_name)
            n += 1


    def len(self):
        dataset_element_length = np.sum([len(self.dataset_list[str(i)]) for i in range(len(self.dataset_list))])

        return dataset_element_length

    def get(self, idx):
        chunk_idx = idx // self.chunk_size
        group = self.dataset_list[str(chunk_idx)][str(idx % self.chunk_size)]
        edge_fts = group['ei_ew_ea'][()]
        x_pos = group['x_pos'][()]
        smi, file_name = str(group['Smi_FileName'][()]).split('{}')

        mol_idx = MoleculeData(
            x=(torch.tensor(x_pos[:, 0], dtype=torch.int64), ),
            edge_index=torch.tensor(edge_fts[:, :2], dtype=torch.int64).T,
            edge_attr=(
                torch.tensor(edge_fts[:, 2], dtype=torch.int64),
                torch.tensor(edge_fts[:, 3], dtype=torch.int64),
                torch.tensor(edge_fts[:, 4:], dtype=torch.float32)
            ),
            edge_weights=torch.tensor(edge_fts[:, 2], dtype=torch.float32),
            initial_target=torch.tensor(group['initial_target'][()], dtype=torch.float32),
            pos=torch.tensor(x_pos[:, 1:], dtype=torch.float32),
            file_name=file_name,
            smiles=smi
        )
        return mol_idx
    

class EnergyDatasetNPZ(Dataset):

    def __init__(self, root, transform=None, pre_transform=None, pre_filter=None, log: bool = True, prefix: str=""):
        self.prefix = prefix
        super().__init__(root, transform, pre_transform, pre_filter, log)
        self.processed_files = self.processed_file_names
    
    @property
    def raw_file_names(self):
        return os.listdir(self.raw_dir)
    
    @property
    def processed_file_names(self):
        return [f for f in os.listdir(self.processed_dir) if f.endswith('.npz')]
    
    def process(self):
        def process_file(raw_path):
            molecule_data = self.transform(raw_path)
            return molecule_data
        
        with Parallel(n_jobs=-1) as parallel:
            molecule_data_list = parallel(delayed(process_file)(raw_path) for raw_path in self.raw_paths)
        
        molecule_data_list = [mol_data for mol_data in molecule_data_list if mol_data is not None]

        for i, mol_data in enumerate(molecule_data_list):
            npz_file_path = os.path.join(self.processed_dir, f'molecule_{self.prefix}_{i}.npz')
            np.savez_compressed(
                npz_file_path,
                x=mol_data.x[0],
                edge_index=mol_data.edge_index,
                edge_attr_0=mol_data.edge_attr[0],
                edge_attr_1=mol_data.edge_attr[1],
                edge_attr_2=mol_data.edge_attr[2],
                edge_weights=mol_data.edge_weights,
                y=mol_data.y,
                pos=mol_data.pos,
                smiles=mol_data.smiles,
                file_name=mol_data.file_name,
                initial_target=mol_data.initial_target,
            )
    
    def len(self):
        return len(self.processed_files)

    def get(self, idx):
        npz_file_path = os.path.join(self.processed_dir, self.processed_files[idx])
        data = np.load(npz_file_path)
        mol_idx = MoleculeData(
            x=(torch.tensor(data['x'], dtype=torch.int64), ),
            edge_index=torch.tensor(data['edge_index'], dtype=torch.int64),
            edge_attr=(
                torch.tensor(data['edge_attr_0'], dtype=torch.int64),
                torch.tensor(data['edge_attr_1'], dtype=torch.int64),
                torch.tensor(data['edge_attr_2'], dtype=torch.float32)
            ),
            edge_weights=torch.tensor(data['edge_weights'], dtype=torch.float32),
            y=torch.tensor(data['y'], dtype=torch.float32),
            pos=torch.tensor(data['pos'], dtype=torch.float32),
            file_name=data['file_name'],
            smiles=data['smiles'],
            initial_target=torch.tensor(data['initial_target'], dtype=torch.float32)
        )
        return mol_idx


class EnergyDatasetChunksNPZ(Dataset):
    def __init__(self, root, transform=None, pre_transform=None, pre_filter=None, log: bool = True, prefix: str="", chunk_size=10000):
        self.prefix = prefix
        self.chunk_size = chunk_size
        super().__init__(root, transform, pre_transform, pre_filter, log)
        self.processed_files = self.processed_file_names
    
    @property
    def raw_file_names(self):
        return os.listdir(self.raw_dir)
    
    @property
    def processed_file_names(self):
        return [f for f in os.listdir(self.processed_dir) if f.endswith('.npz')]
    
    def process(self):
        def process_file(raw_path):
            molecule_data = self.transform(raw_path)
            return molecule_data
        
        with Parallel(n_jobs=-1) as parallel:
            molecule_data_list = parallel(delayed(process_file)(raw_path) for raw_path in self.raw_paths)
        
        molecule_data_list = [mol_data for mol_data in molecule_data_list if mol_data is not None]

        for i, mol_data in enumerate(molecule_data_list):
            npz_file_path = os.path.join(self.processed_dir, f'mol_{self.prefix}_{i}.npz')
            np.savez(
                npz_file_path,
                x=mol_data.x[0],
                edge_index=mol_data.edge_index,
                edge_attr_0=mol_data.edge_attr[0],
                edge_attr_1=mol_data.edge_attr[1],
                edge_attr_2=mol_data.edge_attr[2],
                edge_weights=mol_data.edge_weights,
                initial_target=mol_data.initial_target,
                pos=mol_data.pos,
                smiles=mol_data.smiles,
                file_name=mol_data.file_name
            )
    
    def len(self):
        return len(self.processed_files)

    def get(self, idx):
        npz_file_path = os.path.join(self.processed_dir, f'mol_{self.prefix}_{idx}.npz')
        data = np.load(npz_file_path)

        mol_idx = MoleculeData(
            x=(torch.tensor(data['x'], dtype=torch.int64), ),
            edge_index=torch.tensor(data['edge_index'], dtype=torch.int64),
            edge_attr=(
                torch.tensor(data['edge_attr_0'], dtype=torch.int64),
                torch.tensor(data['edge_attr_1'], dtype=torch.int64),
                torch.tensor(data['edge_attr_2'], dtype=torch.float32)
            ),
            edge_weights=torch.tensor(data['edge_weights'], dtype=torch.float32),
            initial_target=torch.tensor(data['initial_target'], dtype=torch.float32),
            pos=torch.tensor(data['pos'], dtype=torch.float32),
            file_name=data['file_name'],
            smiles=data['smiles']
        )
        return mol_idx
    
