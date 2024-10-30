import os
import sys

import numpy as np
from tqdm import tqdm

from ..data.data import EnergyDatasetH5


if __name__ == "__main__":
    
    prefix = sys.argv[1]  # e.g. 'train', 'valid', or 'test'
    src_h5_path = sys.argv[2]
    dst_dir = sys.argv[3]
    
    dataset = EnergyDatasetH5(src_h5_path, prefix=prefix)

    for i, mol_data in tqdm(enumerate(dataset)):
        npz_file_path = os.path.join(dst_dir, f'mol_{prefix}_{i}.npz')
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
