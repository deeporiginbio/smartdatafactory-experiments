import sys

import torch

from ..data.utils import get_energy_value


if __name__ == "__main__":
    src_path = sys.argv[1]
    dst_path = sys.argv[2]

    for key in ['test', 'valid', 'train']:
        data_list = torch.load(f'{src_path}/energy_{key}.pt')
        for data in data_list:
            energy = get_energy_value(data.file_name)
            data.initial_target = energy
            del data.y
            
        torch.save(data_list, f"{dst_path}/energy_{key}_initial_targets.pt")
