import os
import argparse
import pandas as pd
import torch
import time
import sys
from torch_geometric.loader import DataLoader

from . import *
from ..utils.helpers import load_config, create_gnn_model
from ..train.lightning_model import GNNWrapper
from ..data.data import EnergyDataset, EnergyDatasetH5, EnergyDatasetChunksH5


VALUES_LIST = [-6.02097332e-01, -3.81008055e+01, -5.47375864e+01,
               -7.52024502e+01, -3.98139817e+02, -4.60185141e+02,
                -2.57377264e+03, -9.98234301e+01,]


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Model config file')
    parser.add_argument('--config', type=str, help='yaml filename which is located at model_configs')
    args = parser.parse_args()

    CONFIG_FILE = args.config
    PATH_TO_CONFIGS = "/home/abughdaryan/smart_data_factory_experiments/force_field_models/model_configs"
    if CONFIG_FILE is None:
        raise Exception("Syntax is the following: python -m force_field_models.models.test_run --config <name of config file located>")
    config_path = os.path.join(PATH_TO_CONFIGS, CONFIG_FILE)
    
    config = load_config(config_path=config_path)
    import pdb; pdb.set_trace()
    # data = EnergyDatasetH5(config['train']['path'], prefix='train')
    data = EnergyDatasetChunksH5("/home/shared/smart-df/EnergyDatasetChunksH5/train", prefix='train')
    loader = DataLoader(data, batch_size=32, num_workers=6, prefetch_factor=3)
    
    model = create_gnn_model(config=config)

    for batch in loader:
        start_time = time.time()
        out = model(batch=batch)
        print(time.time() - start_time)
        print(out)



    # parser = argparse.ArgumentParser(description='Model config file')
    # parser.add_argument('--config', type=str, help='yaml filename which is located at model_configs')
    # args = parser.parse_args()

    # CONFIG_FILE = args.config
    # PATH_TO_CONFIGS = "/home/abughdaryan/smart_data_factory_experiments/force_field_models/model_configs"
    # if CONFIG_FILE is None:
    #     raise Exception("No config file found: python -m force_field_models.models.test_run --config <name of config file located at model_configs>")
    # config_path = os.path.join(PATH_TO_CONFIGS, CONFIG_FILE)
    
    # config = load_config(config_path=config_path)
    # data, _ = torch.load("/home/abughdaryan/smart_data_factory_experiments/data/QM9/processed/energy_qm9_set.pt")
    # loader = DataLoader(data, batch_size=1)
    # model_architecture = config['model_architecture']
    # gpu_ids = config['gpu_ids']
    
    # experiment_id = f"{model_architecture}_{config['id']}"
    # CHECKPOINT_DIR = os.path.join("/home/abughdaryan/smart_data_factory_experiments/force_field_models/checkpoints",
    #                               model_architecture)
    # save_dir = os.path.join(CHECKPOINT_DIR, experiment_id) 

    # gnn_model = create_gnn_model(config=config)
    # model = GNNWrapper.load_from_checkpoint(os.path.join(save_dir, config['best_model_name'])
    #                                         , model=gnn_model, config=config, map_location=f"cuda:{gpu_ids[0]}")
    # model.eval()
    # model = model.to("cuda:0")
    # mol_names = []
    # preds = []
    
    # with torch.no_grad():
    #     for batch in loader:
    #         batch = batch.to("cuda:0")
    #         mol_names.extend(batch.file_name)
    #         out = model(batch).cpu() / 627.509
    #         print(out)
    #         for k in batch.x:
    #             out += VALUES_LIST[int(k[0])]
    #         preds.extend(out)
                
    #     df = pd.DataFrame.from_dict({'mol_names': [f"gdb_{int(mol.split('/')[-1].split('.')[0].split('_')[-1]) + 1}" for mol in mol_names]
    #                             , 'preds': [pred.item() for pred in preds]})
    #     df.to_csv("qm9_preds.csv", index=False)