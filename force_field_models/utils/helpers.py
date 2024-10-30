import os
import yaml
import random
import numpy as np
import pandas as pd

import torch

from rdkit import Chem
from .. import models


hybridization_types = [Chem.rdchem.HybridizationType.SP3, Chem.rdchem.HybridizationType.SP2, Chem.rdchem.HybridizationType.UNSPECIFIED]
HYBRIDIZATION_DICT = {key: i for i, key in enumerate(hybridization_types)}

def load_config(config_path):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def create_gnn_model(config):
    
    model_name = config['model_architecture']
    model_cls = getattr(models, model_name, None)
    if model_cls is None:
        raise ValueError(f"Unknown model architecture: {model_name}")
    
    return model_cls(config)
    
def copy_and_assign_coordinates(mol: Chem.Mol, coordinates: np.array):
    mol = Chem.Mol(mol)
    conf = mol.GetConformer()
    for i in range(coordinates.shape[0]):
        conf.SetAtomPosition(i, coordinates[i].astype(np.float64))
    return mol


def set_seed(seed_value):
    np.random.seed(seed_value)
    random.seed(seed_value)
    torch.manual_seed(seed_value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)
    torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False #CuDNN benchmark mode
    # is used to automatically find the best algorithm for your hardware configuration.
