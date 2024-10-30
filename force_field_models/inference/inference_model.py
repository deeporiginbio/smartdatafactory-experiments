import os
from pathlib import Path

import torch
from os.path import join
from torch_geometric.data import Batch

from ..train.lightning_model import GNNWrapper
from ..data.preprocess import preprocess_inference
from ..utils.helpers import load_config, create_gnn_model


class ForceFieldModel:
    def __init__(self, 
                 config_path: str = "force_field_models/model_configs/GENConv_new_normals.yaml",
                 checkpoint_path: str = 'force_field_models/checkpoints/ConfigurableGNNModel_GENConv_new_normals/ConfigurableGNNModel_GENConv_new_normals-epoch=47-valid_MAE=2.422.ckpt',
                 device: str='cpu'):

        self.device = device
        config = load_config(config_path=config_path)
        gnn_model = create_gnn_model(config=config)
        self.model = GNNWrapper.load_from_checkpoint(checkpoint_path=checkpoint_path
                                                    , map_location=device
                                                    , model=gnn_model
                                                    , config=config
                                                    )
        self.model.eval()
    
    def get_energy_and_forces(self, mol_file: str=None):
        mol_graph = preprocess_inference(mol_file)
        
        batch = Batch.from_data_list([mol_graph])
        batch = batch.to(self.device)
        batch.pos.retain_grad()
        output = self.model(batch, True)
        output.backward(torch.ones_like(output), retain_graph=True)
        derivative = batch.pos.grad
        force = -derivative
        return output.item(), force.numpy()


class EnsembleForceField:
    def __init__(self, model_names=None, device='cpu', model_type='original') -> None:
        if model_type not in ['original', '50k', 'MD']:
            raise ValueError('model_type should be one of the following: `original`, `50k`, `MD` ')
        
        REAL_PATH = Path(os.path.dirname(os.path.realpath(__file__)))
        REAL_PATH = join(REAL_PATH.parent, 'model_configs')
        self.device = device
        if model_names is None:
            model_names = ['GENConv', 'PNAConv', 'TransformerConv', 'GeneralConv', 'ResGatedConv']
        self.model_list = []
        for i in range(len(model_names)):
            self.model_list.append(ForceFieldModel(
                config_path=join(REAL_PATH, model_names[i] + f'_{model_type}' + '.yaml'),
                checkpoint_path=join(REAL_PATH, model_names[i] + f'_{model_type}' + '.ckpt'),
                device=device
            ))

    def _get_energy_and_forces_batch(self, batch):
        energies = []
        forces = []
        for module in self.model_list:
            output = module.model(batch, True)
            output.backward(torch.ones_like(output), retain_graph=True)
            derivative = batch.pos.grad
            batch.pos.grad = None
            force = -derivative
            
            forces.append(force.numpy())
            energies.append(output.item())
        
        return energies, forces
    
    def get_energy_and_forces(self, mol_file: str=None):
        mol_graph = preprocess_inference(mol_file)
        
        batch = Batch.from_data_list([mol_graph])
        batch = batch.to(self.device)
        batch.pos.retain_grad()

        energies, forces = self._get_energy_and_forces_batch(batch)
        
        return energies, forces

