import torch
import logging
from typing import Any

from torch_geometric.loader.dataloader import DataLoader

import pytorch_lightning as pl
from pytorch_lightning import Trainer

from pytorch_lightning.utilities.memory import garbage_collection_cuda
from pytorch_lightning.callbacks import RichProgressBar
from pytorch_lightning.callbacks.progress.rich_progress import RichProgressBarTheme
from torchmetrics.regression import MeanSquaredError, MeanAbsolutePercentageError, MeanAbsoluteError

from ..data import data

class GNNWrapper(pl.LightningModule):
    def __init__(self, model, config=None, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.model = model
        self.config_yaml = config
        self.is_valid_loader_loaded = False

        self.criterion = getattr(torch.nn, config['criterion'])()
        self.batch_size = config.get('batch_size', 16)

        self.rmse = MeanSquaredError(squared=False)
        self.mae = MeanAbsoluteError()
        self.mape = MeanAbsolutePercentageError()

    def on_fit_start(self) -> None:
        if self.on_gpu:
            self.model = self.model.cuda()

    def forward(self, data, inference=False):
        return self.model(data, inference=inference)
    
    def run_step(self, batch, subset):
        batch = batch.to(self.device)
        self.remove_self_energies(batch)
        target = batch.y
        del batch.y

        preds = self(batch).reshape(-1)
        
        loss = self.criterion(preds, target)
        rmse_value = self.rmse(preds=preds, target=target)
        mae_value = self.mae(preds=preds, target=target)
        mape_value = self.mape(preds=preds, target=target)
        
        self.log(f"{subset}_criterion", loss, prog_bar=True, batch_size=self.batch_size, sync_dist=True)
        self.log(f"{subset}_MAE", mae_value, prog_bar=True, batch_size=self.batch_size, sync_dist=True)
        self.log(f"{subset} RMSE", rmse_value, prog_bar=True, batch_size=self.batch_size, sync_dist=True)
        self.log(f"{subset} MAPE", mape_value, prog_bar=True, batch_size=self.batch_size, sync_dist=True)

        return loss

    def remove_self_energies(self, batch):
        device = batch.x[0].device
        ORDERED_ATOM_TYPES = ["H", "C", "N", "O", "S", "Cl", "Br", "F"]
        self_energy_values = torch.tensor(
            [self.config_yaml['SELF_ENERGIES_PRED'][atom_type] for atom_type in ORDERED_ATOM_TYPES]
            , dtype=torch.float32).to(device)
        self_energies = self_energy_values[batch.x[0].reshape(-1)]
        sum_per_node = torch.bincount(batch.batch, weights=self_energies)
        batch.y = (batch.initial_target - sum_per_node) * 627.509
        
    def training_step(self, batch):
        self.model.train()
        loss = self.run_step(batch=batch, subset='train')

        return loss
    
    @torch.no_grad()
    def validation_step(self, batch):
        self.model.eval()
        loss = self.run_step(batch=batch, subset='valid')

        return loss

    @torch.no_grad()
    def test_step(self, batch):
        self.model.eval()
        loss = self.run_step(batch=batch, subset='test')

        return loss


    def configure_optimizers(self):
        
        optimizer_cls = getattr(torch.optim, self.config_yaml['optimizer']['name'])
        optimizer = optimizer_cls(self.parameters()
                                    , **self.config_yaml['optimizer']['params']
                                    )
        lr_scheduler_cls = getattr(torch.optim.lr_scheduler
                               , self.config_yaml['lr_scheduler']['name']
                               )
        lr_scheduler = lr_scheduler_cls(optimizer
                                        , **self.config_yaml['lr_scheduler']['params'])
        lr_scheduler_config = {
            "scheduler": lr_scheduler
            , "interval": "epoch"
            , "monitor": self.config_yaml['monitor']
        }
        
        return {'optimizer': optimizer
                , 'lr_scheduler': lr_scheduler_config
                }
    
    def get_loader(self, path_key, shuffle, num_workers):
        path_to_data = self.config_yaml[path_key].get('path', None)
        if path_to_data is None:
            raise Exception(f'No `{path_key}` key found in config file: {self.config_yaml}')
        
        data_type = self.config_yaml[path_key].get('data_type', None)
        molecule_data = self._load_data(path_to_data, data_type, path_key)
        
        graph_indices = self.config_yaml.get('train_file_ids', None)
        if graph_indices is not None and path_key == 'train':
            molecule_data = self._get_subset(molecule_data, data_type, graph_indices)

        print(25*"*", path_key, "*"*25)
        loader = DataLoader(molecule_data, batch_size=self.batch_size, shuffle=shuffle, num_workers=num_workers, prefetch_factor=3)
        return loader

    def _load_data(self, path_to_data, data_type, path_key):
        if data_type == 'data_list':
            logging.info("LOADED as DATA_LIST")
            molecule_data = torch.load(path_to_data)
        else:
            data_cls = getattr(data, data_type, None)
            if data_cls is None:
                raise Exception(f"No `{data_type}` was found in config")
            molecule_data = data_cls(path_to_data, prefix=path_key)

        return molecule_data
    
    def _get_subset(self, molecule_data, data_type, graph_indices):
        if data_type == 'data_list':
            molecule_data = [molecule_data[i] for i in graph_indices]
        else:
            molecule_data = molecule_data[graph_indices]
        
        return molecule_data
    
    def train_dataloader(self) -> Any:
        loader = self.get_loader(path_key='train_normals'
                                , shuffle=True
                                , num_workers=5
                                )
        return loader
    
    def val_dataloader(self) -> Any:
        loader = self.get_loader(path_key='valid_normals'
                                , shuffle=False
                                , num_workers=5
                                )
        return loader
    
    def test_dataloader(self) -> Any:
        loader = self.get_loader(path_key='test_normals'
                                , shuffle=False
                                , num_workers=3
                                )
        return loader
    
    
class GpuUsageProgressBar(RichProgressBar):
    def __init__(self, refresh_rate: int = 1, leave: bool = False, theme: RichProgressBarTheme = ..., console_kwargs: torch.Dict[str, Any] | None = None) -> None:
        super().__init__(refresh_rate, leave, theme, console_kwargs)

    def get_metrics(self, trainer: Trainer, pl_module: pl.LightningModule) -> torch.Dict[str, int | str | float | torch.Dict[str, float]]:
        items = super().get_metrics(trainer, pl_module)
        if trainer.on_gpu:
            max_memory_allocated = garbage_collection_cuda()
            gpu_usage = f'{max_memory_allocated / (1024 **3):.2f} GB'
            items.append(f"GPU Usage: {gpu_usage}")
        
        return items
