import os
import torch
import argparse
import datetime
import shutil

from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint, RichModelSummary, RichProgressBar

import wandb

from .lightning_model import GNNWrapper
from ..utils.helpers import load_config, create_gnn_model, set_seed

torch.set_float32_matmul_precision('medium')

if __name__ == "__main__":
    
    #TODO use of importlib_resources

    #TODO handle training on cpu
    parser = argparse.ArgumentParser(description='Model config file')
    parser.add_argument('--config', type=str, help='yaml filename which is located at model_configs')
    args = parser.parse_args()

    CONFIG_FILE = args.config
    PATH_TO_CONFIGS = "force_field_models/model_configs"
    if CONFIG_FILE is None:
        raise Exception("No config file found: python -m force_field_models.train.fit_model --config <name of config file located at model_configs>")
    config_path = os.path.join(PATH_TO_CONFIGS, CONFIG_FILE)
    if not os.path.exists(config_path):
        raise Exception(f"No config file found in `{config_path}`")
    config = load_config(config_path=config_path)
    
    model_architecture = config['model_architecture']
    max_epochs = config.get('max_epochs', 50)
    gpu_ids = config['gpu_ids']
    monitor = config['monitor']
    set_seed(config.get('seed', 12345))
    
    CHECKPOINT_DIR = os.path.join("force_field_models/checkpoints",
                                  model_architecture)
    experiment_id = f"{model_architecture}_{config['id']}"
    save_dir = os.path.join(CHECKPOINT_DIR, experiment_id) 
    os.makedirs(save_dir, exist_ok=True)
    shutil.copy(config_path, os.path.join(save_dir, f"{experiment_id}.yaml"))

    checkpoint_callback = ModelCheckpoint(
        monitor=monitor,
        dirpath=save_dir,
        filename=f"{experiment_id}-{{epoch:02d}}-{{valid_MAE:.3f}}",
        save_top_k=3,
        mode='min',
    )
    gnn_model = create_gnn_model(config=config)
    model = GNNWrapper(model=gnn_model, config=config)
    
    wandb_logger = WandbLogger(project='smart-df',
                                name=experiment_id,
                               )
    wandb_logger.experiment.config.update(config)
    
    trainer = Trainer(
        callbacks=[checkpoint_callback, RichModelSummary(), RichProgressBar()]
        , max_epochs=max_epochs
        , accelerator="gpu"
        , devices=gpu_ids
        , logger=wandb_logger
    )
    trainer.fit(model)
    wandb.finish()