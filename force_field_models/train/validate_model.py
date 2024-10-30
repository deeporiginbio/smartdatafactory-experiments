import os
import torch
import pandas as pd
import argparse

from pytorch_lightning import Trainer

from .lightning_model import GNNWrapper
from ..utils.helpers import load_config, create_gnn_model


@torch.no_grad()
def validate_model(model, loader, device):
    all_preds = []
    all_targets = []
    all_file_names = []
    for batch in loader:
        batch = batch.to(device)
        preds = model(batch)
        
        all_preds.append(preds)
        all_targets.append(batch.y)

        file_names = batch.file_name
        all_file_names.extend([file.item() for file in file_names])
    
    return all_targets, all_preds, all_file_names
    




if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Model config file')
    parser.add_argument('--config', type=str, help='yaml filename which is located at model_configs')
    args = parser.parse_args()

    CONFIG_FILE = args.config
    PATH_TO_CONFIGS = "force_field_models/model_configs"
    if CONFIG_FILE is None:
        raise Exception("No config file found: python -m force_field_models.train.validate_model --config <name of config file located at model_configs>")
    config_path = os.path.join(PATH_TO_CONFIGS, CONFIG_FILE)
    config = load_config(config_path=config_path)
    
    model_architecture = config['model_architecture']
    gpu_ids = config['gpu_ids']

    CHECKPOINT_DIR = os.path.join("force_field_models/checkpoints",
                                  model_architecture)
    experiment_id = f"{model_architecture}_{config['id']}"
    save_dir = os.path.join(CHECKPOINT_DIR, experiment_id) 

    device = f"cuda:{gpu_ids[0]}"
    gnn_model = create_gnn_model(config=config)
    model = GNNWrapper.load_from_checkpoint(os.path.join(save_dir, config['best_model_name'])
                                            , model=gnn_model, config=config, map_location=device)
    model.eval()

    trainer = Trainer(
        accelerator='gpu'
        , devices=gpu_ids
        # , strategy='fsdp'
    )
    trainer.test(model=model)
    trainer.validate(model=model)
    # test_loader = model.test_dataloader()
    # val_loader = model.val_dataloader()

    # import pdb; pdb.set_trace()
    # targets, preds, file_names = validate_model(model, test_loader, device)
    # targets = torch.cat(targets).reshape(-1).cpu()
    # preds = torch.cat(preds).cpu().reshape(-1)
    # df = pd.DataFrame({'targets': targets, 'preds': preds, 'file_names': file_names})
    # df.to_csv(f"{config['id']}_test_results.csv", index=False)