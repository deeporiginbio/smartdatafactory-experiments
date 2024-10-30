import os
import yaml
import wandb
import torch
import random
import logging
import argparse
import numpy as np
import pandas as pd

from torch_geometric.loader import DataLoader
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint, RichModelSummary, RichProgressBar, EarlyStopping

from .lightning_model import GNNWrapper
from .fit_self_energies import fit_self_energy
from ..utils.helpers import load_config, create_gnn_model, set_seed
from ..data import data

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
parser = argparse.ArgumentParser(description='Model config file')
parser.add_argument('--test_configuration', '-tc', type=str, help='yaml filename which is located at test_configurations')
args = parser.parse_args()

MAXSTEPCOUNT = 5
TEST_CONFIG_NAME = args.test_configuration
TEST_CONFIGURATIONS_DIR = "force_field_models/sdf_experiments/test_configurations"
experiment_config = load_config(os.path.join(TEST_CONFIGURATIONS_DIR, TEST_CONFIG_NAME))


@torch.no_grad()
def write_evaluation_into_csv(model, loader, device, csv_path, indices=None):
    predictions = []
    target_values = []
    
    for batch in loader:
        batch = batch.to(device)
        predictions.extend(model(batch))
        target_values.extend(batch.initial_target)
    
    df = pd.DataFrame.from_dict({
        f"predictions": [pred[0].cpu().numpy() for pred in predictions]
        , "target(hartree)": [target.cpu().numpy() for target in target_values]
        , "index": indices if indices is not None else [-1] * len(predictions)
        })
    df.to_csv(csv_path, index=False)

def get_top_k_based_on_rsd(predictions, top_k):
    std_preds = np.std(predictions, axis=0)
    mean_preds = np.mean(predictions, axis=0)
    rsd_preds = std_preds / mean_preds
    sorted_res_indices = np.argsort(rsd_preds)[::-1]

    return sorted_res_indices[:top_k]

def get_buffer_predictions(path_to_predictions_folder):
    csv_paths = [os.path.join(path_to_predictions_folder, file) for file in os.listdir(path_to_predictions_folder)]
    all_predictions = []
    for csv_file in csv_paths:
        all_predictions.append(pd.read_csv(csv_file).predictions.to_list())
    
    return np.array(all_predictions)

def fit_model(model_config, step_num, trainer_config):
    gnn_model = create_gnn_model(config=model_config)
    model = GNNWrapper(model=gnn_model, config=model_config)            
    early_stopping_callback = EarlyStopping(
                monitor=model_config['monitor']
                , min_delta=0.001
                , patience=25
                , verbose=True
                , mode='min'
            )
            
    trainer = Trainer(
        callbacks=[checkpoint_callback, RichModelSummary(), RichProgressBar(), early_stopping_callback]
        , logger=wandb_logger
        , **trainer_config
    )
    logging.info(f'Start {model_architecture}-{step_num} model fit')
    trainer.fit(model)


def check_if_train_needed(checkpoint_dir, best_model_name, step_num, model_instructions):
    model_checkpoint_dir = f"{os.path.join(checkpoint_dir, best_model_name)}.ckpt"
    file_exists = os.path.isfile(model_checkpoint_dir)
    if not file_exists:
        logging.info(f"No checkpoint found at {model_checkpoint_dir}. Starting training...")
        return True
    
    is_initial_step = step_num == experiment_config.get('step_num', 0)
    train_at_start = model_instructions['continue_train']
    if is_initial_step and not train_at_start:
        logging.info(f"Skipping model training... \nLoading checkpoint from: {checkpoint_dir}/{best_model_name}")
        return False
    else:
        return True

def save_model_config(model_config, path_to_experiment_id, step_num, model_architecture):
    config_file_path = os.path.join(
                path_to_experiment_id
                , f"model_configs/step_{step_num}"
                , f"{model_architecture}_{step_num}.yaml"
                )
    os.makedirs(os.path.join(path_to_experiment_id, f"model_configs/step_{step_num}"), exist_ok=True)
    with open(config_file_path, 'w') as model_config_writer:
        yaml.dump(model_config, model_config_writer)


def write_data_ind_into_csv(indices, folder_path, file_name):
    df = pd.DataFrame.from_dict({"indicies": indices})
    df.to_csv(os.path.join(folder_path, file_name), index=False)


def _load_data(path_to_data, data_type, path_key):
    if data_type == 'data_list':
        logging.info("LOADED as DATA_LIST")
        molecule_data = torch.load(path_to_data)
    else:
        data_cls = getattr(data, data_type, None)
        if data_cls is None:
            raise Exception(f"No `{data_type}` was found in config")
        molecule_data = data_cls(path_to_data, prefix=path_key)

    return molecule_data
    

def _get_subset(molecule_data, data_type, graph_indices):
    if data_type == 'data_list':
        molecule_data = [molecule_data[i] for i in graph_indices]
    elif data_type == 'Dataset':
        molecule_data = molecule_data[graph_indices]
    elif data_type == 'DatasetH5':
        molecule_data = molecule_data[graph_indices]
    elif data_type == 'EnergyDatasetNPZ':
        molecule_data = molecule_data[graph_indices]
    else:
        raise Exception(f"No `{data_type}` was found in config")
    
    return molecule_data
    

if __name__ == '__main__':

    PATH_TO_EXPERIMENT_ID = os.path.join(
        'force_field_models/sdf_experiments'
        , f"experiment_{experiment_config['experiment_id']}"
        )
    step_num = experiment_config.get('step_num', 0)
    os.makedirs(PATH_TO_EXPERIMENT_ID, exist_ok=True)
    DATA_FILE_NAMES = os.path.join(PATH_TO_EXPERIMENT_ID, "data_file_names")
    os.makedirs(DATA_FILE_NAMES, exist_ok=True)

    torch.set_float32_matmul_precision('medium')
    set_seed(experiment_config['seed'])

    dataset = _load_data(path_to_data=experiment_config['train']['path'], path_key=experiment_config['train']['prefix'],
                      data_type=experiment_config['train']['data_type'])

    train_seed_ind = np.random.choice(list(range(len(dataset))), experiment_config['seed_size'], replace=False)
    
    write_data_ind_into_csv(train_seed_ind, DATA_FILE_NAMES, "seed_train_initial.csv")

    sdf_train_ids = np.array(experiment_config.get('initial_sdf_ids', []), dtype=np.int64)
    print("INITIAL TRAIN SET LOADED", len(sdf_train_ids))
    write_data_ind_into_csv(sdf_train_ids, DATA_FILE_NAMES, f'sdf_train_{step_num}.csv')
    
    pool_indices = np.array(list(set(range(len(dataset))) - set(train_seed_ind) - set(sdf_train_ids)), dtype=np.int64)
    
    buffer_examples_ind = np.array(np.random.choice(pool_indices, size=experiment_config['buffer_size'], replace=False)
                                   , dtype=np.int64)

    pool_indices = np.setdiff1d(pool_indices, buffer_examples_ind)
    
    write_data_ind_into_csv(pool_indices, DATA_FILE_NAMES, f"pool_mols_{step_num}.csv")
    
    write_data_ind_into_csv(buffer_examples_ind, DATA_FILE_NAMES, f"buffer_mols_{step_num}.csv")
    del dataset
    
    atom_type_list = ["H", "C", "N", "O", "S", "Cl", "Br", "F"]

    while len(pool_indices) > experiment_config['buffer_step_size']:
        train_seed_ind = np.concatenate((train_seed_ind, sdf_train_ids), dtype=np.int64)

        logging.info(f"STEP NUMBER: {step_num} started")
        logging.info("LOADING train file to fit self energies")
        dataset = _load_data(path_to_data=experiment_config['train']['path'], path_key=experiment_config['train']['prefix'],
                      data_type=experiment_config['train']['data_type'])
        buffer_data = _get_subset(dataset, experiment_config['train']['data_type'], buffer_examples_ind)
        train_data = _get_subset(dataset, experiment_config['train']['data_type'], train_seed_ind)

        write_data_ind_into_csv(train_seed_ind, DATA_FILE_NAMES, f"seed_train_{step_num}.csv")
        
        calculated_self_energies = fit_self_energy(train_data)
        calculated_self_energy_dict = dict(zip(atom_type_list, calculated_self_energies))
        logging.info("Self energies was calculated")

        del train_data
        del dataset

        for model_i in experiment_config['model_ensemble_configs']:
            model_config = load_config(model_i['model_config_path'])
            model_config['train_file_ids'] = train_seed_ind.tolist()
            model_architecture = model_config['id']
            
            model_config['SELF_ENERGIES_PRED'] = calculated_self_energy_dict
            
            CHECKPOINT_DIR = os.path.join(PATH_TO_EXPERIMENT_ID, 'checkpoints', model_architecture)
            os.makedirs(CHECKPOINT_DIR, exist_ok=True)
            best_model_name = f"{model_architecture}-{step_num}-{experiment_config['experiment_id']}"
            checkpoint_callback = ModelCheckpoint(
                  monitor=model_config['monitor']
                , dirpath=CHECKPOINT_DIR
                , filename=best_model_name
                , save_top_k=1
                , mode='min'
            )

            

            cuda_index = model_i['trainer_config'].get('devices', None)
            map_location = f"cuda:{cuda_index[0]}" if cuda_index is not None else 'cpu'
            
            save_model_config(model_config=model_config
                               , path_to_experiment_id=PATH_TO_EXPERIMENT_ID
                               , step_num=step_num
                               , model_architecture=model_architecture)

            is_training_needed = check_if_train_needed(step_num=step_num
                                , checkpoint_dir=CHECKPOINT_DIR
                                , best_model_name=best_model_name
                                , model_instructions=model_i
                                )
            
            if is_training_needed:
                logging.info(f"Start training {best_model_name}")
                wandb_logger = WandbLogger(
                    project=f"smart-df-cycle_{experiment_config['experiment_id']}",
                    name=f"{model_architecture}_{step_num}",
                    )
                wandb_logger.experiment.config.update(model_config)
                fit_model(model_config=model_config, step_num=step_num
                          , trainer_config=model_i['trainer_config'])
                logging.info(f"Finished training {best_model_name}")
                wandb.finish()
            else:
                logging.info(f"Skipping training... loading {os.path.join(CHECKPOINT_DIR, best_model_name)}.ckpt")

            model = GNNWrapper.load_from_checkpoint(
                f"{os.path.join(CHECKPOINT_DIR, best_model_name)}.ckpt"
                , model=create_gnn_model(config=model_config)
                , config=model_config
                , map_location=map_location
                )
            
            model.eval()
            logging.info(f'Start inference on validation set of model {model_architecture}_{step_num}')
            val_laoder = model.val_dataloader()
            valid_csv_folder = os.path.join(
                PATH_TO_EXPERIMENT_ID
                , f"valid_preds/step_{step_num}"
                )
            os.makedirs(valid_csv_folder, exist_ok=True)
            write_evaluation_into_csv(model=model
                                    , loader=val_laoder
                                    , device=map_location
                                    , csv_path=os.path.join(valid_csv_folder, f"preds_{model_architecture}_{step_num}_val.csv")
                                    )
            
            logging.info(f'Start inference on test set of model {model_architecture}_{step_num}')
            test_csv_folder = os.path.join(
                PATH_TO_EXPERIMENT_ID
                , f"test_preds/step_{step_num}"
                )
            test_loader = model.test_dataloader()
            os.makedirs(test_csv_folder, exist_ok=True)
            write_evaluation_into_csv(model=model
                                    , loader=test_loader
                                    , device=map_location
                                    , csv_path=os.path.join(test_csv_folder, f"preds_{model_architecture}_{step_num}_test.csv")
                                    )
            
            logging.info(f'Start inference on buffer set of model {model_architecture}_{step_num}')
            buffer_loader = DataLoader(buffer_data, batch_size=16)
            buffer_csv_folder = os.path.join(
                PATH_TO_EXPERIMENT_ID
                , f"buffer_predictions/step_{step_num}"
                )
            os.makedirs(buffer_csv_folder, exist_ok=True)
            write_evaluation_into_csv(model=model
                                      , loader=buffer_loader
                                      , device=map_location
                                      , csv_path=(os.path.join(buffer_csv_folder, f"preds_{model_architecture}_{step_num}_buffer.csv"))
                                      , indices=buffer_examples_ind)
            
        
        step_num += 1
        logging.info(f"Processing step {step_num}")
        if experiment_config['use_buffer_predictions']:
            logging.info("Using ranking from MODELS")
            
            predictions_on_buffer = get_buffer_predictions(buffer_csv_folder)
            top_k_ind_from_buffer_relative_buffer = get_top_k_based_on_rsd(
                predictions=predictions_on_buffer
                , top_k=experiment_config['buffer_step_size']
                )
        else:
            logging.info("Using RANDOM buffer")
            top_k_ind_from_buffer_relative_buffer = np.array(random.sample(list(range(len(buffer_data))), experiment_config['buffer_step_size']))

        top_k_from_buffer_ind = np.array([buffer_examples_ind[i] for i in top_k_ind_from_buffer_relative_buffer])
        
        write_data_ind_into_csv(top_k_from_buffer_ind, DATA_FILE_NAMES, f"sdf_train_ids_{step_num}.csv")
        
        sdf_train_ids = top_k_from_buffer_ind
        
        buffer_examples_ind = np.setdiff1d(buffer_examples_ind, sdf_train_ids)
        
        new_buffer_examples = np.array(np.random.choice(pool_indices, experiment_config['buffer_step_size'], replace=False),
                                       dtype=np.int64)
        buffer_examples_ind = np.concatenate((buffer_examples_ind, new_buffer_examples))
        
        pool_indices = np.setdiff1d(pool_indices, buffer_examples_ind)