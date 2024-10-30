import os
import argparse

from tqdm import tqdm
import torch
from torch_geometric.loader import DataLoader

from force_field_models.data.preprocess import preprocess_inference
from force_field_models.data.utils import ATOM_TYPE_MAP, SELF_ENERGIES_PRED
from force_field_models.utils.helpers import load_config, create_gnn_model


DEVICE = 'cpu'

ATOM_TYPE_MAP_INV = {v: k for k, v in ATOM_TYPE_MAP.items()}
 
ID2SELF_ENERGIES = {
   ATOM_TYPE_MAP[k]: v for k, v in SELF_ENERGIES_PRED.items()}


def process_input_args():
    parser = argparse.ArgumentParser(description='Model config file')
    parser.add_argument('--config', type=str, help='yaml filename which is located at model_configs')
    parser.add_argument('--checkpoint', type=str, help='path to the checkpoint')
    parser.add_argument('--data_dir', type=str, help='path of the dir with the molecule files')
    parser.add_argument('--output_file', type=str, help='path of the file to save the outputs')
    args = parser.parse_args()
    if args.config is None:
        raise Exception("Missing the required argument 'config'!")
    
    return args


def get_data_loader(data_dir):
    molecule_data_list = []

    mol_files = [
        os.path.join(data_dir, sdf_file) for sdf_file in os.listdir(data_dir) if sdf_file.endswith('.sdf')
    ]

    for mol in tqdm(mol_files):
        graph_data = preprocess_inference(mol_file=mol, input_type="sdf")
        if graph_data is not None:
            molecule_data_list.append(graph_data)

    return DataLoader(molecule_data_list, batch_size=20)


def get_model(config_path, checkpoint_path):
    config = load_config(config_path=config_path)
    
    with torch.no_grad():
        model = create_gnn_model(config=config)
        model.eval()
        state_dict = torch.load(checkpoint_path, map_location=DEVICE, weights_only=True)
        model.load_state_dict(state_dict)
    
    return model


def predict_energies(model, data_loader):
    with torch.no_grad():
        outputs = []
        file_names = []

        for batch in tqdm(data_loader):
            batch = batch.to(DEVICE)
            output = model(batch, True)

            for atom_idx, mol_idx in enumerate(batch.batch):
                atom_id = batch.x[0][atom_idx].item()
                output[mol_idx] += ID2SELF_ENERGIES.get(atom_id, torch.nan)

            outputs.extend(output.flatten().tolist())
            file_names.extend(batch.file_name)
        
    return {file_name: output for file_name, output in zip(file_names, outputs)}


def save_outputs(outputs, save_to):
    with open(save_to, 'w') as f:
        f.write("file_name,prediction\n")
        for file_name, prediction in outputs.items():
            f.write(f"{file_name},{prediction}\n")


if __name__ == '__main__':
    """
    This script is used to predict the energies of molecular conformations given the model checkpoint and the molecule files.
    The energies are predicted in Hartree units.
    
    Example usage:
    python -m force_field_models.inference.inference --config GENConv_new_normals.yaml --checkpoint ConfigurableGNNModel_GENConv_new_normals-epoch=47-valid_MAE=2.422.ckpt --data_dir dataset/molecules --output_file predictions.csv

    Arguments:
    --config: yaml filename with the model config
    --checkpoint: path to the model checkpoint
    --data_dir: path of the dir with the molecule files in .sdf format
    --output_file: path of the file to save the outputs
    """
    args = process_input_args()
    loader = get_data_loader(args.data_dir)
    model = get_model(args.config, args.checkpoint)
    predictions = predict_energies(model, loader)
    save_outputs(predictions, args.output_file)
