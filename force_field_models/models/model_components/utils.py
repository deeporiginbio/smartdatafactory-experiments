import torch
import torch_geometric
from torch_geometric.loader import DataLoader

from ...models import model_components
from ...data.data import EnergyDataset, EnergyDatasetH5, EnergyDatasetChunksH5
_pna_degree_cache = {}

def build_custom_model_from_config(config):
    layer_cls = getattr(model_components, config['type'])
    layer = layer_cls(config=config.get('params', {}))
    
    return layer

def build_layer(config):
    layer_cls = config.get('type', None)

    if layer_cls == 'Sequential':
        layer = build_sequential_layer_from_config(config=config)
    elif layer_cls in dir(torch_geometric.nn):
        layer = build_gnn_layer_from_config(config=config)
    elif layer_cls in dir(model_components):
        layer = build_custom_model_from_config(config=config)
    else:
        raise ValueError(f"No config type in: `{config}`")
    
    return layer

def build_sequential_layer_from_config(config):
    layers = torch.nn.Sequential()
    for layer in config['layers']:
        layer_cls = layer['type']
        layer_params = layer.get('params', {})

        if layer_cls in dir(torch.nn):
            layer_module = getattr(torch.nn, layer_cls)
        elif layer_cls in dir(torch_geometric.nn):
            layer_module = getattr(torch_geometric.nn, layer_cls)
        else:
            raise ValueError(f"Unsupported layer type: {layer_cls}")
        layers.append(layer_module(**layer_params))

    return layers    

def build_gnn_layer_from_config(config):
    layer_cls = getattr(torch_geometric.nn, config['type'])
    if config['type'] == "PNAConv":
        _pna_degree_cache['degree'] = torch.tensor(config['deg'])
        if 'degree' not in _pna_degree_cache.keys():
            loader = get_dataloader(config)
            _pna_degree_cache['degree'] = layer_cls.get_degree_histogram(loader)
            del loader

        print("Degree Histogram for PNAConv constructed!")
        layer = layer_cls(deg=_pna_degree_cache['degree'], **config.get('params', {}))
    else:
        layer = layer_cls(**config.get('params', {}))
    
    return layer


def get_dataloader(config):
    path_to_data = config.get('train_data_path', None)
    data_type = config['data_type']
    if data_type == 'data_list':
        molecule_data = torch.load(path_to_data)
    elif data_type == 'Dataset':
        molecule_data = EnergyDataset(path_to_data, prefix=config['prefix'])
    elif data_type == 'DatasetH5':
            molecule_data = EnergyDatasetH5(path_to_data, prefix=config['prefix'])
    elif data_type == 'EnergyDatasetChunksH5':
            molecule_data = EnergyDatasetChunksH5(path_to_data, prefix=config['prefix'])
    else:
        raise Exception(f"No `{data_type}` was found in config")
    
    return DataLoader(molecule_data, batch_size=128)

def get_merge_func(func_name):
    """
    Get the appropriate merge function based on the specified merge type.

    Args:
        func_name (str): The name of the merge operation to use.

    Returns:
        Callable[[List[torch.Tensor]], torch.Tensor]: The merge function.

    Raises:
        ValueError: If the provided `func_name` is not supported.

    Available merge function options:
        - 'cat': Concatenates the input tensors along the feature dimension (dim=1).
        - 'sum': Sums the input tensors along the feature dimension (dim=1).
        - 'mean': Computes the mean of the input tensors along the feature dimension (dim=1).
        - 'max': Computes the maximum value of the input tensors along the feature dimension (dim=1).
        - 'min': Computes the minimum value of the input tensors along the feature dimension (dim=1).
    """
    merge_func_dict = {
        'cat': lambda x: torch.cat(x, dim=1),
        'sum': lambda x: torch.sum(x, dim=1),
        'mean': lambda x: torch.mean(x, dim=1),
        'max': lambda x: torch.max(x, dim=1)[0],
        'min': lambda x: torch.min(x, dim=1)[0]
    }

    merge_func = merge_func_dict.get(func_name, None)
    if merge_func is None:
        raise ValueError(f"Unsupported merge function: {func_name}. Supported functions: {list(merge_func_dict.keys())}")
    return merge_func

