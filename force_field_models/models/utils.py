
import torch
import numpy as np

# uni_mol_repr = None


def get_unimol_input(batch, config):
    ATOM_TYPE_MAP = config['ATOM_TYPE_MAP']
    INVERSE_ATOM_TYPE_MAP = {v: k for k, v in ATOM_TYPE_MAP.items()}
    
    molecule_coordinates = [batch.coordinates[batch.batch == index] for index in range(torch.max(batch.batch) + 1)]

    atom_types_numeric = np.array([INVERSE_ATOM_TYPE_MAP[int(i[0])] for i in batch.x])
    atom_types_symbolic = [atom_types_numeric[batch.batch == index] for index in range(torch.max(batch.batch) + 1)]

    return atom_types_symbolic, molecule_coordinates

@torch.no_grad()
def compute_unimol_features(clf, batch, config):
    """
    Return mol_feature and atomic features for given molecule
    """
    device = batch.x.device
    batch = batch.cpu()
    atom_types, coordinates = get_unimol_input(batch=batch, config=config)
    data = {
        'atoms': atom_types,
        'coordinates': coordinates,
        'target': [-1] * len(atom_types)
    }
    out = clf.get_repr(data=data, return_atomic_reprs=True)

    batch = batch.to(device)
    return torch.tensor(np.concatenate(out['atomic_reprs'])).to(device)

def get_unimol_fts(batch):
    raise NotImplementedError("get_unimol_fts is not implemented")
    # if uni_mol_repr == None:
        #     from unimol_tools import UniMolRepr
        #     uni_mol_repr = UniMolRepr(remove_hs=False
        #                 , use_gpu=True
        #                 )
    
        # return get_unimol_features(clf=uni_mol_repr, batch=batch, config=config)

def extract_node_embedding(embedding, batch, config):
    # global uni_mol_repr
    EMBEDDING_DICT = {'unimol_precomputed': lambda input: input.x[1]
                      , 'unimol_compute': lambda input: get_unimol_fts(input)
                      }
    return EMBEDDING_DICT[embedding['type']](batch)
    
def extract_edge_embedding(embedding, batch):
    if embedding == 'ppf':
        return batch.edge_attr[2]
    else:
        raise NotImplementedError("extract_edge_embedding not implemented")
    

