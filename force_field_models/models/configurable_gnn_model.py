import torch
import torch.nn as nn

from force_field_models.data.preprocess import featurize_batch
from .model_components.utils import get_merge_func, build_layer

import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class ConfigurableGNNModel(torch.nn.Module):
    def __init__(self, config=None, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        logger.info("Loading GNN Model...")
        self.config = config
        self.model_blocks = nn.ModuleList()
        for model_config in self.config['models']['gnn_block']:
            self.model_blocks.append(
                build_layer(config=model_config)
            )
        self.pooling_layer = build_layer(
            config=self.config['models']['pooling_block']
            )
        self.energy_predictor = build_layer(
            config=self.config['models']['energy_predictor']
            )
        self.initialize_embedding_new()

        logger.info("GNN created successfully!")
                    
    def initialize_embedding_new(self):
        self.node_embedding_list = nn.ModuleList()
        for embedding_config in self.config['embeddings']['node_embeddings']:
            self.node_embedding_list.append(
                build_layer(config=embedding_config)
                )
        self.edge_embedding_list = nn.ModuleList()
        for embedding_config in self.config['embeddings']['edge_embeddings']:
            self.edge_embedding_list.append(
                build_layer(config=embedding_config)
                )
        self.embedding_module_dict = {
            'node_embedding_module': self.node_embedding_list
            , 'node_merge_func': get_merge_func(self.config['embeddings']['merge_node_embeddings'])
            , 'edge_embedding_module': self.edge_embedding_list
            , 'edge_merge_func': get_merge_func(self.config['embeddings']['merge_edge_embeddings'])
        }

    def process_embedding(self, batch_dict, embedding_key):
        embedding_list = []
        for embedding_module in self.embedding_module_dict[f'{embedding_key}_embedding_module']:
            embedding_list.append(embedding_module(batch_dict))
        merged_embs = self.embedding_module_dict[f'{embedding_key}_merge_func'](embedding_list)
        
        return merged_embs
        
    def update_batch_dict(self, batch_dict, node_embs, edge_embs):
        if self.config['embeddings']['multiply_edge_weights']:
            edge_embs *= batch_dict['edge_weights'].unsqueeze(1)
        
        batch_dict['x'] = node_embs
        batch_dict['edge_attr'] = edge_embs
        

    def forward(self, batch, inference=False):
        if inference:
            featurize_batch(batch=batch)
            
        batch_dict = {k: getattr(batch, k, None) for k in ['x', 'edge_index', 'edge_attr',
                                                            'batch', 'edge_weights', 'pos', 'norm']}

        node_embeddings = self.process_embedding(batch_dict=batch_dict, embedding_key='node')
        edge_embeddings = self.process_embedding(batch_dict=batch_dict, embedding_key='edge')
        self.update_batch_dict(batch_dict, node_embeddings, edge_embeddings)

        for model_block in self.model_blocks:
            batch_dict['x'] = model_block(batch_dict)
            
        pooled_embeddings = self.pooling_layer(batch_dict)
        energy_prediction = self.energy_predictor(pooled_embeddings)

        return energy_prediction
    