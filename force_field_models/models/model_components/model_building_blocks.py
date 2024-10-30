import math
import torch
import torch.nn as nn
import torch_geometric
import torch_geometric.nn

from .utils import build_layer, get_merge_func


class GNNBlock(torch.nn.Module):
    def __init__(self, config, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.pre_gnn = build_layer(config=config['pre_gnn_layer'])
        self.gnn_layer = build_layer(config=config['gnn_layer'])
        self.post_gnn = build_layer(config=config['post_gnn_layer'])

        self.forward_args = config['forward_args']
        self.skip_conncetion_flag = config.get('skip_connection', False)

    def forward(self, batch_dict):
        batch_args = {key: batch_dict.get(key, None) for key in self.forward_args}
        if None in batch_args.values():
            missing_keys = [k for k, v in batch_args.items() if v is None]
            raise ValueError(f"Missing keys in batch_dict: `{missing_keys}`")
        
        if self.skip_conncetion_flag:
            skip_connection = batch_args['x']
        
        batch_args['x'] = self.pre_gnn(batch_args['x'])
        x = self.gnn_layer(**batch_args)
        x = self.post_gnn(x)
        x = x + skip_connection if self.skip_conncetion_flag else x
        
        return x


class PoolingLayer(nn.Module):
    def __init__(self, config, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.pooling_ops = self.create_pooling_ops(config)
        self.merge_func = get_merge_func(config['merge_func'])
        self.forward_args = config['forward_args']

    def create_pooling_ops(self, pooling_config):
        pooling_ops = []
        for op in pooling_config['layers']:
            pooling_op = getattr(
                torch_geometric.nn.pool
                , op['type']
                , None
                )
            if pooling_op is None:
                raise ValueError(f"Unsupported pooling operation: {op}")
            pooling_ops.append(pooling_op)
            
        return pooling_ops
            
    def forward(self, batch_dict):
        
        batch_args = {key: batch_dict.get(key, None) for key in self.forward_args}

        if None in batch_args.values():
            missing_keys = [k for k, v in batch_args.items() if v is None]
            raise ValueError(f"Missing keys in batch_dict: `{missing_keys}`")
        
        pooled_outputs = []
        for pool_layer in self.pooling_ops:
            pooled_output = pool_layer(**batch_args).squeeze(1)
            pooled_outputs.append(pooled_output)

        merged_output = self.merge_func(pooled_outputs)

        return merged_output
    

class IdenticalLayer(torch.nn.Module):
    def __init__(self, config=None, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
    
    def forward(self, x, edge_index=None, edge_attr=None, *args, **kwargs):
        return x

class OneHotAndScale(torch.nn.Module):
    def __init__(self, config, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        
        # config = config['params']
        self.distance_bounds = torch.tensor(config['distance_bounds'])
        self.angle_bounds = torch.tensor(config['angle_bounds'])

        self.dist_embed_model = build_layer(config['dist_embed_model'])
        self.angle_embed_model = build_layer(config['angle_embed_model'])

        feat_funcs = {'transform_one_hot': self.transform_one_hot
                      , 'featurize_embedding_with_relative_distance': self.featurize_embedding_with_relative_distance}
        
        self.featurizer_func = feat_funcs[config['feat_func']]

    def transform_one_hot(self, embeddings):
        distance_one_hot = nn.functional.one_hot(
            torch.bucketize(embeddings[:, 0], self.distance_bounds)
            , num_classes=self.distance_bounds.shape[0] + 1).type(torch.float32)
        
        angle_one_hot = nn.functional.one_hot(
            torch.bucketize(embeddings[:, 1:], self.angle_bounds)
            , num_classes=self.angle_bounds.shape[0] + 1
            ).type(torch.float32)
        
        return distance_one_hot, angle_one_hot

    def featurize_embedding_with_relative_distance(self, embeddings):
        relative_embedd_dist = torch.exp(-torch.abs((embeddings[:, 0] - self.distance_bounds.reshape(-1, 1)).T))
        relative_angles_flatten = torch.flatten(embeddings[:, 1:])

        relative_embedd_angle = torch.exp(-torch.abs((relative_angles_flatten - self.angle_bounds.reshape(-1, 1)).T))
        relative_embedd_angle = relative_embedd_angle.reshape(embeddings.shape[0], 24)

        return relative_embedd_dist, relative_embedd_angle


    def forward(self, embeddings):
        device = embeddings.device

        self.distance_bounds = self.distance_bounds.to(device=device)
        self.angle_bounds = self.angle_bounds.to(device=device)

        distance_embedding, angle_embedding = self.featurizer_func(embeddings)
        out_dist = self.dist_embed_model(distance_embedding)
        out_angle = self.angle_embed_model(angle_embedding)
        out_angle = out_angle.reshape(embeddings.shape[0], -1)

        return torch.cat([out_angle, out_dist], dim=1)


class EmbeddingLayer(torch.nn.Module):
    def __init__(self, config, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.config = config
        if config['embedding_type'] == 'embedding':
            self.embedding = nn.Embedding(**config['embedding_params'])
        elif config['embedding_type'] == 'precomputed':
            self.embedding = IdenticalLayer()
        else:
            raise NotImplementedError(f"config type {config['embedding_type']} for embedding layer is not implemented")
        
        self.post_processing = self.make_process_embedding()
        self.forward_args = config['forward_argument']
        self.index = config['forward_index']
    
    def make_process_embedding(self):
        if self.config['post_processing_config']['type'] in ["Sequential", "IdenticalLayer"]:
            return build_layer(config=self.config['post_processing_config'])
        elif self.config['post_processing_config']['type'] == 'OneHotAndScale':
            return OneHotAndScale(self.config['post_processing_config'])
        else:
            raise NotImplementedError(f"In make process embedding not impelemted {self.config['post_processing_config']['type']}")
        
    
    def forward(self, batch_dict):
        #TODO fix in data generation to generate atom types as int. not float32
        forward_argument = batch_dict[self.forward_args][self.index]
        if self.config['embedding_type'] == 'embedding':
            forward_argument = forward_argument.type(torch.int32).reshape(-1)

        embeddings = self.embedding(forward_argument)
        post_processed = self.post_processing(embeddings)

        return post_processed

                
        

    

        
