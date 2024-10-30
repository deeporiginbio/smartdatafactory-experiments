import torch
import numpy as np
from tqdm import tqdm
from sklearn.linear_model import LinearRegression


def fit_self_energy(data_list):
    atom_count_matrix = []
    initial_target_list = []
    for graph in tqdm(data_list):
        atom_count_matrix.append(torch.bincount(torch.flatten(graph.x[0]), minlength=9).numpy())
        initial_target_list.append(graph.initial_target)

    data_np = np.vstack(atom_count_matrix)
    initial_target_list = np.array(initial_target_list)

    model = LinearRegression()
    model.fit(data_np, initial_target_list)
    floated_values = [float(num) for num in model.coef_]
    
    return floated_values[:-1]


