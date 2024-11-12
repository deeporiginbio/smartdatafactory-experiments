# Smart Distributed Data Factory

![Static Badge](https://img.shields.io/badge/bioRxiv-10.1101%2F2024.10.22.619651-red) [![Zenodo](https://zenodo.org/badge/DOI/10.5281/zenodo.14008357.svg)](https://doi.org/10.5281/zenodo.14008357)

This repository hosts the source code for the experiments in the paper **"Smart Distributed Data Factory: Volunteer Computing Platform for Active Learning-Driven Molecular Dara Acquisition"**.
The repository provides scripts for the training and inference of energy prediction models, as well as the active-learning framework simulation.

The pre-print with the detailed description of the methods and implementation is available on [bioRxiv](https://www.biorxiv.org/content/10.1101/2024.10.22.619651v2).\
The conformational energy dataset and the benchmark for machine learning models is available on [Zenodo](https://zenodo.org/records/14008357).

### Conformational energy prediction

We provide a script for running the inference with our conformational energy prediction models. For example, you can run the GENConv model on input conformations (each in a separate .sdf file) as follows:
```
python -m force_field_models.inference.inference --config force_field_models/model_configs/ConfigurableGNNModel_GENConv_new_normals.yaml --checkpoint GENConv.pth --data_dir dataset/molecules --output_file predictions.csv
```
The predictions are in Hartree units.

Model checkpoints:
- GENConv: [[Download link](https://sddf-checkpoints.s3.us-east-1.amazonaws.com/energy-v2024-Q3/GENConv.pth)][config file: `ConfigurableGNNModel_GENConv_new_normals.yaml`]
- PNAConv: [[Download link](https://sddf-checkpoints.s3.us-east-1.amazonaws.com/energy-v2024-Q3/PNAConv.pth)][config file: `ConfigurableGNNModel_PNAConv_new_normals.yaml`]
- ResGatedConv: [[Download link](https://sddf-checkpoints.s3.us-east-1.amazonaws.com/energy-v2024-Q3/ResGatedConv.pth)][config file: `ConfigurableGNNModel_ResGatedConv.yaml`]
- GeneralConv: [[Download link](https://sddf-checkpoints.s3.us-east-1.amazonaws.com/energy-v2024-Q3/GeneralConv.pth)][config file: `ConfigurableGNNModel_GeneralConv_new_normals.yaml`]
- TransformerConv: [[Download link](https://sddf-checkpoints.s3.us-east-1.amazonaws.com/energy-v2024-Q3/TransformerConv.pth)][config file: `ConfigurableGNNModel_TransformerConv_new_normals_1.yaml`]

#### Installation steps

In order to run the code, you first need to have Python 3.11 or Python 3.12 installed on your system. 
Then, you should install the remaining dependencies using:
```
pip install -r requirements.txt
```

### Active learning-based conformation sampling

We also provide a script (`force_field_models/train/cycle.py`) for running the simulation of the active learning-based dataset sampling. 
In order to run it, you should specify the model configs, the initial datasets, and training-related hyperparameters.
A detailed explanation and an example is given in [cycle_README.md](cycle_README.md).

### How to cite this work

For citation please use:
- The paper (pre-print):\
*Ghukasyan, T., Altunyan, V., Bughdaryan, A., Aghajanyan, T., Smbatyan, K., Papoian, G. A., & Petrosyan, G. (2024). SMART DATA FACTORY: VOLUNTEER COMPUTING PLATFORM FOR ACTIVE LEARNING-DRIVEN MOLECULAR DATA ACQUISITION. bioRxiv, 2024-10.*
- The dataset:\
*Altunyan, V., Ghukasyan, T., Bughdaryan, A., Aghajanyan, T., Smbatyan, K., Papoian, G., & Petrosyan, G. (2024). SDDF Energy Dataset (2024-Q3) [Data set]. Zenodo. https://doi.org/10.5281/zenodo.14008357*
