import sys

from .data import EnergyDatasetNPZ
from .preprocess import preprocess_mol


if __name__ == "__main__":
    transform_function = preprocess_mol
    
    test_root = sys.argv[1]
    valid_root = sys.argv[2]
    train_root = sys.argv[3]
    
    EnergyDatasetNPZ(root=test_root, transform=transform_function, prefix="test_normals")
    EnergyDatasetNPZ(root=valid_root, transform=transform_function, prefix="valid_normals")
    EnergyDatasetNPZ(root=train_root, transform=transform_function, prefix="train_normals")
