import torch
from torch.utils.data import DataLoader, Dataset
import random
import yaml
import pickle
import os

from data_util import *

def download_data_from_hf(config):
    hf_url = config['dataset_url']
    train_data, val_data, test_data = get_hf_splits(hf_url)
    return train_data, val_data, test_data\

def load_data(config_path):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Set random seed for reproducibility
    seed = config.get('seed', 42)
    random.seed(seed)
    torch.manual_seed(seed)

    # Load and return saved dataset if specified
    if config['use_saved_dataset']:
        train_dataloader = load_from_pkl(config['dataset_storage_train'])
        val_dataloader = load_from_pkl(config['dataset_storage_val'])
        test_dataloader = load_from_pkl(config['dataset_storage_test'])
        return train_dataloader, val_dataloader, test_dataloader
    else:
        # Load raw data from pickle or download
        if not os.path.exists(config['raw_train_data_path']) or not os.path.exists(config['raw_val_data_path']) or not os.path.exists(config['raw_test_data_path']):
            raw_train_data, raw_val_data, raw_test_data = download_data_from_hf(config)
            if config.get('save_raw_data', False):
                save_to_pkl(raw_train_data, config['raw_train_data_path'])
                save_to_pkl(raw_val_data, config['raw_val_data_path'])
                save_to_pkl(raw_test_data, config['raw_test_data_path'])
        else:
            raw_train_data = load_from_pkl(config['raw_train_data_path'])
            raw_val_data = load_from_pkl(config['raw_val_data_path'])
            raw_test_data = load_from_pkl(config['raw_test_data_path'])


        # Apply transformations and form dataloaders
        if config.get('apply_transformations_in_getitem', False):
            train_dataloader = form_dataloader(
                raw_train_data,
                batch_size=config.get('batch_size', 32),
                shuffle=True,
                num_workers=config.get('num_workers', 4),
                transform=config.get('transformations', None)
            )
            val_dataloader = form_dataloader(
                raw_val_data,
                batch_size=config.get('batch_size', 32),
                shuffle=False,
                num_workers=config.get('num_workers', 4),
                transform=config.get('transformations', None)
            )
            test_dataloader = form_dataloader(
                raw_test_data,
                batch_size=config.get('batch_size', 32),
                shuffle=False,
                num_workers=config.get('num_workers', 4),
                transform=config.get('transformations', None)
            )
        else:
            processed_train_data = apply_transformations(raw_train_data, config.get('transformations', None))
            processed_val_data = apply_transformations(raw_val_data, config.get('transformations', None))
            processed_test_data = apply_transformations(raw_test_data, config.get('transformations', None))
            train_dataloader = form_dataloader(
                processed_train_data,
                batch_size=config.get('batch_size', 32),
                shuffle=True,
                num_workers=config.get('num_workers', 4)
            )
            val_dataloader = form_dataloader(
                processed_val_data,
                batch_size=config.get('batch_size', 32),
                shuffle=False,
                num_workers=config.get('num_workers', 4)    
            )
            test_dataloader = form_dataloader(
                processed_test_data,
                batch_size=config.get('batch_size', 32),
                shuffle=False,
                num_workers=config.get('num_workers', 4)
            )

        # Save processed dataloaders and return
        save_to_pkl(train_dataloader, config['dataset_storage_train'])
        save_to_pkl(val_dataloader, config['dataset_storage_val'])
        save_to_pkl(test_dataloader, config['dataset_storage_test'])
        return train_dataloader, val_dataloader, test_dataloader

if __name__ == "__main__":
    config_path = 'config.yaml'
    data = load_data(config_path)
    print(f"Loaded dataset with {len(data)} samples.")