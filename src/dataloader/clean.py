import os
import yaml


def clean_workspace(config):

    os.remove(config['dataset_storage_train']) if os.path.exists(config['dataset_storage_train']) else None
    os.remove(config['dataset_storage_val']) if os.path.exists(config['dataset_storage_val']) else None
    os.remove(config['dataset_storage_test']) if os.path.exists(config['dataset_storage_test']) else None

    os.remove(config['raw_train_data_path']) if os.path.exists(config['raw_train_data_path']) else None
    os.remove(config['raw_val_data_path']) if os.path.exists(config['raw_val_data_path']) else None
    os.remove(config['raw_test_data_path']) if os.path.exists(config['raw_test_data_path']) else None   

if __name__ == "__main__":
    config_path = 'config.yaml'
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    clean_workspace(config)