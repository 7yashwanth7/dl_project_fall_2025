import torchvision.transforms as T
import string
import torch
from torch.utils.data import DataLoader, Dataset
import random
import yaml
import pickle
from datasets import load_dataset

IMAGE_TRANSFORMS = {
    'Resize': lambda **params: T.Resize(tuple(params['size'])),
    'Normalize': lambda **params: T.Normalize(mean=params['mean'], std=params['std'])
}

TEXT_TRANSFORMS = {
    'lowercase': lambda text, **params: lowercase(text),
    'remove_punctuation': lambda text, **params: remove_punctuation(text)
}

class ROCOv2Dataset(Dataset):
    def __init__(self, data, transform=None):
        self.data = data
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        if self.transform:
            for t in self.transform:
                item = t(item)
        return item
    
def form_dataloader(data, batch_size=32, shuffle=True, num_workers=4, transform=None):
    dataset = ROCOv2Dataset(data, transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    return dataloader

def lowercase(text):
    return text.lower()

def remove_punctuation(text):
    return text.translate(str.maketrans('', '', string.punctuation))

def apply_transformations(data, transformations):
    processed_data = []
    for item in data:
        new_item = item.copy() if hasattr(item, 'copy') else item
        # Image transformations
        if 'image' in transformations and 'image' in item:
            img_transforms = []
            for t in transformations['image']:
                func = IMAGE_TRANSFORMS.get(t['type'])
                if func:
                    img_transforms.append(func(**t.get('params', {})))
            if img_transforms:
                composed = T.Compose(img_transforms)
                new_item['image'] = composed(item['image'])
        # Text transformations
        if 'text' in transformations and 'text' in item:
            text = item['text']
            for t in transformations['text']:
                func = TEXT_TRANSFORMS.get(t['type'])
                if func:
                    text = func(text, **t.get('params', {}))
            new_item['text'] = text
        processed_data.append(new_item)
    return processed_data

def save_to_pkl(dataset, filepath):
    with open(filepath, 'wb') as f:
        pickle.dump(dataset, f)

def load_from_pkl(filepath):
    with open(filepath, 'rb') as f:
        dataset = pickle.load(f)
    return dataset

def load_config(config_path):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def get_hf_splits(hf_url):
    dataset = load_dataset(hf_url)
    train = dataset.get('train')
    val = dataset.get('validation')
    test = dataset.get('test')
    return train, val, test