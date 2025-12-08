import yaml
from pathlib import Path
import json

def read_yaml(path: str):
    p = Path(path)
    with p.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)  # safe for untrusted YAML

def read_json(path:str):
    p = Path(path)
    with p.open("r") as f: # Corrected line: removed the redundant path argument
        return json.load(f)

def get_cui_mapping(items):
    """
    Takes a list of dicts with keys:
        - 'CUI'
        - 'Canonical name'
    Returns a dict mapping 'CUI' -> 'Canonical name'
    """
    return {item['CUI']: item['Canonical name'] for item in items}