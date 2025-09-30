from dataclasses import is_dataclass
from typing import Any
import json
import importlib

def dataclass_to_dict(obj: Any) -> dict:
    if is_dataclass(obj):
        result = {
            '__class__': obj.__class__.__name__,
            '__module__': obj.__class__.__module__,
            'data': {k: dataclass_to_dict(v) for k, v in obj.__dict__.items()}
        }
        return result
    elif isinstance(obj, list):
        return [dataclass_to_dict(i) for i in obj]
    elif isinstance(obj, dict):
        return {k: dataclass_to_dict(v) for k, v in obj.items()}
    else:
        return obj

def dict_to_dataclass(obj: Any) -> Any:
    if isinstance(obj, dict) and '__class__' in obj:
        module = importlib.import_module(obj['__module__'])
        cls = getattr(module, obj['__class__'])
        kwargs = {k: dict_to_dataclass(v) for k, v in obj['data'].items()}
        return cls(**kwargs)
    elif isinstance(obj, list):
        return [dict_to_dataclass(i) for i in obj]
    elif isinstance(obj, dict):
        return {k: dict_to_dataclass(v) for k, v in obj.items()}
    else:
        return obj

def save_dataclass(obj, path):
    with open(path, 'w') as f:
        json.dump(dataclass_to_dict(obj), f, indent=4)

def load_dataclass(path):
    with open(path, 'r') as f:
        data = json.load(f)
    return dict_to_dataclass(data)

