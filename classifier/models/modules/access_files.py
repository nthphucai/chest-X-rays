import os
from typing import Dict, Optional, Union

"""
Picke File
"""
import pickle


def write_pickle_file(data: dict, path: str, name: Optional[str] = None) -> None:
    if name is not None:
        save_path = os.path.join(f"{path}", f"{name}" + ".pkl")
    else:
        save_path = path

    f = open(save_path, "wb")
    pickle.dump(data, f)
    f.close()


def read_pickle_file(path, name: Optional[str] = None) -> dict:
    if name is not None:
        save_path = os.path.join(f"{path}", f"{name}" + ".pkl")
    else:
        save_path = path

    f = open(save_path, "rb")
    pickle_file = pickle.load(f)
    return pickle_file


"""
Json File
"""
import json


def write_json_file(
    data: dict, path: str, name: Optional[str] = None, **kwargs
) -> None:
    if name is not None:
        save_path = os.path.join(f"{path}", f"{name}" + ".json")
    else:
        save_path = path

    with open(save_path, "w") as outfile:
        json.dump(data, outfile, ensure_ascii=False, indent=4, **kwargs)


def load_json_file(path: str, name: Optional[str] = None) -> dict:
    if name is not None:
        save_path = os.path.join(f"{path}", f"{name}" + ".json")
    else:
        save_path = path

    with open(save_path) as outfile:
        data = json.load(outfile)
        return data


"""
Yaml File
"""
import yaml


def read_yaml(file_path: str):
    with open(file_path, "r") as f:
        return yaml.safe_load(f)
