from pathlib import Path
from .hub_entries import HubEntries


def get_entries(path):
    """
    path: path to python module
    """
    return HubEntries(path, Path(path).name.replace(".py", ""))
