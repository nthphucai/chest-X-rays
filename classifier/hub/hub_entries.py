from dataclasses import dataclass
from importlib.machinery import SourceFileLoader
from pathlib import Path

"""
Load function from given path.
"""


class HubEntries:
    def __init__(self, absolute_path, module_name):
        self.path = absolute_path
        self.name = module_name

    def load_func(self, entry_name, *args, **kwargs):
        """
        load a function from given full path
        """
        module = SourceFileLoader(self.name, self.path).load_module()
        function = getattr(module, entry_name)

        return function(*args, **kwargs)
