from abc import ABC, abstractmethod
import json
from typing import Any, Dict

AnyDict = Dict[str, Any]


class ConfigBase(ABC):
    """Abstract base class for all configs."""

    @classmethod
    def from_file(cls, file: str):
        """Create config from file.

        Args:
            file: path to config file.
        """
        with open(file, 'r') as f:
            configs = json.load(f)

        return cls(**configs)

    @property
    def name(self) -> str:
        """Get config name.

        Returns:
            config name.
        """
        return self.__class__.__name__

    @abstractmethod
    def __init__(self):
        """Init config."""
        ...

    def dump(self) -> AnyDict:
        """Dump config to dict.

        Returns:
            dumped config.
        """
        return {k: v for k, v in self.__dict__.items() if not k.startswith('_')}


class ServiceBase(ABC):
    """Abstract base class for all services."""

    @classmethod
    def from_files(cls, path: str, config='configs.json'):
        """Create config from files.

        Args:
            path: path to config files.
            config: config file name.
        """
        with open(f'{path}/{config}', 'r') as f:
            configs = json.load(f)

        cls.parse_refs(configs, path)

        return cls(**configs)

    @abstractmethod
    def __init__(self):
        """Init config."""
        ...

    @staticmethod
    def parse_refs(target: AnyDict, path: str, ref='refs.json'):
        """Parse refs to target.

        Args:
            target: target dict.
            path: path to reference.
            ref: reference file name.
        """
        try:
            with open(f'{path}/{ref}', 'r') as f:
                refs = json.load(f)
        except FileNotFoundError:
            return

        for k, v in refs.items():
            keys = k.split('.')
            tgt = target
            for key in keys[:-1]:
                tgt = tgt[key]
            ref = None
            with open(f'{path}/{v["path"]}', 'r') as f:
                if v['type'] == 'text':
                    ref = f.read()
                elif v['type'] == 'json':
                    ref = json.load(f)
            tgt[keys[-1]] = ref
