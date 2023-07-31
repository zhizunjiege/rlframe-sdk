from abc import ABC, abstractmethod
import json
from typing import Any, Dict


class ConfigBase(ABC):
    """Abstract base class for all configs."""

    @classmethod
    @property
    def name(cls) -> str:
        """Return name of this config."""
        return cls.__name__

    @abstractmethod
    def __init__(self):
        """Init config."""
        ...

    def dump(self) -> Dict[str, Any]:
        """Dump config to dict.

        Returns:
            dumped config.
        """
        return {k: v for k, v in self.__dict__.items() if not k.startswith('_')}


class ServiceBase(ABC):
    """Abstract base class for all services."""

    @abstractmethod
    def __init__(self):
        """Init config."""
        ...

    @staticmethod
    def parse_refs(target: Dict[str, Any], path: str, ref='refs.json'):
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
