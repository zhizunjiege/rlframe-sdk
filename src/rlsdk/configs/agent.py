import json
from typing import List

from .base import ConfigBase, ServiceBase

from .hooks import HookConfigs
from .models import ModelConfigs


class Agent(ServiceBase):
    """Class for agent configs."""

    def __init__(
        self,
        model: ConfigBase,
        training: bool,
        sifunc: str,
        oafunc: str,
        rewfunc: str,
        hooks: List[ConfigBase] = [],
    ):
        """Init config.

        Args:
            model: model config.
            training: whether this agent is for training.
            sifunc: states to inputs function in python code.
            oafunc: outputs to actions function in python code.
            rewfunc: reward function in python code.
            hooks: hook configs.
        """
        self.model = model

        self.training = training
        self.sifunc = sifunc
        self.oafunc = oafunc
        self.rewfunc = rewfunc

        self.hooks = hooks

    @classmethod
    def from_files(cls, path: str):
        """Create config from files.

        Args:
            path: path to config files.
        """
        with open(f'{path}/configs.json', 'r') as f:
            configs = json.load(f)

        super().parse_refs(configs, path)

        model = ModelConfigs[configs['name']](**configs['hypers'])
        hooks = [HookConfigs[hook['name']](**hook['args']) for hook in configs['hooks']]
        return cls(
            model=model,
            training=configs['training'],
            sifunc=configs['sifunc'],
            oafunc=configs['oafunc'],
            rewfunc=configs['rewfunc'],
            hooks=hooks,
        )
