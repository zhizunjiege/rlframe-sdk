from typing import List, Union

from .base import AnyDict, ConfigBase, ServiceBase

from .hooks import HookConfigs
from .models import ModelConfigs


class Agent(ServiceBase):
    """Class for agent configs."""

    def __init__(
        self,
        name: str,
        hypers: Union[ConfigBase, AnyDict],
        training: bool,
        sifunc: str,
        oafunc: str,
        rewfunc: str,
        hooks: List[Union[ConfigBase, AnyDict]] = [],
    ):
        """Init config.

        Args:
            name: model name.
            hypers: model hypers.
            training: whether this agent is for training.
            sifunc: states to inputs function in python code.
            oafunc: outputs to actions function in python code.
            rewfunc: reward function in python code.
            hooks: hook configs.
        """
        if isinstance(hypers, ConfigBase):
            self.name = hypers.name
            self.hypers = hypers.dump()
        else:
            self.name = name
            if name in ModelConfigs:
                self.hypers = ModelConfigs[name](**hypers).dump()
            else:
                print(f'Warning: model {name} not found, using raw hypers.')
                self.hypers = hypers

        self.training = training

        self.sifunc = sifunc
        self.oafunc = oafunc
        self.rewfunc = rewfunc

        self.hooks = []
        for hook in hooks:
            if isinstance(hook, ConfigBase):
                self.hooks.append({'name': hook.name, 'args': hook.dump()})
            else:
                HookConfigs[hook['name']](**hook['args'])
                self.hooks.append(hook)
