from typing import Union

from .base import AnyDict, ConfigBase, ServiceBase

from .engines import EngineConfigs


class Simenv(ServiceBase):
    """Class for simenv configs."""

    def __init__(
        self,
        name: str,
        args: Union[ConfigBase, AnyDict],
    ):
        """Init config.

        Args:
            name: engine name.
            args: engine args.
        """
        self.name = name
        if isinstance(args, ConfigBase):
            self.args = args.dump()
        else:
            if name in EngineConfigs:
                self.args = EngineConfigs[name](**args).dump()
            else:
                print(f'Warning: engine {name} not found, using raw args.')
                self.args = args
