import json

from .base import ConfigBase, ServiceBase

from .engines import EngineConfigs


class Simenv(ServiceBase):
    """Class for simenv configs."""

    def __init__(
        self,
        engine: ConfigBase,
    ):
        """Init config.

        Args:
            engine: engine config.
        """
        self.engine = engine

    @classmethod
    def from_files(cls, path: str):
        """Create config from files.

        Args:
            path: path to config files.
        """
        with open(f'{path}/configs.json', 'r') as f:
            configs = json.load(f)

        super().parse_refs(configs, path)

        engine = EngineConfigs[configs['name']](**configs['args'])
        return cls(engine=engine)
