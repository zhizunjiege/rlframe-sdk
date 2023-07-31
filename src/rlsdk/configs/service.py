from typing import Literal


class Service:
    """Class for service configs."""

    def __init__(
        self,
        type: Literal['agent', 'simenv'],
        name: str,
        host: str,
        port: int,
        desc: str,
    ):
        """Init config.

        Args:
            type: type of this service, agent or simenv.
            name: name of this service, optional.
            host: host of this service, ip address or domain name.
            port: port of this service, in range [0, 65535].
            desc: description of this service, optional.
        """
        if type not in ['agent', 'simenv']:
            raise ValueError('service type must be agent or simenv')
        if port < 0 or port > 65535:
            raise ValueError('port must be in range [0, 65535]')

        self.type = type
        self.name = name
        self.host = host
        self.port = port
        self.desc = desc
