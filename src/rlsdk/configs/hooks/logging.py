from typing import Literal

from ..base import ConfigBase


class Logging(ConfigBase):
    """Auto logging to terminal and tensorboard."""

    levels = ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'FATAL', 'CRITICAL']

    def __init__(
        self,
        *,
        loglvl: Literal['DEBUG', 'INFO', 'WARNING', 'ERROR', 'FATAL', 'CRITICAL'] = 'INFO',
        terminal=True,
        tensorboard=True,
    ):
        """Init config.

        Args:
            loglvl: Logging level.
            terminal: Whether to log to terminal.
            tensorboard: Whether to log to tensorboard.
        """
        loglvl = loglvl.upper()
        if loglvl not in self.levels:
            raise ValueError(f'loglvl must be one of {", ".join(self.levels)}.')

        self.loglvl = loglvl
        self.terminal = terminal
        self.tensorboard = tensorboard
