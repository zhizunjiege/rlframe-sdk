from abc import ABC
from abc import abstractmethod
from enum import auto, Enum
from typing import Any, Dict, List, Tuple

AnyDict = Dict[str, Any]


class CommandType(Enum):
    """Command type."""

    INIT = auto()
    START = auto()
    PAUSE = auto()
    STEP = auto()
    RESUME = auto()
    STOP = auto()
    EPISODE = auto()
    PARAM = auto()


class EngineState(Enum):
    """Engine state."""

    UNINITED = auto()
    STOPPED = auto()
    RUNNING = auto()
    SUSPENDED = auto()


class SimEngineBase(ABC):
    """Abstract base class for all simulation engines."""

    def __init__(self):
        """Init engine."""
        self._state = EngineState.UNINITED

    @abstractmethod
    def control(
        self,
        type: CommandType,
        params: AnyDict,
    ) -> bool:
        """Control engine.

        Args:
            type: Command type.
            params: Command params.

        Returns:
            True if success.
        """
        ...

    @abstractmethod
    def monitor(self) -> Tuple[List[AnyDict], List[str]]:
        """Monitor engine.

        Returns:
            Data of simulation process.
            Logs of simulation engine.
        """
        ...

    @property
    def state(self) -> EngineState:
        """Getter of state."""
        return self._state

    @state.setter
    def state(self, value: EngineState) -> None:
        """Setter of state."""
        self._state = value

    def call(self, name: str, dstr='', dbin=b'') -> Tuple[str, str, bytes]:
        """Any method can be called.

        Args:
            name: Name of method.
            dstr: String data.
            dbin: Binary data.

        Returns:
            Name of method, string data and binary data.
        """
        return name, '', b''

    def __del__(self):
        """Close engine."""
        ...
