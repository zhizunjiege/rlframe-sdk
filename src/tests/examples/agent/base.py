from abc import ABC
from abc import abstractmethod
from typing import Any, Dict, Tuple

AnyDict = Dict[str, Any]


class RLModelBase(ABC):
    """Abstract base class for all reinforcement learning models."""

    def __init__(self, training: bool):
        """Init model.

        Args:
            training: whether model is used for `train` or `infer`.
        """
        self._training = training

    @abstractmethod
    def react(self, states: Any) -> Any:
        """Get action from model.

        Args:
            states: States of enviroment.

        Returns:
            Action.
        """
        ...

    @abstractmethod
    def store(
        self,
        states: Any,
        actions: Any,
        next_states: Any,
        reward: Any,
        terminated: bool,
        truncated: bool,
    ) -> Any:
        """Store experience replay data.

        Args:
            states: States of enviroment.
            actions: Actions of model.
            next_states: Next states of enviroment.
            reward: Reward of enviroment.
            terminated: Whether a `terminal state` (as defined under the MDP of the task) is reached.
            truncated: Whether a `truncation condition` outside the scope of the MDP is satisfied.
        """
        ...

    @abstractmethod
    def train(self) -> Any:
        """Train model.

        Returns:
            Necessary infos.
        """
        ...

    @abstractmethod
    def get_weights(self) -> Any:
        """Get weights of neural networks.

        Returns:
            Weights of networks.
        """
        ...

    @abstractmethod
    def set_weights(self, weights: Any) -> Any:
        """Set weights of neural networks.

        Args:
            weights: Weights of networks.
        """
        ...

    @abstractmethod
    def get_buffer(self) -> Any:
        """Get buffer of experience replay.

        Returns:
            Buffer of replay.
        """
        ...

    @abstractmethod
    def set_buffer(self, buffer: Any) -> Any:
        """Set buffer of experience replay.

        Args:
            buffer: Buffer of replay.
        """
        ...

    def get_status(self) -> AnyDict:
        """Get intermediate status of model.

        Returns:
            Intermediate status.
        """
        status = {}
        for key in self.__dict__:
            if key[0] == '_' and key.find('__') == -1:
                name = key[1:]
                status[name] = self.__dict__[key]
        return status

    def set_status(self, status: AnyDict) -> None:
        """Set intermediate status of model.

        Args:
            status: Intermediate status.

        Raises:
            KeyError: When key of status is invalid.
        """
        for name in status:
            key = f'_{name}'
            if key in self.__dict__:
                setattr(self, key, status[name])
            else:
                raise KeyError(f'Key: `{name}` is invalid.')

    @property
    def training(self) -> bool:
        """Getter of training."""
        return self._training

    @training.setter
    def training(self, value: bool):
        """Setter of training."""
        self._training = value

    def call(self, name: str, dstr: str = '', dbin: bytes = b'') -> Tuple[str, str, bytes]:
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
        """Close model."""
        ...
