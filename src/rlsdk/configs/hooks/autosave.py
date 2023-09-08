from ..base import ConfigBase


class AutoSave(ConfigBase):
    """Auto save model weights."""

    def __init__(
        self,
        *,
        per_steps=10000,
        per_episodes=100,
    ):
        """Init hook.

        Args:
            per_steps: Save weights every per_steps steps.
            per_episodes: Save weights every per_episodes episodes.
        """
        if per_steps <= 0:
            raise ValueError('per_steps must be positive.')
        if per_episodes <= 0:
            raise ValueError('per_episodes must be positive.')

        self.per_steps = per_steps
        self.per_episodes = per_episodes
