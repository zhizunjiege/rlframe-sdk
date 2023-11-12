from ..base import ConfigBase


class AutoSave(ConfigBase):
    """Auto save model weights, buffer and status."""

    def __init__(
        self,
        *,
        per_steps=10000,
        per_episodes=100,
        save_weights=True,
        save_buffer=False,
        save_status=False,
    ):
        """Init hook.

        Args:
            per_steps: Save every per_steps steps.
            per_episodes: Save every per_episodes episodes.
            save_weights: Whether to save weights.
            save_buffer: Whether to save buffer.
            save_status: Whether to save status.
        """
        if per_steps <= 0:
            raise ValueError('per_steps must be positive.')
        if per_episodes <= 0:
            raise ValueError('per_episodes must be positive.')
        if not save_weights and not save_buffer and not save_status:
            raise ValueError('At least one of save_weights, save_buffer, save_status must be True.')

        self.per_steps = per_steps
        self.per_episodes = per_episodes
        self.save_weights = save_weights
        self.save_buffer = save_buffer
        self.save_status = save_status
