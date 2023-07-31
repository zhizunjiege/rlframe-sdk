from ..base import ConfigBase


class Training(ConfigBase):
    """Auto switch training mode."""

    def __init__(
        self,
        *,
        test_policy_every=100,
        test_policy_total=5,
    ):
        """Init config.

        Args:
            test_policy_every: Test policy every test_policy_every steps.
            test_policy_total: Test policy total test_policy_total times.
        """
        super().__init__()

        if test_policy_every <= 0:
            raise ValueError('test_policy_every must be positive.')
        if test_policy_total <= 0:
            raise ValueError('test_policy_total must be positive.')

        self.test_policy_every = test_policy_every
        self.test_policy_total = test_policy_total
