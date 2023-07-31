from .autosave import AutoSave
from .logging import Logging
from .training import Training

HookConfigs = {hook.name: hook for hook in [
    Training,
    Logging,
    AutoSave,
]}
