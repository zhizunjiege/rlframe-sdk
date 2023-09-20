from .dqn import DQN
from .doubledqn import DoubleDQN
from .ddpg import DDPG
from .maddpg import MADDPG

ModelConfigs = {model.__name__: model for model in [
    DQN,
    DoubleDQN,
    DDPG,
    MADDPG,
]}
