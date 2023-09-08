from .dqn import DQN

ModelConfigs = {model.__name__: model for model in [
    DQN,
]}
