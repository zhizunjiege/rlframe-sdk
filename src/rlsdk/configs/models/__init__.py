from .dqn import DQN

ModelConfigs = {model.name: model for model in [
    DQN,
]}
