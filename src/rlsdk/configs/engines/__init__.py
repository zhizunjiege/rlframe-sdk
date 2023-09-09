from .cqsim import CQSIM

EngineConfigs = {engine.__name__: engine for engine in [
    CQSIM,
]}
