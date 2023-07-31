"""Generated protocol buffer code."""
from google.protobuf import descriptor as _descriptor
from google.protobuf import descriptor_pool as _descriptor_pool
from google.protobuf import message as _message
from google.protobuf import reflection as _reflection
from google.protobuf import symbol_database as _symbol_database

_sym_db = _symbol_database.Default()
from . import types_pb2 as types__pb2

DESCRIPTOR = _descriptor_pool.Default().AddSerializedFile(
    b'\n\x0bagent.proto\x12\ngame.agent\x1a\x0btypes.proto"\xbf\x01\n\x0bAgentConfig\x12\x10\n\x08training\x18\x01 \x01(\x08\x12\x0c\n\x04name\x18\x02 \x01(\t\x12\x0e\n\x06hypers\x18\x03 \x01(\t\x12\x0e\n\x06sifunc\x18\x04 \x01(\t\x12\x0e\n\x06oafunc\x18\x05 \x01(\t\x12\x0f\n\x07rewfunc\x18\x06 \x01(\t\x12+\n\x05hooks\x18\x07 \x03(\x0b2\x1c.game.agent.AgentConfig.Hook\x1a"\n\x04Hook\x12\x0c\n\x04name\x18\x01 \x01(\t\x12\x0c\n\x04args\x18\x02 \x01(\t"\x1d\n\tAgentMode\x12\x10\n\x08training\x18\x01 \x01(\x08"\x1f\n\x0cModelWeights\x12\x0f\n\x07weights\x18\x01 \x01(\x0c"\x1d\n\x0bModelBuffer\x12\x0e\n\x06buffer\x18\x01 \x01(\x0c"\x1d\n\x0bModelStatus\x12\x0e\n\x06status\x18\x01 \x01(\t2\xc2\x07\n\x05Agent\x12E\n\x0cResetService\x12\x19.game.types.CommonRequest\x1a\x1a.game.types.CommonResponse\x12C\n\x0cQueryService\x12\x19.game.types.CommonRequest\x1a\x18.game.types.ServiceState\x12D\n\x0eGetAgentConfig\x12\x19.game.types.CommonRequest\x1a\x17.game.agent.AgentConfig\x12E\n\x0eSetAgentConfig\x12\x17.game.agent.AgentConfig\x1a\x1a.game.types.CommonResponse\x12@\n\x0cGetAgentMode\x12\x19.game.types.CommonRequest\x1a\x15.game.agent.AgentMode\x12A\n\x0cSetAgentMode\x12\x15.game.agent.AgentMode\x1a\x1a.game.types.CommonResponse\x12F\n\x0fGetModelWeights\x12\x19.game.types.CommonRequest\x1a\x18.game.agent.ModelWeights\x12G\n\x0fSetModelWeights\x12\x18.game.agent.ModelWeights\x1a\x1a.game.types.CommonResponse\x12D\n\x0eGetModelBuffer\x12\x19.game.types.CommonRequest\x1a\x17.game.agent.ModelBuffer\x12E\n\x0eSetModelBuffer\x12\x17.game.agent.ModelBuffer\x1a\x1a.game.types.CommonResponse\x12D\n\x0eGetModelStatus\x12\x19.game.types.CommonRequest\x1a\x17.game.agent.ModelStatus\x12E\n\x0eSetModelStatus\x12\x17.game.agent.ModelStatus\x1a\x1a.game.types.CommonResponse\x12<\n\tGetAction\x12\x14.game.types.SimState\x1a\x15.game.types.SimAction(\x010\x01\x122\n\x04Call\x12\x14.game.types.CallData\x1a\x14.game.types.CallDatab\x06proto3'
)
_AGENTCONFIG = DESCRIPTOR.message_types_by_name['AgentConfig']
_AGENTCONFIG_HOOK = _AGENTCONFIG.nested_types_by_name['Hook']
_AGENTMODE = DESCRIPTOR.message_types_by_name['AgentMode']
_MODELWEIGHTS = DESCRIPTOR.message_types_by_name['ModelWeights']
_MODELBUFFER = DESCRIPTOR.message_types_by_name['ModelBuffer']
_MODELSTATUS = DESCRIPTOR.message_types_by_name['ModelStatus']
AgentConfig = _reflection.GeneratedProtocolMessageType(
    'AgentConfig', (_message.Message,), {
        'Hook':
            _reflection.GeneratedProtocolMessageType('Hook', (_message.Message,), {
                'DESCRIPTOR': _AGENTCONFIG_HOOK,
                '__module__': 'agent_pb2'
            }),
        'DESCRIPTOR':
            _AGENTCONFIG,
        '__module__':
            'agent_pb2'
    })
_sym_db.RegisterMessage(AgentConfig)
_sym_db.RegisterMessage(AgentConfig.Hook)
AgentMode = _reflection.GeneratedProtocolMessageType('AgentMode', (_message.Message,), {
    'DESCRIPTOR': _AGENTMODE,
    '__module__': 'agent_pb2'
})
_sym_db.RegisterMessage(AgentMode)
ModelWeights = _reflection.GeneratedProtocolMessageType('ModelWeights', (_message.Message,), {
    'DESCRIPTOR': _MODELWEIGHTS,
    '__module__': 'agent_pb2'
})
_sym_db.RegisterMessage(ModelWeights)
ModelBuffer = _reflection.GeneratedProtocolMessageType('ModelBuffer', (_message.Message,), {
    'DESCRIPTOR': _MODELBUFFER,
    '__module__': 'agent_pb2'
})
_sym_db.RegisterMessage(ModelBuffer)
ModelStatus = _reflection.GeneratedProtocolMessageType('ModelStatus', (_message.Message,), {
    'DESCRIPTOR': _MODELSTATUS,
    '__module__': 'agent_pb2'
})
_sym_db.RegisterMessage(ModelStatus)
_AGENT = DESCRIPTOR.services_by_name['Agent']
if _descriptor._USE_C_DESCRIPTORS == False:
    DESCRIPTOR._options = None
    _AGENTCONFIG._serialized_start = 41
    _AGENTCONFIG._serialized_end = 232
    _AGENTCONFIG_HOOK._serialized_start = 198
    _AGENTCONFIG_HOOK._serialized_end = 232
    _AGENTMODE._serialized_start = 234
    _AGENTMODE._serialized_end = 263
    _MODELWEIGHTS._serialized_start = 265
    _MODELWEIGHTS._serialized_end = 296
    _MODELBUFFER._serialized_start = 298
    _MODELBUFFER._serialized_end = 327
    _MODELSTATUS._serialized_start = 329
    _MODELSTATUS._serialized_end = 358
    _AGENT._serialized_start = 361
    _AGENT._serialized_end = 1323
