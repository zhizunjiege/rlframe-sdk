"""Generated protocol buffer code."""
from google.protobuf import descriptor as _descriptor
from google.protobuf import descriptor_pool as _descriptor_pool
from google.protobuf import message as _message
from google.protobuf import reflection as _reflection
from google.protobuf import symbol_database as _symbol_database

_sym_db = _symbol_database.Default()
DESCRIPTOR = _descriptor_pool.Default().AddSerializedFile(
    b'\n\x0btypes.proto\x12\ngame.types"\x0f\n\rCommonRequest"\x10\n\x0eCommonResponse"4\n\x08CallData\x12\x0c\n\x04name\x18\x01 \x01(\t\x12\x0c\n\x04dstr\x18\x02 \x01(\t\x12\x0c\n\x04dbin\x18\x03 \x01(\x0c"`\n\x0cServiceState\x12-\n\x05state\x18\x01 \x01(\x0e2\x1e.game.types.ServiceState.State"!\n\x05State\x12\x0c\n\x08UNINITED\x10\x00\x12\n\n\x06INITED\x10\x01"\xf1\x02\n\x08SimParam\x12\x11\n\x07vdouble\x18\x01 \x01(\x01H\x00\x12\x10\n\x06vint32\x18\x02 \x01(\x05H\x00\x12\x0f\n\x05vbool\x18\x03 \x01(\x08H\x00\x12\x11\n\x07vstring\x18\x04 \x01(\tH\x00\x12,\n\x06varray\x18\x05 \x01(\x0b2\x1a.game.types.SimParam.ArrayH\x00\x12.\n\x07vstruct\x18\x06 \x01(\x0b2\x1b.game.types.SimParam.StructH\x00\x1a,\n\x05Array\x12#\n\x05items\x18\x01 \x03(\x0b2\x14.game.types.SimParam\x1a\x86\x01\n\x06Struct\x127\n\x06fields\x18\x01 \x03(\x0b2\'.game.types.SimParam.Struct.FieldsEntry\x1aC\n\x0bFieldsEntry\x12\x0b\n\x03key\x18\x01 \x01(\t\x12#\n\x05value\x18\x02 \x01(\x0b2\x14.game.types.SimParam:\x028\x01B\x07\n\x05value"\x83\x01\n\tSimEntity\x121\n\x06params\x18\x01 \x03(\x0b2!.game.types.SimEntity.ParamsEntry\x1aC\n\x0bParamsEntry\x12\x0b\n\x03key\x18\x01 \x01(\t\x12#\n\x05value\x18\x02 \x01(\x0b2\x14.game.types.SimParam:\x028\x01"3\n\x08SimModel\x12\'\n\x08entities\x18\x01 \x03(\x0b2\x15.game.types.SimEntity"\xb8\x01\n\x08SimState\x120\n\x06states\x18\x01 \x03(\x0b2 .game.types.SimState.StatesEntry\x12\x12\n\nterminated\x18\x02 \x01(\x08\x12\x11\n\ttruncated\x18\x03 \x01(\x08\x12\x0e\n\x06reward\x18\x04 \x01(\x01\x1aC\n\x0bStatesEntry\x12\x0b\n\x03key\x18\x01 \x01(\t\x12#\n\x05value\x18\x02 \x01(\x0b2\x14.game.types.SimModel:\x028\x01"\x86\x01\n\tSimAction\x123\n\x07actions\x18\x01 \x03(\x0b2".game.types.SimAction.ActionsEntry\x1aD\n\x0cActionsEntry\x12\x0b\n\x03key\x18\x01 \x01(\t\x12#\n\x05value\x18\x02 \x01(\x0b2\x14.game.types.SimModel:\x028\x01b\x06proto3'
)
_COMMONREQUEST = DESCRIPTOR.message_types_by_name['CommonRequest']
_COMMONRESPONSE = DESCRIPTOR.message_types_by_name['CommonResponse']
_CALLDATA = DESCRIPTOR.message_types_by_name['CallData']
_SERVICESTATE = DESCRIPTOR.message_types_by_name['ServiceState']
_SIMPARAM = DESCRIPTOR.message_types_by_name['SimParam']
_SIMPARAM_ARRAY = _SIMPARAM.nested_types_by_name['Array']
_SIMPARAM_STRUCT = _SIMPARAM.nested_types_by_name['Struct']
_SIMPARAM_STRUCT_FIELDSENTRY = _SIMPARAM_STRUCT.nested_types_by_name['FieldsEntry']
_SIMENTITY = DESCRIPTOR.message_types_by_name['SimEntity']
_SIMENTITY_PARAMSENTRY = _SIMENTITY.nested_types_by_name['ParamsEntry']
_SIMMODEL = DESCRIPTOR.message_types_by_name['SimModel']
_SIMSTATE = DESCRIPTOR.message_types_by_name['SimState']
_SIMSTATE_STATESENTRY = _SIMSTATE.nested_types_by_name['StatesEntry']
_SIMACTION = DESCRIPTOR.message_types_by_name['SimAction']
_SIMACTION_ACTIONSENTRY = _SIMACTION.nested_types_by_name['ActionsEntry']
_SERVICESTATE_STATE = _SERVICESTATE.enum_types_by_name['State']
CommonRequest = _reflection.GeneratedProtocolMessageType('CommonRequest', (_message.Message,), {
    'DESCRIPTOR': _COMMONREQUEST,
    '__module__': 'types_pb2'
})
_sym_db.RegisterMessage(CommonRequest)
CommonResponse = _reflection.GeneratedProtocolMessageType('CommonResponse', (_message.Message,), {
    'DESCRIPTOR': _COMMONRESPONSE,
    '__module__': 'types_pb2'
})
_sym_db.RegisterMessage(CommonResponse)
CallData = _reflection.GeneratedProtocolMessageType('CallData', (_message.Message,), {
    'DESCRIPTOR': _CALLDATA,
    '__module__': 'types_pb2'
})
_sym_db.RegisterMessage(CallData)
ServiceState = _reflection.GeneratedProtocolMessageType('ServiceState', (_message.Message,), {
    'DESCRIPTOR': _SERVICESTATE,
    '__module__': 'types_pb2'
})
_sym_db.RegisterMessage(ServiceState)
SimParam = _reflection.GeneratedProtocolMessageType(
    'SimParam', (_message.Message,), {
        'Array':
            _reflection.GeneratedProtocolMessageType('Array', (_message.Message,), {
                'DESCRIPTOR': _SIMPARAM_ARRAY,
                '__module__': 'types_pb2'
            }),
        'Struct':
            _reflection.GeneratedProtocolMessageType(
                'Struct', (_message.Message,), {
                    'FieldsEntry':
                        _reflection.GeneratedProtocolMessageType('FieldsEntry', (_message.Message,), {
                            'DESCRIPTOR': _SIMPARAM_STRUCT_FIELDSENTRY,
                            '__module__': 'types_pb2'
                        }),
                    'DESCRIPTOR':
                        _SIMPARAM_STRUCT,
                    '__module__':
                        'types_pb2'
                }),
        'DESCRIPTOR':
            _SIMPARAM,
        '__module__':
            'types_pb2'
    })
_sym_db.RegisterMessage(SimParam)
_sym_db.RegisterMessage(SimParam.Array)
_sym_db.RegisterMessage(SimParam.Struct)
_sym_db.RegisterMessage(SimParam.Struct.FieldsEntry)
SimEntity = _reflection.GeneratedProtocolMessageType(
    'SimEntity', (_message.Message,), {
        'ParamsEntry':
            _reflection.GeneratedProtocolMessageType('ParamsEntry', (_message.Message,), {
                'DESCRIPTOR': _SIMENTITY_PARAMSENTRY,
                '__module__': 'types_pb2'
            }),
        'DESCRIPTOR':
            _SIMENTITY,
        '__module__':
            'types_pb2'
    })
_sym_db.RegisterMessage(SimEntity)
_sym_db.RegisterMessage(SimEntity.ParamsEntry)
SimModel = _reflection.GeneratedProtocolMessageType('SimModel', (_message.Message,), {
    'DESCRIPTOR': _SIMMODEL,
    '__module__': 'types_pb2'
})
_sym_db.RegisterMessage(SimModel)
SimState = _reflection.GeneratedProtocolMessageType(
    'SimState', (_message.Message,), {
        'StatesEntry':
            _reflection.GeneratedProtocolMessageType('StatesEntry', (_message.Message,), {
                'DESCRIPTOR': _SIMSTATE_STATESENTRY,
                '__module__': 'types_pb2'
            }),
        'DESCRIPTOR':
            _SIMSTATE,
        '__module__':
            'types_pb2'
    })
_sym_db.RegisterMessage(SimState)
_sym_db.RegisterMessage(SimState.StatesEntry)
SimAction = _reflection.GeneratedProtocolMessageType(
    'SimAction', (_message.Message,), {
        'ActionsEntry':
            _reflection.GeneratedProtocolMessageType('ActionsEntry', (_message.Message,), {
                'DESCRIPTOR': _SIMACTION_ACTIONSENTRY,
                '__module__': 'types_pb2'
            }),
        'DESCRIPTOR':
            _SIMACTION,
        '__module__':
            'types_pb2'
    })
_sym_db.RegisterMessage(SimAction)
_sym_db.RegisterMessage(SimAction.ActionsEntry)
if _descriptor._USE_C_DESCRIPTORS == False:
    DESCRIPTOR._options = None
    _SIMPARAM_STRUCT_FIELDSENTRY._options = None
    _SIMPARAM_STRUCT_FIELDSENTRY._serialized_options = b'8\x01'
    _SIMENTITY_PARAMSENTRY._options = None
    _SIMENTITY_PARAMSENTRY._serialized_options = b'8\x01'
    _SIMSTATE_STATESENTRY._options = None
    _SIMSTATE_STATESENTRY._serialized_options = b'8\x01'
    _SIMACTION_ACTIONSENTRY._options = None
    _SIMACTION_ACTIONSENTRY._serialized_options = b'8\x01'
    _COMMONREQUEST._serialized_start = 27
    _COMMONREQUEST._serialized_end = 42
    _COMMONRESPONSE._serialized_start = 44
    _COMMONRESPONSE._serialized_end = 60
    _CALLDATA._serialized_start = 62
    _CALLDATA._serialized_end = 114
    _SERVICESTATE._serialized_start = 116
    _SERVICESTATE._serialized_end = 212
    _SERVICESTATE_STATE._serialized_start = 179
    _SERVICESTATE_STATE._serialized_end = 212
    _SIMPARAM._serialized_start = 215
    _SIMPARAM._serialized_end = 584
    _SIMPARAM_ARRAY._serialized_start = 394
    _SIMPARAM_ARRAY._serialized_end = 438
    _SIMPARAM_STRUCT._serialized_start = 441
    _SIMPARAM_STRUCT._serialized_end = 575
    _SIMPARAM_STRUCT_FIELDSENTRY._serialized_start = 508
    _SIMPARAM_STRUCT_FIELDSENTRY._serialized_end = 575
    _SIMENTITY._serialized_start = 587
    _SIMENTITY._serialized_end = 718
    _SIMENTITY_PARAMSENTRY._serialized_start = 651
    _SIMENTITY_PARAMSENTRY._serialized_end = 718
    _SIMMODEL._serialized_start = 720
    _SIMMODEL._serialized_end = 771
    _SIMSTATE._serialized_start = 774
    _SIMSTATE._serialized_end = 958
    _SIMSTATE_STATESENTRY._serialized_start = 891
    _SIMSTATE_STATESENTRY._serialized_end = 958
    _SIMACTION._serialized_start = 961
    _SIMACTION._serialized_end = 1095
    _SIMACTION_ACTIONSENTRY._serialized_start = 1027
    _SIMACTION_ACTIONSENTRY._serialized_end = 1095
