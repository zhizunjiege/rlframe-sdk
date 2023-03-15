"""Generated protocol buffer code."""
from google.protobuf import descriptor as _descriptor
from google.protobuf import descriptor_pool as _descriptor_pool
from google.protobuf import message as _message
from google.protobuf import reflection as _reflection
from google.protobuf import symbol_database as _symbol_database

_sym_db = _symbol_database.Default()
from . import types_pb2 as types__pb2
from . import agent_pb2 as agent__pb2
from . import simenv_pb2 as simenv__pb2

DESCRIPTOR = _descriptor_pool.Default().AddSerializedFile(
    b'\n\tbff.proto\x12\x08game.bff\x1a\x0btypes.proto\x1a\x0bagent.proto\x1a\x0csimenv.proto"S\n\x0bServiceInfo\x12\x0c\n\x04type\x18\x01 \x01(\t\x12\x0c\n\x04name\x18\x02 \x01(\t\x12\x0c\n\x04host\x18\x03 \x01(\t\x12\x0c\n\x04port\x18\x04 \x01(\r\x12\x0c\n\x04desc\x18\x05 \x01(\t"\x1c\n\rServiceIdList\x12\x0b\n\x03ids\x18\x01 \x03(\t"\x92\x01\n\x0eServiceInfoMap\x128\n\x08services\x18\x01 \x03(\x0b2&.game.bff.ServiceInfoMap.ServicesEntry\x1aF\n\rServicesEntry\x12\x0b\n\x03key\x18\x01 \x01(\t\x12$\n\x05value\x18\x02 \x01(\x0b2\x15.game.bff.ServiceInfo:\x028\x01"\xb9\x02\n\x0bRouteConfig\x121\n\x06routes\x18\x01 \x03(\x0b2!.game.bff.RouteConfig.RoutesEntry\x1a\x18\n\x06Config\x12\x0e\n\x06models\x18\x01 \x03(\t\x1a\x90\x01\n\x05Route\x129\n\x07configs\x18\x01 \x03(\x0b2(.game.bff.RouteConfig.Route.ConfigsEntry\x1aL\n\x0cConfigsEntry\x12\x0b\n\x03key\x18\x01 \x01(\t\x12+\n\x05value\x18\x02 \x01(\x0b2\x1c.game.bff.RouteConfig.Config:\x028\x01\x1aJ\n\x0bRoutesEntry\x12\x0b\n\x03key\x18\x01 \x01(\t\x12*\n\x05value\x18\x02 \x01(\x0b2\x1b.game.bff.RouteConfig.Route:\x028\x01"\x91\x01\n\x0fServiceStateMap\x125\n\x06states\x18\x01 \x03(\x0b2%.game.bff.ServiceStateMap.StatesEntry\x1aG\n\x0bStatesEntry\x12\x0b\n\x03key\x18\x01 \x01(\t\x12\'\n\x05value\x18\x02 \x01(\x0b2\x18.game.types.ServiceState:\x028\x01"\x95\x01\n\x0fSimenvConfigMap\x127\n\x07configs\x18\x01 \x03(\x0b2&.game.bff.SimenvConfigMap.ConfigsEntry\x1aI\n\x0cConfigsEntry\x12\x0b\n\x03key\x18\x01 \x01(\t\x12(\n\x05value\x18\x02 \x01(\x0b2\x19.game.simenv.SimenvConfig:\x028\x01"z\n\tSimCmdMap\x12+\n\x04cmds\x18\x01 \x03(\x0b2\x1d.game.bff.SimCmdMap.CmdsEntry\x1a@\n\tCmdsEntry\x12\x0b\n\x03key\x18\x01 \x01(\t\x12"\n\x05value\x18\x02 \x01(\x0b2\x13.game.simenv.SimCmd:\x028\x01"\x80\x01\n\nSimInfoMap\x12.\n\x05infos\x18\x01 \x03(\x0b2\x1f.game.bff.SimInfoMap.InfosEntry\x1aB\n\nInfosEntry\x12\x0b\n\x03key\x18\x01 \x01(\t\x12#\n\x05value\x18\x02 \x01(\x0b2\x14.game.simenv.SimInfo:\x028\x01"\x91\x01\n\x0eAgentConfigMap\x126\n\x07configs\x18\x01 \x03(\x0b2%.game.bff.AgentConfigMap.ConfigsEntry\x1aG\n\x0cConfigsEntry\x12\x0b\n\x03key\x18\x01 \x01(\t\x12&\n\x05value\x18\x02 \x01(\x0b2\x17.game.agent.AgentConfig:\x028\x01"\x85\x01\n\x0cAgentModeMap\x120\n\x05modes\x18\x01 \x03(\x0b2!.game.bff.AgentModeMap.ModesEntry\x1aC\n\nModesEntry\x12\x0b\n\x03key\x18\x01 \x01(\t\x12$\n\x05value\x18\x02 \x01(\x0b2\x15.game.agent.AgentMode:\x028\x01"\x94\x01\n\x0fModelWeightsMap\x127\n\x07weights\x18\x01 \x03(\x0b2&.game.bff.ModelWeightsMap.WeightsEntry\x1aH\n\x0cWeightsEntry\x12\x0b\n\x03key\x18\x01 \x01(\t\x12\'\n\x05value\x18\x02 \x01(\x0b2\x18.game.agent.ModelWeights:\x028\x01"\x91\x01\n\x0eModelBufferMap\x126\n\x07buffers\x18\x01 \x03(\x0b2%.game.bff.ModelBufferMap.BuffersEntry\x1aG\n\x0cBuffersEntry\x12\x0b\n\x03key\x18\x01 \x01(\t\x12&\n\x05value\x18\x02 \x01(\x0b2\x17.game.agent.ModelBuffer:\x028\x01"\x8e\x01\n\x0eModelStatusMap\x124\n\x06status\x18\x01 \x03(\x0b2$.game.bff.ModelStatusMap.StatusEntry\x1aF\n\x0bStatusEntry\x12\x0b\n\x03key\x18\x01 \x01(\t\x12&\n\x05value\x18\x02 \x01(\x0b2\x17.game.agent.ModelStatus:\x028\x01"\x7f\n\x0bCallDataMap\x12-\n\x04data\x18\x01 \x03(\x0b2\x1f.game.bff.CallDataMap.DataEntry\x1aA\n\tDataEntry\x12\x0b\n\x03key\x18\x01 \x01(\t\x12#\n\x05value\x18\x02 \x01(\x0b2\x14.game.types.CallData:\x028\x012\xfd\x0c\n\x03BFF\x12D\n\x0bResetServer\x12\x19.game.types.CommonRequest\x1a\x1a.game.types.CommonResponse\x12G\n\x0fRegisterService\x12\x18.game.bff.ServiceInfoMap\x1a\x1a.game.types.CommonResponse\x12H\n\x11UnRegisterService\x12\x17.game.bff.ServiceIdList\x1a\x1a.game.types.CommonResponse\x12C\n\x0eGetServiceInfo\x12\x17.game.bff.ServiceIdList\x1a\x18.game.bff.ServiceInfoMap\x12F\n\x0eSetServiceInfo\x12\x18.game.bff.ServiceInfoMap\x1a\x1a.game.types.CommonResponse\x12B\n\x0eGetRouteConfig\x12\x19.game.types.CommonRequest\x1a\x15.game.bff.RouteConfig\x12C\n\x0eSetRouteConfig\x12\x15.game.bff.RouteConfig\x1a\x1a.game.types.CommonResponse\x12C\n\x0cResetService\x12\x17.game.bff.ServiceIdList\x1a\x1a.game.types.CommonResponse\x12B\n\x0cQueryService\x12\x17.game.bff.ServiceIdList\x1a\x19.game.bff.ServiceStateMap\x12E\n\x0fGetSimenvConfig\x12\x17.game.bff.ServiceIdList\x1a\x19.game.bff.SimenvConfigMap\x12H\n\x0fSetSimenvConfig\x12\x19.game.bff.SimenvConfigMap\x1a\x1a.game.types.CommonResponse\x12=\n\nSimControl\x12\x13.game.bff.SimCmdMap\x1a\x1a.game.types.CommonResponse\x12;\n\nSimMonitor\x12\x17.game.bff.ServiceIdList\x1a\x14.game.bff.SimInfoMap\x12C\n\x0eGetAgentConfig\x12\x17.game.bff.ServiceIdList\x1a\x18.game.bff.AgentConfigMap\x12F\n\x0eSetAgentConfig\x12\x18.game.bff.AgentConfigMap\x1a\x1a.game.types.CommonResponse\x12?\n\x0cGetAgentMode\x12\x17.game.bff.ServiceIdList\x1a\x16.game.bff.AgentModeMap\x12B\n\x0cSetAgentMode\x12\x16.game.bff.AgentModeMap\x1a\x1a.game.types.CommonResponse\x12E\n\x0fGetModelWeights\x12\x17.game.bff.ServiceIdList\x1a\x19.game.bff.ModelWeightsMap\x12H\n\x0fSetModelWeights\x12\x19.game.bff.ModelWeightsMap\x1a\x1a.game.types.CommonResponse\x12C\n\x0eGetModelBuffer\x12\x17.game.bff.ServiceIdList\x1a\x18.game.bff.ModelBufferMap\x12F\n\x0eSetModelBuffer\x12\x18.game.bff.ModelBufferMap\x1a\x1a.game.types.CommonResponse\x12C\n\x0eGetModelStatus\x12\x17.game.bff.ServiceIdList\x1a\x18.game.bff.ModelStatusMap\x12F\n\x0eSetModelStatus\x12\x18.game.bff.ModelStatusMap\x1a\x1a.game.types.CommonResponse\x124\n\x04Call\x12\x15.game.bff.CallDataMap\x1a\x15.game.bff.CallDataMapb\x06proto3'
)
_SERVICEINFO = DESCRIPTOR.message_types_by_name['ServiceInfo']
_SERVICEIDLIST = DESCRIPTOR.message_types_by_name['ServiceIdList']
_SERVICEINFOMAP = DESCRIPTOR.message_types_by_name['ServiceInfoMap']
_SERVICEINFOMAP_SERVICESENTRY = _SERVICEINFOMAP.nested_types_by_name['ServicesEntry']
_ROUTECONFIG = DESCRIPTOR.message_types_by_name['RouteConfig']
_ROUTECONFIG_CONFIG = _ROUTECONFIG.nested_types_by_name['Config']
_ROUTECONFIG_ROUTE = _ROUTECONFIG.nested_types_by_name['Route']
_ROUTECONFIG_ROUTE_CONFIGSENTRY = _ROUTECONFIG_ROUTE.nested_types_by_name['ConfigsEntry']
_ROUTECONFIG_ROUTESENTRY = _ROUTECONFIG.nested_types_by_name['RoutesEntry']
_SERVICESTATEMAP = DESCRIPTOR.message_types_by_name['ServiceStateMap']
_SERVICESTATEMAP_STATESENTRY = _SERVICESTATEMAP.nested_types_by_name['StatesEntry']
_SIMENVCONFIGMAP = DESCRIPTOR.message_types_by_name['SimenvConfigMap']
_SIMENVCONFIGMAP_CONFIGSENTRY = _SIMENVCONFIGMAP.nested_types_by_name['ConfigsEntry']
_SIMCMDMAP = DESCRIPTOR.message_types_by_name['SimCmdMap']
_SIMCMDMAP_CMDSENTRY = _SIMCMDMAP.nested_types_by_name['CmdsEntry']
_SIMINFOMAP = DESCRIPTOR.message_types_by_name['SimInfoMap']
_SIMINFOMAP_INFOSENTRY = _SIMINFOMAP.nested_types_by_name['InfosEntry']
_AGENTCONFIGMAP = DESCRIPTOR.message_types_by_name['AgentConfigMap']
_AGENTCONFIGMAP_CONFIGSENTRY = _AGENTCONFIGMAP.nested_types_by_name['ConfigsEntry']
_AGENTMODEMAP = DESCRIPTOR.message_types_by_name['AgentModeMap']
_AGENTMODEMAP_MODESENTRY = _AGENTMODEMAP.nested_types_by_name['ModesEntry']
_MODELWEIGHTSMAP = DESCRIPTOR.message_types_by_name['ModelWeightsMap']
_MODELWEIGHTSMAP_WEIGHTSENTRY = _MODELWEIGHTSMAP.nested_types_by_name['WeightsEntry']
_MODELBUFFERMAP = DESCRIPTOR.message_types_by_name['ModelBufferMap']
_MODELBUFFERMAP_BUFFERSENTRY = _MODELBUFFERMAP.nested_types_by_name['BuffersEntry']
_MODELSTATUSMAP = DESCRIPTOR.message_types_by_name['ModelStatusMap']
_MODELSTATUSMAP_STATUSENTRY = _MODELSTATUSMAP.nested_types_by_name['StatusEntry']
_CALLDATAMAP = DESCRIPTOR.message_types_by_name['CallDataMap']
_CALLDATAMAP_DATAENTRY = _CALLDATAMAP.nested_types_by_name['DataEntry']
ServiceInfo = _reflection.GeneratedProtocolMessageType('ServiceInfo', (_message.Message,), {
    'DESCRIPTOR': _SERVICEINFO,
    '__module__': 'bff_pb2'
})
_sym_db.RegisterMessage(ServiceInfo)
ServiceIdList = _reflection.GeneratedProtocolMessageType('ServiceIdList', (_message.Message,), {
    'DESCRIPTOR': _SERVICEIDLIST,
    '__module__': 'bff_pb2'
})
_sym_db.RegisterMessage(ServiceIdList)
ServiceInfoMap = _reflection.GeneratedProtocolMessageType(
    'ServiceInfoMap', (_message.Message,), {
        'ServicesEntry':
            _reflection.GeneratedProtocolMessageType('ServicesEntry', (_message.Message,), {
                'DESCRIPTOR': _SERVICEINFOMAP_SERVICESENTRY,
                '__module__': 'bff_pb2'
            }),
        'DESCRIPTOR':
            _SERVICEINFOMAP,
        '__module__':
            'bff_pb2'
    })
_sym_db.RegisterMessage(ServiceInfoMap)
_sym_db.RegisterMessage(ServiceInfoMap.ServicesEntry)
RouteConfig = _reflection.GeneratedProtocolMessageType(
    'RouteConfig', (_message.Message,), {
        'Config':
            _reflection.GeneratedProtocolMessageType('Config', (_message.Message,), {
                'DESCRIPTOR': _ROUTECONFIG_CONFIG,
                '__module__': 'bff_pb2'
            }),
        'Route':
            _reflection.GeneratedProtocolMessageType(
                'Route', (_message.Message,), {
                    'ConfigsEntry':
                        _reflection.GeneratedProtocolMessageType('ConfigsEntry', (_message.Message,), {
                            'DESCRIPTOR': _ROUTECONFIG_ROUTE_CONFIGSENTRY,
                            '__module__': 'bff_pb2'
                        }),
                    'DESCRIPTOR':
                        _ROUTECONFIG_ROUTE,
                    '__module__':
                        'bff_pb2'
                }),
        'RoutesEntry':
            _reflection.GeneratedProtocolMessageType('RoutesEntry', (_message.Message,), {
                'DESCRIPTOR': _ROUTECONFIG_ROUTESENTRY,
                '__module__': 'bff_pb2'
            }),
        'DESCRIPTOR':
            _ROUTECONFIG,
        '__module__':
            'bff_pb2'
    })
_sym_db.RegisterMessage(RouteConfig)
_sym_db.RegisterMessage(RouteConfig.Config)
_sym_db.RegisterMessage(RouteConfig.Route)
_sym_db.RegisterMessage(RouteConfig.Route.ConfigsEntry)
_sym_db.RegisterMessage(RouteConfig.RoutesEntry)
ServiceStateMap = _reflection.GeneratedProtocolMessageType(
    'ServiceStateMap', (_message.Message,), {
        'StatesEntry':
            _reflection.GeneratedProtocolMessageType('StatesEntry', (_message.Message,), {
                'DESCRIPTOR': _SERVICESTATEMAP_STATESENTRY,
                '__module__': 'bff_pb2'
            }),
        'DESCRIPTOR':
            _SERVICESTATEMAP,
        '__module__':
            'bff_pb2'
    })
_sym_db.RegisterMessage(ServiceStateMap)
_sym_db.RegisterMessage(ServiceStateMap.StatesEntry)
SimenvConfigMap = _reflection.GeneratedProtocolMessageType(
    'SimenvConfigMap', (_message.Message,), {
        'ConfigsEntry':
            _reflection.GeneratedProtocolMessageType('ConfigsEntry', (_message.Message,), {
                'DESCRIPTOR': _SIMENVCONFIGMAP_CONFIGSENTRY,
                '__module__': 'bff_pb2'
            }),
        'DESCRIPTOR':
            _SIMENVCONFIGMAP,
        '__module__':
            'bff_pb2'
    })
_sym_db.RegisterMessage(SimenvConfigMap)
_sym_db.RegisterMessage(SimenvConfigMap.ConfigsEntry)
SimCmdMap = _reflection.GeneratedProtocolMessageType(
    'SimCmdMap', (_message.Message,), {
        'CmdsEntry':
            _reflection.GeneratedProtocolMessageType('CmdsEntry', (_message.Message,), {
                'DESCRIPTOR': _SIMCMDMAP_CMDSENTRY,
                '__module__': 'bff_pb2'
            }),
        'DESCRIPTOR':
            _SIMCMDMAP,
        '__module__':
            'bff_pb2'
    })
_sym_db.RegisterMessage(SimCmdMap)
_sym_db.RegisterMessage(SimCmdMap.CmdsEntry)
SimInfoMap = _reflection.GeneratedProtocolMessageType(
    'SimInfoMap', (_message.Message,), {
        'InfosEntry':
            _reflection.GeneratedProtocolMessageType('InfosEntry', (_message.Message,), {
                'DESCRIPTOR': _SIMINFOMAP_INFOSENTRY,
                '__module__': 'bff_pb2'
            }),
        'DESCRIPTOR':
            _SIMINFOMAP,
        '__module__':
            'bff_pb2'
    })
_sym_db.RegisterMessage(SimInfoMap)
_sym_db.RegisterMessage(SimInfoMap.InfosEntry)
AgentConfigMap = _reflection.GeneratedProtocolMessageType(
    'AgentConfigMap', (_message.Message,), {
        'ConfigsEntry':
            _reflection.GeneratedProtocolMessageType('ConfigsEntry', (_message.Message,), {
                'DESCRIPTOR': _AGENTCONFIGMAP_CONFIGSENTRY,
                '__module__': 'bff_pb2'
            }),
        'DESCRIPTOR':
            _AGENTCONFIGMAP,
        '__module__':
            'bff_pb2'
    })
_sym_db.RegisterMessage(AgentConfigMap)
_sym_db.RegisterMessage(AgentConfigMap.ConfigsEntry)
AgentModeMap = _reflection.GeneratedProtocolMessageType(
    'AgentModeMap', (_message.Message,), {
        'ModesEntry':
            _reflection.GeneratedProtocolMessageType('ModesEntry', (_message.Message,), {
                'DESCRIPTOR': _AGENTMODEMAP_MODESENTRY,
                '__module__': 'bff_pb2'
            }),
        'DESCRIPTOR':
            _AGENTMODEMAP,
        '__module__':
            'bff_pb2'
    })
_sym_db.RegisterMessage(AgentModeMap)
_sym_db.RegisterMessage(AgentModeMap.ModesEntry)
ModelWeightsMap = _reflection.GeneratedProtocolMessageType(
    'ModelWeightsMap', (_message.Message,), {
        'WeightsEntry':
            _reflection.GeneratedProtocolMessageType('WeightsEntry', (_message.Message,), {
                'DESCRIPTOR': _MODELWEIGHTSMAP_WEIGHTSENTRY,
                '__module__': 'bff_pb2'
            }),
        'DESCRIPTOR':
            _MODELWEIGHTSMAP,
        '__module__':
            'bff_pb2'
    })
_sym_db.RegisterMessage(ModelWeightsMap)
_sym_db.RegisterMessage(ModelWeightsMap.WeightsEntry)
ModelBufferMap = _reflection.GeneratedProtocolMessageType(
    'ModelBufferMap', (_message.Message,), {
        'BuffersEntry':
            _reflection.GeneratedProtocolMessageType('BuffersEntry', (_message.Message,), {
                'DESCRIPTOR': _MODELBUFFERMAP_BUFFERSENTRY,
                '__module__': 'bff_pb2'
            }),
        'DESCRIPTOR':
            _MODELBUFFERMAP,
        '__module__':
            'bff_pb2'
    })
_sym_db.RegisterMessage(ModelBufferMap)
_sym_db.RegisterMessage(ModelBufferMap.BuffersEntry)
ModelStatusMap = _reflection.GeneratedProtocolMessageType(
    'ModelStatusMap', (_message.Message,), {
        'StatusEntry':
            _reflection.GeneratedProtocolMessageType('StatusEntry', (_message.Message,), {
                'DESCRIPTOR': _MODELSTATUSMAP_STATUSENTRY,
                '__module__': 'bff_pb2'
            }),
        'DESCRIPTOR':
            _MODELSTATUSMAP,
        '__module__':
            'bff_pb2'
    })
_sym_db.RegisterMessage(ModelStatusMap)
_sym_db.RegisterMessage(ModelStatusMap.StatusEntry)
CallDataMap = _reflection.GeneratedProtocolMessageType(
    'CallDataMap', (_message.Message,), {
        'DataEntry':
            _reflection.GeneratedProtocolMessageType('DataEntry', (_message.Message,), {
                'DESCRIPTOR': _CALLDATAMAP_DATAENTRY,
                '__module__': 'bff_pb2'
            }),
        'DESCRIPTOR':
            _CALLDATAMAP,
        '__module__':
            'bff_pb2'
    })
_sym_db.RegisterMessage(CallDataMap)
_sym_db.RegisterMessage(CallDataMap.DataEntry)
_BFF = DESCRIPTOR.services_by_name['BFF']
if _descriptor._USE_C_DESCRIPTORS == False:
    DESCRIPTOR._options = None
    _SERVICEINFOMAP_SERVICESENTRY._options = None
    _SERVICEINFOMAP_SERVICESENTRY._serialized_options = b'8\x01'
    _ROUTECONFIG_ROUTE_CONFIGSENTRY._options = None
    _ROUTECONFIG_ROUTE_CONFIGSENTRY._serialized_options = b'8\x01'
    _ROUTECONFIG_ROUTESENTRY._options = None
    _ROUTECONFIG_ROUTESENTRY._serialized_options = b'8\x01'
    _SERVICESTATEMAP_STATESENTRY._options = None
    _SERVICESTATEMAP_STATESENTRY._serialized_options = b'8\x01'
    _SIMENVCONFIGMAP_CONFIGSENTRY._options = None
    _SIMENVCONFIGMAP_CONFIGSENTRY._serialized_options = b'8\x01'
    _SIMCMDMAP_CMDSENTRY._options = None
    _SIMCMDMAP_CMDSENTRY._serialized_options = b'8\x01'
    _SIMINFOMAP_INFOSENTRY._options = None
    _SIMINFOMAP_INFOSENTRY._serialized_options = b'8\x01'
    _AGENTCONFIGMAP_CONFIGSENTRY._options = None
    _AGENTCONFIGMAP_CONFIGSENTRY._serialized_options = b'8\x01'
    _AGENTMODEMAP_MODESENTRY._options = None
    _AGENTMODEMAP_MODESENTRY._serialized_options = b'8\x01'
    _MODELWEIGHTSMAP_WEIGHTSENTRY._options = None
    _MODELWEIGHTSMAP_WEIGHTSENTRY._serialized_options = b'8\x01'
    _MODELBUFFERMAP_BUFFERSENTRY._options = None
    _MODELBUFFERMAP_BUFFERSENTRY._serialized_options = b'8\x01'
    _MODELSTATUSMAP_STATUSENTRY._options = None
    _MODELSTATUSMAP_STATUSENTRY._serialized_options = b'8\x01'
    _CALLDATAMAP_DATAENTRY._options = None
    _CALLDATAMAP_DATAENTRY._serialized_options = b'8\x01'
    _SERVICEINFO._serialized_start = 63
    _SERVICEINFO._serialized_end = 146
    _SERVICEIDLIST._serialized_start = 148
    _SERVICEIDLIST._serialized_end = 176
    _SERVICEINFOMAP._serialized_start = 179
    _SERVICEINFOMAP._serialized_end = 325
    _SERVICEINFOMAP_SERVICESENTRY._serialized_start = 255
    _SERVICEINFOMAP_SERVICESENTRY._serialized_end = 325
    _ROUTECONFIG._serialized_start = 328
    _ROUTECONFIG._serialized_end = 641
    _ROUTECONFIG_CONFIG._serialized_start = 394
    _ROUTECONFIG_CONFIG._serialized_end = 418
    _ROUTECONFIG_ROUTE._serialized_start = 421
    _ROUTECONFIG_ROUTE._serialized_end = 565
    _ROUTECONFIG_ROUTE_CONFIGSENTRY._serialized_start = 489
    _ROUTECONFIG_ROUTE_CONFIGSENTRY._serialized_end = 565
    _ROUTECONFIG_ROUTESENTRY._serialized_start = 567
    _ROUTECONFIG_ROUTESENTRY._serialized_end = 641
    _SERVICESTATEMAP._serialized_start = 644
    _SERVICESTATEMAP._serialized_end = 789
    _SERVICESTATEMAP_STATESENTRY._serialized_start = 718
    _SERVICESTATEMAP_STATESENTRY._serialized_end = 789
    _SIMENVCONFIGMAP._serialized_start = 792
    _SIMENVCONFIGMAP._serialized_end = 941
    _SIMENVCONFIGMAP_CONFIGSENTRY._serialized_start = 868
    _SIMENVCONFIGMAP_CONFIGSENTRY._serialized_end = 941
    _SIMCMDMAP._serialized_start = 943
    _SIMCMDMAP._serialized_end = 1065
    _SIMCMDMAP_CMDSENTRY._serialized_start = 1001
    _SIMCMDMAP_CMDSENTRY._serialized_end = 1065
    _SIMINFOMAP._serialized_start = 1068
    _SIMINFOMAP._serialized_end = 1196
    _SIMINFOMAP_INFOSENTRY._serialized_start = 1130
    _SIMINFOMAP_INFOSENTRY._serialized_end = 1196
    _AGENTCONFIGMAP._serialized_start = 1199
    _AGENTCONFIGMAP._serialized_end = 1344
    _AGENTCONFIGMAP_CONFIGSENTRY._serialized_start = 1273
    _AGENTCONFIGMAP_CONFIGSENTRY._serialized_end = 1344
    _AGENTMODEMAP._serialized_start = 1347
    _AGENTMODEMAP._serialized_end = 1480
    _AGENTMODEMAP_MODESENTRY._serialized_start = 1413
    _AGENTMODEMAP_MODESENTRY._serialized_end = 1480
    _MODELWEIGHTSMAP._serialized_start = 1483
    _MODELWEIGHTSMAP._serialized_end = 1631
    _MODELWEIGHTSMAP_WEIGHTSENTRY._serialized_start = 1559
    _MODELWEIGHTSMAP_WEIGHTSENTRY._serialized_end = 1631
    _MODELBUFFERMAP._serialized_start = 1634
    _MODELBUFFERMAP._serialized_end = 1779
    _MODELBUFFERMAP_BUFFERSENTRY._serialized_start = 1708
    _MODELBUFFERMAP_BUFFERSENTRY._serialized_end = 1779
    _MODELSTATUSMAP._serialized_start = 1782
    _MODELSTATUSMAP._serialized_end = 1924
    _MODELSTATUSMAP_STATUSENTRY._serialized_start = 1854
    _MODELSTATUSMAP_STATUSENTRY._serialized_end = 1924
    _CALLDATAMAP._serialized_start = 1926
    _CALLDATAMAP._serialized_end = 2053
    _CALLDATAMAP_DATAENTRY._serialized_start = 1988
    _CALLDATAMAP_DATAENTRY._serialized_end = 2053
    _BFF._serialized_start = 2056
    _BFF._serialized_end = 3717
