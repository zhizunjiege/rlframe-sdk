"""Client and server classes corresponding to protobuf-defined services."""
import grpc
from . import bff_pb2 as bff__pb2
from . import types_pb2 as types__pb2


class BFFStub(object):
    """Missing associated documentation comment in .proto file."""

    def __init__(self, channel):
        """Constructor.

        Args:
            channel: A grpc.Channel.
        """
        self.ResetServer = channel.unary_unary('/game.bff.BFF/ResetServer',
                                               request_serializer=types__pb2.CommonRequest.SerializeToString,
                                               response_deserializer=types__pb2.CommonResponse.FromString)
        self.RegisterService = channel.unary_unary('/game.bff.BFF/RegisterService',
                                                   request_serializer=bff__pb2.ServiceInfoMap.SerializeToString,
                                                   response_deserializer=types__pb2.CommonResponse.FromString)
        self.UnRegisterService = channel.unary_unary('/game.bff.BFF/UnRegisterService',
                                                     request_serializer=bff__pb2.ServiceIdList.SerializeToString,
                                                     response_deserializer=types__pb2.CommonResponse.FromString)
        self.GetServiceInfo = channel.unary_unary('/game.bff.BFF/GetServiceInfo',
                                                  request_serializer=bff__pb2.ServiceIdList.SerializeToString,
                                                  response_deserializer=bff__pb2.ServiceInfoMap.FromString)
        self.SetServiceInfo = channel.unary_unary('/game.bff.BFF/SetServiceInfo',
                                                  request_serializer=bff__pb2.ServiceInfoMap.SerializeToString,
                                                  response_deserializer=types__pb2.CommonResponse.FromString)
        self.ResetService = channel.unary_unary('/game.bff.BFF/ResetService',
                                                request_serializer=bff__pb2.ServiceIdList.SerializeToString,
                                                response_deserializer=types__pb2.CommonResponse.FromString)
        self.QueryService = channel.unary_unary('/game.bff.BFF/QueryService',
                                                request_serializer=bff__pb2.ServiceIdList.SerializeToString,
                                                response_deserializer=bff__pb2.ServiceStateMap.FromString)
        self.GetSimenvConfig = channel.unary_unary('/game.bff.BFF/GetSimenvConfig',
                                                   request_serializer=bff__pb2.ServiceIdList.SerializeToString,
                                                   response_deserializer=bff__pb2.SimenvConfigMap.FromString)
        self.SetSimenvConfig = channel.unary_unary('/game.bff.BFF/SetSimenvConfig',
                                                   request_serializer=bff__pb2.SimenvConfigMap.SerializeToString,
                                                   response_deserializer=types__pb2.CommonResponse.FromString)
        self.SimControl = channel.unary_unary('/game.bff.BFF/SimControl',
                                              request_serializer=bff__pb2.SimCmdMap.SerializeToString,
                                              response_deserializer=types__pb2.CommonResponse.FromString)
        self.SimMonitor = channel.unary_unary('/game.bff.BFF/SimMonitor',
                                              request_serializer=bff__pb2.ServiceIdList.SerializeToString,
                                              response_deserializer=bff__pb2.SimInfoMap.FromString)
        self.GetAgentConfig = channel.unary_unary('/game.bff.BFF/GetAgentConfig',
                                                  request_serializer=bff__pb2.ServiceIdList.SerializeToString,
                                                  response_deserializer=bff__pb2.AgentConfigMap.FromString)
        self.SetAgentConfig = channel.unary_unary('/game.bff.BFF/SetAgentConfig',
                                                  request_serializer=bff__pb2.AgentConfigMap.SerializeToString,
                                                  response_deserializer=types__pb2.CommonResponse.FromString)
        self.GetAgentMode = channel.unary_unary('/game.bff.BFF/GetAgentMode',
                                                request_serializer=bff__pb2.ServiceIdList.SerializeToString,
                                                response_deserializer=bff__pb2.AgentModeMap.FromString)
        self.SetAgentMode = channel.unary_unary('/game.bff.BFF/SetAgentMode',
                                                request_serializer=bff__pb2.AgentModeMap.SerializeToString,
                                                response_deserializer=types__pb2.CommonResponse.FromString)
        self.GetModelWeights = channel.unary_unary('/game.bff.BFF/GetModelWeights',
                                                   request_serializer=bff__pb2.ServiceIdList.SerializeToString,
                                                   response_deserializer=bff__pb2.ModelWeightsMap.FromString)
        self.SetModelWeights = channel.unary_unary('/game.bff.BFF/SetModelWeights',
                                                   request_serializer=bff__pb2.ModelWeightsMap.SerializeToString,
                                                   response_deserializer=types__pb2.CommonResponse.FromString)
        self.GetModelBuffer = channel.unary_unary('/game.bff.BFF/GetModelBuffer',
                                                  request_serializer=bff__pb2.ServiceIdList.SerializeToString,
                                                  response_deserializer=bff__pb2.ModelBufferMap.FromString)
        self.SetModelBuffer = channel.unary_unary('/game.bff.BFF/SetModelBuffer',
                                                  request_serializer=bff__pb2.ModelBufferMap.SerializeToString,
                                                  response_deserializer=types__pb2.CommonResponse.FromString)
        self.GetModelStatus = channel.unary_unary('/game.bff.BFF/GetModelStatus',
                                                  request_serializer=bff__pb2.ServiceIdList.SerializeToString,
                                                  response_deserializer=bff__pb2.ModelStatusMap.FromString)
        self.SetModelStatus = channel.unary_unary('/game.bff.BFF/SetModelStatus',
                                                  request_serializer=bff__pb2.ModelStatusMap.SerializeToString,
                                                  response_deserializer=types__pb2.CommonResponse.FromString)
        self.Call = channel.unary_unary('/game.bff.BFF/Call',
                                        request_serializer=bff__pb2.CallDataMap.SerializeToString,
                                        response_deserializer=bff__pb2.CallDataMap.FromString)


class BFFServicer(object):
    """Missing associated documentation comment in .proto file."""

    def ResetServer(self, request, context):
        """reset bff server
        """
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')

    def RegisterService(self, request, context):
        """register services
        """
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')

    def UnRegisterService(self, request, context):
        """unregister services
        """
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')

    def GetServiceInfo(self, request, context):
        """get services info
        """
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')

    def SetServiceInfo(self, request, context):
        """set services info
        """
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')

    def ResetService(self, request, context):
        """reset simenv/agent services state
        """
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')

    def QueryService(self, request, context):
        """start simenv/agent services state
        """
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')

    def GetSimenvConfig(self, request, context):
        """get simenv configs
        """
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')

    def SetSimenvConfig(self, request, context):
        """set simenv configs
        """
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')

    def SimControl(self, request, context):
        """control simenv
        """
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')

    def SimMonitor(self, request, context):
        """get simenv info
        """
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')

    def GetAgentConfig(self, request, context):
        """get agent configs
        """
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')

    def SetAgentConfig(self, request, context):
        """set agent configs
        """
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')

    def GetAgentMode(self, request, context):
        """get agent mode
        """
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')

    def SetAgentMode(self, request, context):
        """set agent mode
        """
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')

    def GetModelWeights(self, request, context):
        """get model weights
        """
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')

    def SetModelWeights(self, request, context):
        """set model weights
        """
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')

    def GetModelBuffer(self, request, context):
        """get model buffer
        """
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')

    def SetModelBuffer(self, request, context):
        """set model buffer
        """
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')

    def GetModelStatus(self, request, context):
        """get model status
        """
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')

    def SetModelStatus(self, request, context):
        """set model status
        """
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')

    def Call(self, request, context):
        """any rpc call
        """
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')


def add_BFFServicer_to_server(servicer, server):
    rpc_method_handlers = {
        'ResetServer':
            grpc.unary_unary_rpc_method_handler(servicer.ResetServer,
                                                request_deserializer=types__pb2.CommonRequest.FromString,
                                                response_serializer=types__pb2.CommonResponse.SerializeToString),
        'RegisterService':
            grpc.unary_unary_rpc_method_handler(servicer.RegisterService,
                                                request_deserializer=bff__pb2.ServiceInfoMap.FromString,
                                                response_serializer=types__pb2.CommonResponse.SerializeToString),
        'UnRegisterService':
            grpc.unary_unary_rpc_method_handler(servicer.UnRegisterService,
                                                request_deserializer=bff__pb2.ServiceIdList.FromString,
                                                response_serializer=types__pb2.CommonResponse.SerializeToString),
        'GetServiceInfo':
            grpc.unary_unary_rpc_method_handler(servicer.GetServiceInfo,
                                                request_deserializer=bff__pb2.ServiceIdList.FromString,
                                                response_serializer=bff__pb2.ServiceInfoMap.SerializeToString),
        'SetServiceInfo':
            grpc.unary_unary_rpc_method_handler(servicer.SetServiceInfo,
                                                request_deserializer=bff__pb2.ServiceInfoMap.FromString,
                                                response_serializer=types__pb2.CommonResponse.SerializeToString),
        'ResetService':
            grpc.unary_unary_rpc_method_handler(servicer.ResetService,
                                                request_deserializer=bff__pb2.ServiceIdList.FromString,
                                                response_serializer=types__pb2.CommonResponse.SerializeToString),
        'QueryService':
            grpc.unary_unary_rpc_method_handler(servicer.QueryService,
                                                request_deserializer=bff__pb2.ServiceIdList.FromString,
                                                response_serializer=bff__pb2.ServiceStateMap.SerializeToString),
        'GetSimenvConfig':
            grpc.unary_unary_rpc_method_handler(servicer.GetSimenvConfig,
                                                request_deserializer=bff__pb2.ServiceIdList.FromString,
                                                response_serializer=bff__pb2.SimenvConfigMap.SerializeToString),
        'SetSimenvConfig':
            grpc.unary_unary_rpc_method_handler(servicer.SetSimenvConfig,
                                                request_deserializer=bff__pb2.SimenvConfigMap.FromString,
                                                response_serializer=types__pb2.CommonResponse.SerializeToString),
        'SimControl':
            grpc.unary_unary_rpc_method_handler(servicer.SimControl,
                                                request_deserializer=bff__pb2.SimCmdMap.FromString,
                                                response_serializer=types__pb2.CommonResponse.SerializeToString),
        'SimMonitor':
            grpc.unary_unary_rpc_method_handler(servicer.SimMonitor,
                                                request_deserializer=bff__pb2.ServiceIdList.FromString,
                                                response_serializer=bff__pb2.SimInfoMap.SerializeToString),
        'GetAgentConfig':
            grpc.unary_unary_rpc_method_handler(servicer.GetAgentConfig,
                                                request_deserializer=bff__pb2.ServiceIdList.FromString,
                                                response_serializer=bff__pb2.AgentConfigMap.SerializeToString),
        'SetAgentConfig':
            grpc.unary_unary_rpc_method_handler(servicer.SetAgentConfig,
                                                request_deserializer=bff__pb2.AgentConfigMap.FromString,
                                                response_serializer=types__pb2.CommonResponse.SerializeToString),
        'GetAgentMode':
            grpc.unary_unary_rpc_method_handler(servicer.GetAgentMode,
                                                request_deserializer=bff__pb2.ServiceIdList.FromString,
                                                response_serializer=bff__pb2.AgentModeMap.SerializeToString),
        'SetAgentMode':
            grpc.unary_unary_rpc_method_handler(servicer.SetAgentMode,
                                                request_deserializer=bff__pb2.AgentModeMap.FromString,
                                                response_serializer=types__pb2.CommonResponse.SerializeToString),
        'GetModelWeights':
            grpc.unary_unary_rpc_method_handler(servicer.GetModelWeights,
                                                request_deserializer=bff__pb2.ServiceIdList.FromString,
                                                response_serializer=bff__pb2.ModelWeightsMap.SerializeToString),
        'SetModelWeights':
            grpc.unary_unary_rpc_method_handler(servicer.SetModelWeights,
                                                request_deserializer=bff__pb2.ModelWeightsMap.FromString,
                                                response_serializer=types__pb2.CommonResponse.SerializeToString),
        'GetModelBuffer':
            grpc.unary_unary_rpc_method_handler(servicer.GetModelBuffer,
                                                request_deserializer=bff__pb2.ServiceIdList.FromString,
                                                response_serializer=bff__pb2.ModelBufferMap.SerializeToString),
        'SetModelBuffer':
            grpc.unary_unary_rpc_method_handler(servicer.SetModelBuffer,
                                                request_deserializer=bff__pb2.ModelBufferMap.FromString,
                                                response_serializer=types__pb2.CommonResponse.SerializeToString),
        'GetModelStatus':
            grpc.unary_unary_rpc_method_handler(servicer.GetModelStatus,
                                                request_deserializer=bff__pb2.ServiceIdList.FromString,
                                                response_serializer=bff__pb2.ModelStatusMap.SerializeToString),
        'SetModelStatus':
            grpc.unary_unary_rpc_method_handler(servicer.SetModelStatus,
                                                request_deserializer=bff__pb2.ModelStatusMap.FromString,
                                                response_serializer=types__pb2.CommonResponse.SerializeToString),
        'Call':
            grpc.unary_unary_rpc_method_handler(servicer.Call,
                                                request_deserializer=bff__pb2.CallDataMap.FromString,
                                                response_serializer=bff__pb2.CallDataMap.SerializeToString)
    }
    generic_handler = grpc.method_handlers_generic_handler('game.bff.BFF', rpc_method_handlers)
    server.add_generic_rpc_handlers((generic_handler,))


class BFF(object):
    """Missing associated documentation comment in .proto file."""

    @staticmethod
    def ResetServer(request,
                    target,
                    options=(),
                    channel_credentials=None,
                    call_credentials=None,
                    insecure=False,
                    compression=None,
                    wait_for_ready=None,
                    timeout=None,
                    metadata=None):
        return grpc.experimental.unary_unary(request, target, '/game.bff.BFF/ResetServer',
                                             types__pb2.CommonRequest.SerializeToString, types__pb2.CommonResponse.FromString,
                                             options, channel_credentials, insecure, call_credentials, compression,
                                             wait_for_ready, timeout, metadata)

    @staticmethod
    def RegisterService(request,
                        target,
                        options=(),
                        channel_credentials=None,
                        call_credentials=None,
                        insecure=False,
                        compression=None,
                        wait_for_ready=None,
                        timeout=None,
                        metadata=None):
        return grpc.experimental.unary_unary(request, target, '/game.bff.BFF/RegisterService',
                                             bff__pb2.ServiceInfoMap.SerializeToString, types__pb2.CommonResponse.FromString,
                                             options, channel_credentials, insecure, call_credentials, compression,
                                             wait_for_ready, timeout, metadata)

    @staticmethod
    def UnRegisterService(request,
                          target,
                          options=(),
                          channel_credentials=None,
                          call_credentials=None,
                          insecure=False,
                          compression=None,
                          wait_for_ready=None,
                          timeout=None,
                          metadata=None):
        return grpc.experimental.unary_unary(request, target, '/game.bff.BFF/UnRegisterService',
                                             bff__pb2.ServiceIdList.SerializeToString, types__pb2.CommonResponse.FromString,
                                             options, channel_credentials, insecure, call_credentials, compression,
                                             wait_for_ready, timeout, metadata)

    @staticmethod
    def GetServiceInfo(request,
                       target,
                       options=(),
                       channel_credentials=None,
                       call_credentials=None,
                       insecure=False,
                       compression=None,
                       wait_for_ready=None,
                       timeout=None,
                       metadata=None):
        return grpc.experimental.unary_unary(request, target, '/game.bff.BFF/GetServiceInfo',
                                             bff__pb2.ServiceIdList.SerializeToString, bff__pb2.ServiceInfoMap.FromString,
                                             options, channel_credentials, insecure, call_credentials, compression,
                                             wait_for_ready, timeout, metadata)

    @staticmethod
    def SetServiceInfo(request,
                       target,
                       options=(),
                       channel_credentials=None,
                       call_credentials=None,
                       insecure=False,
                       compression=None,
                       wait_for_ready=None,
                       timeout=None,
                       metadata=None):
        return grpc.experimental.unary_unary(request, target, '/game.bff.BFF/SetServiceInfo',
                                             bff__pb2.ServiceInfoMap.SerializeToString, types__pb2.CommonResponse.FromString,
                                             options, channel_credentials, insecure, call_credentials, compression,
                                             wait_for_ready, timeout, metadata)

    @staticmethod
    def ResetService(request,
                     target,
                     options=(),
                     channel_credentials=None,
                     call_credentials=None,
                     insecure=False,
                     compression=None,
                     wait_for_ready=None,
                     timeout=None,
                     metadata=None):
        return grpc.experimental.unary_unary(request, target, '/game.bff.BFF/ResetService',
                                             bff__pb2.ServiceIdList.SerializeToString, types__pb2.CommonResponse.FromString,
                                             options, channel_credentials, insecure, call_credentials, compression,
                                             wait_for_ready, timeout, metadata)

    @staticmethod
    def QueryService(request,
                     target,
                     options=(),
                     channel_credentials=None,
                     call_credentials=None,
                     insecure=False,
                     compression=None,
                     wait_for_ready=None,
                     timeout=None,
                     metadata=None):
        return grpc.experimental.unary_unary(request, target, '/game.bff.BFF/QueryService',
                                             bff__pb2.ServiceIdList.SerializeToString, bff__pb2.ServiceStateMap.FromString,
                                             options, channel_credentials, insecure, call_credentials, compression,
                                             wait_for_ready, timeout, metadata)

    @staticmethod
    def GetSimenvConfig(request,
                        target,
                        options=(),
                        channel_credentials=None,
                        call_credentials=None,
                        insecure=False,
                        compression=None,
                        wait_for_ready=None,
                        timeout=None,
                        metadata=None):
        return grpc.experimental.unary_unary(request, target, '/game.bff.BFF/GetSimenvConfig',
                                             bff__pb2.ServiceIdList.SerializeToString, bff__pb2.SimenvConfigMap.FromString,
                                             options, channel_credentials, insecure, call_credentials, compression,
                                             wait_for_ready, timeout, metadata)

    @staticmethod
    def SetSimenvConfig(request,
                        target,
                        options=(),
                        channel_credentials=None,
                        call_credentials=None,
                        insecure=False,
                        compression=None,
                        wait_for_ready=None,
                        timeout=None,
                        metadata=None):
        return grpc.experimental.unary_unary(request, target, '/game.bff.BFF/SetSimenvConfig',
                                             bff__pb2.SimenvConfigMap.SerializeToString, types__pb2.CommonResponse.FromString,
                                             options, channel_credentials, insecure, call_credentials, compression,
                                             wait_for_ready, timeout, metadata)

    @staticmethod
    def SimControl(request,
                   target,
                   options=(),
                   channel_credentials=None,
                   call_credentials=None,
                   insecure=False,
                   compression=None,
                   wait_for_ready=None,
                   timeout=None,
                   metadata=None):
        return grpc.experimental.unary_unary(request, target, '/game.bff.BFF/SimControl', bff__pb2.SimCmdMap.SerializeToString,
                                             types__pb2.CommonResponse.FromString, options, channel_credentials, insecure,
                                             call_credentials, compression, wait_for_ready, timeout, metadata)

    @staticmethod
    def SimMonitor(request,
                   target,
                   options=(),
                   channel_credentials=None,
                   call_credentials=None,
                   insecure=False,
                   compression=None,
                   wait_for_ready=None,
                   timeout=None,
                   metadata=None):
        return grpc.experimental.unary_unary(request, target, '/game.bff.BFF/SimMonitor',
                                             bff__pb2.ServiceIdList.SerializeToString, bff__pb2.SimInfoMap.FromString, options,
                                             channel_credentials, insecure, call_credentials, compression, wait_for_ready,
                                             timeout, metadata)

    @staticmethod
    def GetAgentConfig(request,
                       target,
                       options=(),
                       channel_credentials=None,
                       call_credentials=None,
                       insecure=False,
                       compression=None,
                       wait_for_ready=None,
                       timeout=None,
                       metadata=None):
        return grpc.experimental.unary_unary(request, target, '/game.bff.BFF/GetAgentConfig',
                                             bff__pb2.ServiceIdList.SerializeToString, bff__pb2.AgentConfigMap.FromString,
                                             options, channel_credentials, insecure, call_credentials, compression,
                                             wait_for_ready, timeout, metadata)

    @staticmethod
    def SetAgentConfig(request,
                       target,
                       options=(),
                       channel_credentials=None,
                       call_credentials=None,
                       insecure=False,
                       compression=None,
                       wait_for_ready=None,
                       timeout=None,
                       metadata=None):
        return grpc.experimental.unary_unary(request, target, '/game.bff.BFF/SetAgentConfig',
                                             bff__pb2.AgentConfigMap.SerializeToString, types__pb2.CommonResponse.FromString,
                                             options, channel_credentials, insecure, call_credentials, compression,
                                             wait_for_ready, timeout, metadata)

    @staticmethod
    def GetAgentMode(request,
                     target,
                     options=(),
                     channel_credentials=None,
                     call_credentials=None,
                     insecure=False,
                     compression=None,
                     wait_for_ready=None,
                     timeout=None,
                     metadata=None):
        return grpc.experimental.unary_unary(request, target, '/game.bff.BFF/GetAgentMode',
                                             bff__pb2.ServiceIdList.SerializeToString, bff__pb2.AgentModeMap.FromString,
                                             options, channel_credentials, insecure, call_credentials, compression,
                                             wait_for_ready, timeout, metadata)

    @staticmethod
    def SetAgentMode(request,
                     target,
                     options=(),
                     channel_credentials=None,
                     call_credentials=None,
                     insecure=False,
                     compression=None,
                     wait_for_ready=None,
                     timeout=None,
                     metadata=None):
        return grpc.experimental.unary_unary(request, target, '/game.bff.BFF/SetAgentMode',
                                             bff__pb2.AgentModeMap.SerializeToString, types__pb2.CommonResponse.FromString,
                                             options, channel_credentials, insecure, call_credentials, compression,
                                             wait_for_ready, timeout, metadata)

    @staticmethod
    def GetModelWeights(request,
                        target,
                        options=(),
                        channel_credentials=None,
                        call_credentials=None,
                        insecure=False,
                        compression=None,
                        wait_for_ready=None,
                        timeout=None,
                        metadata=None):
        return grpc.experimental.unary_unary(request, target, '/game.bff.BFF/GetModelWeights',
                                             bff__pb2.ServiceIdList.SerializeToString, bff__pb2.ModelWeightsMap.FromString,
                                             options, channel_credentials, insecure, call_credentials, compression,
                                             wait_for_ready, timeout, metadata)

    @staticmethod
    def SetModelWeights(request,
                        target,
                        options=(),
                        channel_credentials=None,
                        call_credentials=None,
                        insecure=False,
                        compression=None,
                        wait_for_ready=None,
                        timeout=None,
                        metadata=None):
        return grpc.experimental.unary_unary(request, target, '/game.bff.BFF/SetModelWeights',
                                             bff__pb2.ModelWeightsMap.SerializeToString, types__pb2.CommonResponse.FromString,
                                             options, channel_credentials, insecure, call_credentials, compression,
                                             wait_for_ready, timeout, metadata)

    @staticmethod
    def GetModelBuffer(request,
                       target,
                       options=(),
                       channel_credentials=None,
                       call_credentials=None,
                       insecure=False,
                       compression=None,
                       wait_for_ready=None,
                       timeout=None,
                       metadata=None):
        return grpc.experimental.unary_unary(request, target, '/game.bff.BFF/GetModelBuffer',
                                             bff__pb2.ServiceIdList.SerializeToString, bff__pb2.ModelBufferMap.FromString,
                                             options, channel_credentials, insecure, call_credentials, compression,
                                             wait_for_ready, timeout, metadata)

    @staticmethod
    def SetModelBuffer(request,
                       target,
                       options=(),
                       channel_credentials=None,
                       call_credentials=None,
                       insecure=False,
                       compression=None,
                       wait_for_ready=None,
                       timeout=None,
                       metadata=None):
        return grpc.experimental.unary_unary(request, target, '/game.bff.BFF/SetModelBuffer',
                                             bff__pb2.ModelBufferMap.SerializeToString, types__pb2.CommonResponse.FromString,
                                             options, channel_credentials, insecure, call_credentials, compression,
                                             wait_for_ready, timeout, metadata)

    @staticmethod
    def GetModelStatus(request,
                       target,
                       options=(),
                       channel_credentials=None,
                       call_credentials=None,
                       insecure=False,
                       compression=None,
                       wait_for_ready=None,
                       timeout=None,
                       metadata=None):
        return grpc.experimental.unary_unary(request, target, '/game.bff.BFF/GetModelStatus',
                                             bff__pb2.ServiceIdList.SerializeToString, bff__pb2.ModelStatusMap.FromString,
                                             options, channel_credentials, insecure, call_credentials, compression,
                                             wait_for_ready, timeout, metadata)

    @staticmethod
    def SetModelStatus(request,
                       target,
                       options=(),
                       channel_credentials=None,
                       call_credentials=None,
                       insecure=False,
                       compression=None,
                       wait_for_ready=None,
                       timeout=None,
                       metadata=None):
        return grpc.experimental.unary_unary(request, target, '/game.bff.BFF/SetModelStatus',
                                             bff__pb2.ModelStatusMap.SerializeToString, types__pb2.CommonResponse.FromString,
                                             options, channel_credentials, insecure, call_credentials, compression,
                                             wait_for_ready, timeout, metadata)

    @staticmethod
    def Call(request,
             target,
             options=(),
             channel_credentials=None,
             call_credentials=None,
             insecure=False,
             compression=None,
             wait_for_ready=None,
             timeout=None,
             metadata=None):
        return grpc.experimental.unary_unary(request, target, '/game.bff.BFF/Call', bff__pb2.CallDataMap.SerializeToString,
                                             bff__pb2.CallDataMap.FromString, options, channel_credentials, insecure,
                                             call_credentials, compression, wait_for_ready, timeout, metadata)
