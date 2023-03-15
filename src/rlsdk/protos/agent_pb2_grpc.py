"""Client and server classes corresponding to protobuf-defined services."""
import grpc
from . import agent_pb2 as agent__pb2
from . import types_pb2 as types__pb2


class AgentStub(object):
    """Missing associated documentation comment in .proto file."""

    def __init__(self, channel):
        """Constructor.

        Args:
            channel: A grpc.Channel.
        """
        self.ResetService = channel.unary_unary('/game.agent.Agent/ResetService',
                                                request_serializer=types__pb2.CommonRequest.SerializeToString,
                                                response_deserializer=types__pb2.CommonResponse.FromString)
        self.QueryService = channel.unary_unary('/game.agent.Agent/QueryService',
                                                request_serializer=types__pb2.CommonRequest.SerializeToString,
                                                response_deserializer=types__pb2.ServiceState.FromString)
        self.GetAgentConfig = channel.unary_unary('/game.agent.Agent/GetAgentConfig',
                                                  request_serializer=types__pb2.CommonRequest.SerializeToString,
                                                  response_deserializer=agent__pb2.AgentConfig.FromString)
        self.SetAgentConfig = channel.unary_unary('/game.agent.Agent/SetAgentConfig',
                                                  request_serializer=agent__pb2.AgentConfig.SerializeToString,
                                                  response_deserializer=types__pb2.CommonResponse.FromString)
        self.GetAgentMode = channel.unary_unary('/game.agent.Agent/GetAgentMode',
                                                request_serializer=types__pb2.CommonRequest.SerializeToString,
                                                response_deserializer=agent__pb2.AgentMode.FromString)
        self.SetAgentMode = channel.unary_unary('/game.agent.Agent/SetAgentMode',
                                                request_serializer=agent__pb2.AgentMode.SerializeToString,
                                                response_deserializer=types__pb2.CommonResponse.FromString)
        self.GetModelWeights = channel.unary_unary('/game.agent.Agent/GetModelWeights',
                                                   request_serializer=types__pb2.CommonRequest.SerializeToString,
                                                   response_deserializer=agent__pb2.ModelWeights.FromString)
        self.SetModelWeights = channel.unary_unary('/game.agent.Agent/SetModelWeights',
                                                   request_serializer=agent__pb2.ModelWeights.SerializeToString,
                                                   response_deserializer=types__pb2.CommonResponse.FromString)
        self.GetModelBuffer = channel.unary_unary('/game.agent.Agent/GetModelBuffer',
                                                  request_serializer=types__pb2.CommonRequest.SerializeToString,
                                                  response_deserializer=agent__pb2.ModelBuffer.FromString)
        self.SetModelBuffer = channel.unary_unary('/game.agent.Agent/SetModelBuffer',
                                                  request_serializer=agent__pb2.ModelBuffer.SerializeToString,
                                                  response_deserializer=types__pb2.CommonResponse.FromString)
        self.GetModelStatus = channel.unary_unary('/game.agent.Agent/GetModelStatus',
                                                  request_serializer=types__pb2.CommonRequest.SerializeToString,
                                                  response_deserializer=agent__pb2.ModelStatus.FromString)
        self.SetModelStatus = channel.unary_unary('/game.agent.Agent/SetModelStatus',
                                                  request_serializer=agent__pb2.ModelStatus.SerializeToString,
                                                  response_deserializer=types__pb2.CommonResponse.FromString)
        self.GetAction = channel.stream_stream('/game.agent.Agent/GetAction',
                                               request_serializer=types__pb2.SimState.SerializeToString,
                                               response_deserializer=types__pb2.SimAction.FromString)
        self.Call = channel.unary_unary('/game.agent.Agent/Call',
                                        request_serializer=types__pb2.CallData.SerializeToString,
                                        response_deserializer=types__pb2.CallData.FromString)


class AgentServicer(object):
    """Missing associated documentation comment in .proto file."""

    def ResetService(self, request, context):
        """reset agent service state
        """
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')

    def QueryService(self, request, context):
        """query agent service state
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

    def GetAction(self, request_iterator, context):
        """get action
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


def add_AgentServicer_to_server(servicer, server):
    rpc_method_handlers = {
        'ResetService':
            grpc.unary_unary_rpc_method_handler(servicer.ResetService,
                                                request_deserializer=types__pb2.CommonRequest.FromString,
                                                response_serializer=types__pb2.CommonResponse.SerializeToString),
        'QueryService':
            grpc.unary_unary_rpc_method_handler(servicer.QueryService,
                                                request_deserializer=types__pb2.CommonRequest.FromString,
                                                response_serializer=types__pb2.ServiceState.SerializeToString),
        'GetAgentConfig':
            grpc.unary_unary_rpc_method_handler(servicer.GetAgentConfig,
                                                request_deserializer=types__pb2.CommonRequest.FromString,
                                                response_serializer=agent__pb2.AgentConfig.SerializeToString),
        'SetAgentConfig':
            grpc.unary_unary_rpc_method_handler(servicer.SetAgentConfig,
                                                request_deserializer=agent__pb2.AgentConfig.FromString,
                                                response_serializer=types__pb2.CommonResponse.SerializeToString),
        'GetAgentMode':
            grpc.unary_unary_rpc_method_handler(servicer.GetAgentMode,
                                                request_deserializer=types__pb2.CommonRequest.FromString,
                                                response_serializer=agent__pb2.AgentMode.SerializeToString),
        'SetAgentMode':
            grpc.unary_unary_rpc_method_handler(servicer.SetAgentMode,
                                                request_deserializer=agent__pb2.AgentMode.FromString,
                                                response_serializer=types__pb2.CommonResponse.SerializeToString),
        'GetModelWeights':
            grpc.unary_unary_rpc_method_handler(servicer.GetModelWeights,
                                                request_deserializer=types__pb2.CommonRequest.FromString,
                                                response_serializer=agent__pb2.ModelWeights.SerializeToString),
        'SetModelWeights':
            grpc.unary_unary_rpc_method_handler(servicer.SetModelWeights,
                                                request_deserializer=agent__pb2.ModelWeights.FromString,
                                                response_serializer=types__pb2.CommonResponse.SerializeToString),
        'GetModelBuffer':
            grpc.unary_unary_rpc_method_handler(servicer.GetModelBuffer,
                                                request_deserializer=types__pb2.CommonRequest.FromString,
                                                response_serializer=agent__pb2.ModelBuffer.SerializeToString),
        'SetModelBuffer':
            grpc.unary_unary_rpc_method_handler(servicer.SetModelBuffer,
                                                request_deserializer=agent__pb2.ModelBuffer.FromString,
                                                response_serializer=types__pb2.CommonResponse.SerializeToString),
        'GetModelStatus':
            grpc.unary_unary_rpc_method_handler(servicer.GetModelStatus,
                                                request_deserializer=types__pb2.CommonRequest.FromString,
                                                response_serializer=agent__pb2.ModelStatus.SerializeToString),
        'SetModelStatus':
            grpc.unary_unary_rpc_method_handler(servicer.SetModelStatus,
                                                request_deserializer=agent__pb2.ModelStatus.FromString,
                                                response_serializer=types__pb2.CommonResponse.SerializeToString),
        'GetAction':
            grpc.stream_stream_rpc_method_handler(servicer.GetAction,
                                                  request_deserializer=types__pb2.SimState.FromString,
                                                  response_serializer=types__pb2.SimAction.SerializeToString),
        'Call':
            grpc.unary_unary_rpc_method_handler(servicer.Call,
                                                request_deserializer=types__pb2.CallData.FromString,
                                                response_serializer=types__pb2.CallData.SerializeToString)
    }
    generic_handler = grpc.method_handlers_generic_handler('game.agent.Agent', rpc_method_handlers)
    server.add_generic_rpc_handlers((generic_handler,))


class Agent(object):
    """Missing associated documentation comment in .proto file."""

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
        return grpc.experimental.unary_unary(request, target, '/game.agent.Agent/ResetService',
                                             types__pb2.CommonRequest.SerializeToString, types__pb2.CommonResponse.FromString,
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
        return grpc.experimental.unary_unary(request, target, '/game.agent.Agent/QueryService',
                                             types__pb2.CommonRequest.SerializeToString, types__pb2.ServiceState.FromString,
                                             options, channel_credentials, insecure, call_credentials, compression,
                                             wait_for_ready, timeout, metadata)

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
        return grpc.experimental.unary_unary(request, target, '/game.agent.Agent/GetAgentConfig',
                                             types__pb2.CommonRequest.SerializeToString, agent__pb2.AgentConfig.FromString,
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
        return grpc.experimental.unary_unary(request, target, '/game.agent.Agent/SetAgentConfig',
                                             agent__pb2.AgentConfig.SerializeToString, types__pb2.CommonResponse.FromString,
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
        return grpc.experimental.unary_unary(request, target, '/game.agent.Agent/GetAgentMode',
                                             types__pb2.CommonRequest.SerializeToString, agent__pb2.AgentMode.FromString,
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
        return grpc.experimental.unary_unary(request, target, '/game.agent.Agent/SetAgentMode',
                                             agent__pb2.AgentMode.SerializeToString, types__pb2.CommonResponse.FromString,
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
        return grpc.experimental.unary_unary(request, target, '/game.agent.Agent/GetModelWeights',
                                             types__pb2.CommonRequest.SerializeToString, agent__pb2.ModelWeights.FromString,
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
        return grpc.experimental.unary_unary(request, target, '/game.agent.Agent/SetModelWeights',
                                             agent__pb2.ModelWeights.SerializeToString, types__pb2.CommonResponse.FromString,
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
        return grpc.experimental.unary_unary(request, target, '/game.agent.Agent/GetModelBuffer',
                                             types__pb2.CommonRequest.SerializeToString, agent__pb2.ModelBuffer.FromString,
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
        return grpc.experimental.unary_unary(request, target, '/game.agent.Agent/SetModelBuffer',
                                             agent__pb2.ModelBuffer.SerializeToString, types__pb2.CommonResponse.FromString,
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
        return grpc.experimental.unary_unary(request, target, '/game.agent.Agent/GetModelStatus',
                                             types__pb2.CommonRequest.SerializeToString, agent__pb2.ModelStatus.FromString,
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
        return grpc.experimental.unary_unary(request, target, '/game.agent.Agent/SetModelStatus',
                                             agent__pb2.ModelStatus.SerializeToString, types__pb2.CommonResponse.FromString,
                                             options, channel_credentials, insecure, call_credentials, compression,
                                             wait_for_ready, timeout, metadata)

    @staticmethod
    def GetAction(request_iterator,
                  target,
                  options=(),
                  channel_credentials=None,
                  call_credentials=None,
                  insecure=False,
                  compression=None,
                  wait_for_ready=None,
                  timeout=None,
                  metadata=None):
        return grpc.experimental.stream_stream(request_iterator, target, '/game.agent.Agent/GetAction',
                                               types__pb2.SimState.SerializeToString, types__pb2.SimAction.FromString, options,
                                               channel_credentials, insecure, call_credentials, compression, wait_for_ready,
                                               timeout, metadata)

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
        return grpc.experimental.unary_unary(request, target, '/game.agent.Agent/Call', types__pb2.CallData.SerializeToString,
                                             types__pb2.CallData.FromString, options, channel_credentials, insecure,
                                             call_credentials, compression, wait_for_ready, timeout, metadata)
