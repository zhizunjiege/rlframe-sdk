"""Client and server classes corresponding to protobuf-defined services."""
import grpc
from . import simenv_pb2 as simenv__pb2
from . import types_pb2 as types__pb2


class SimenvStub(object):
    """Missing associated documentation comment in .proto file."""

    def __init__(self, channel):
        """Constructor.

        Args:
            channel: A grpc.Channel.
        """
        self.ResetService = channel.unary_unary('/game.simenv.Simenv/ResetService',
                                                request_serializer=types__pb2.CommonRequest.SerializeToString,
                                                response_deserializer=types__pb2.CommonResponse.FromString)
        self.QueryService = channel.unary_unary('/game.simenv.Simenv/QueryService',
                                                request_serializer=types__pb2.CommonRequest.SerializeToString,
                                                response_deserializer=types__pb2.ServiceState.FromString)
        self.GetSimenvConfig = channel.unary_unary('/game.simenv.Simenv/GetSimenvConfig',
                                                   request_serializer=types__pb2.CommonRequest.SerializeToString,
                                                   response_deserializer=simenv__pb2.SimenvConfig.FromString)
        self.SetSimenvConfig = channel.unary_unary('/game.simenv.Simenv/SetSimenvConfig',
                                                   request_serializer=simenv__pb2.SimenvConfig.SerializeToString,
                                                   response_deserializer=types__pb2.CommonResponse.FromString)
        self.SimControl = channel.unary_unary('/game.simenv.Simenv/SimControl',
                                              request_serializer=simenv__pb2.SimCmd.SerializeToString,
                                              response_deserializer=types__pb2.CommonResponse.FromString)
        self.SimMonitor = channel.unary_unary('/game.simenv.Simenv/SimMonitor',
                                              request_serializer=types__pb2.CommonRequest.SerializeToString,
                                              response_deserializer=simenv__pb2.SimInfo.FromString)
        self.Call = channel.unary_unary('/game.simenv.Simenv/Call',
                                        request_serializer=types__pb2.CallData.SerializeToString,
                                        response_deserializer=types__pb2.CallData.FromString)


class SimenvServicer(object):
    """Missing associated documentation comment in .proto file."""

    def ResetService(self, request, context):
        """reset simenv service state
        """
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')

    def QueryService(self, request, context):
        """query simenv service state
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

    def Call(self, request, context):
        """any rpc call
        """
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')


def add_SimenvServicer_to_server(servicer, server):
    rpc_method_handlers = {
        'ResetService':
            grpc.unary_unary_rpc_method_handler(servicer.ResetService,
                                                request_deserializer=types__pb2.CommonRequest.FromString,
                                                response_serializer=types__pb2.CommonResponse.SerializeToString),
        'QueryService':
            grpc.unary_unary_rpc_method_handler(servicer.QueryService,
                                                request_deserializer=types__pb2.CommonRequest.FromString,
                                                response_serializer=types__pb2.ServiceState.SerializeToString),
        'GetSimenvConfig':
            grpc.unary_unary_rpc_method_handler(servicer.GetSimenvConfig,
                                                request_deserializer=types__pb2.CommonRequest.FromString,
                                                response_serializer=simenv__pb2.SimenvConfig.SerializeToString),
        'SetSimenvConfig':
            grpc.unary_unary_rpc_method_handler(servicer.SetSimenvConfig,
                                                request_deserializer=simenv__pb2.SimenvConfig.FromString,
                                                response_serializer=types__pb2.CommonResponse.SerializeToString),
        'SimControl':
            grpc.unary_unary_rpc_method_handler(servicer.SimControl,
                                                request_deserializer=simenv__pb2.SimCmd.FromString,
                                                response_serializer=types__pb2.CommonResponse.SerializeToString),
        'SimMonitor':
            grpc.unary_unary_rpc_method_handler(servicer.SimMonitor,
                                                request_deserializer=types__pb2.CommonRequest.FromString,
                                                response_serializer=simenv__pb2.SimInfo.SerializeToString),
        'Call':
            grpc.unary_unary_rpc_method_handler(servicer.Call,
                                                request_deserializer=types__pb2.CallData.FromString,
                                                response_serializer=types__pb2.CallData.SerializeToString)
    }
    generic_handler = grpc.method_handlers_generic_handler('game.simenv.Simenv', rpc_method_handlers)
    server.add_generic_rpc_handlers((generic_handler,))


class Simenv(object):
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
        return grpc.experimental.unary_unary(request, target, '/game.simenv.Simenv/ResetService',
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
        return grpc.experimental.unary_unary(request, target, '/game.simenv.Simenv/QueryService',
                                             types__pb2.CommonRequest.SerializeToString, types__pb2.ServiceState.FromString,
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
        return grpc.experimental.unary_unary(request, target, '/game.simenv.Simenv/GetSimenvConfig',
                                             types__pb2.CommonRequest.SerializeToString, simenv__pb2.SimenvConfig.FromString,
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
        return grpc.experimental.unary_unary(request, target, '/game.simenv.Simenv/SetSimenvConfig',
                                             simenv__pb2.SimenvConfig.SerializeToString, types__pb2.CommonResponse.FromString,
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
        return grpc.experimental.unary_unary(request, target, '/game.simenv.Simenv/SimControl',
                                             simenv__pb2.SimCmd.SerializeToString, types__pb2.CommonResponse.FromString,
                                             options, channel_credentials, insecure, call_credentials, compression,
                                             wait_for_ready, timeout, metadata)

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
        return grpc.experimental.unary_unary(request, target, '/game.simenv.Simenv/SimMonitor',
                                             types__pb2.CommonRequest.SerializeToString, simenv__pb2.SimInfo.FromString,
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
        return grpc.experimental.unary_unary(request, target, '/game.simenv.Simenv/Call', types__pb2.CallData.SerializeToString,
                                             types__pb2.CallData.FromString, options, channel_credentials, insecure,
                                             call_credentials, compression, wait_for_ready, timeout, metadata)
