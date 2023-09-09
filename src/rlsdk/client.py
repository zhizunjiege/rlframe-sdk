import json
import pathlib
import pickle
import shutil
import tempfile
from typing import Dict, List, Tuple

import grpc
import numpy as np  # noqa: F401

from .configs import AnyDict, Service, Agent, Simenv

from .protos import bff_pb2, bff_pb2_grpc
from .protos import types_pb2


class Client:

    def __init__(self, address: str, max_msg_len=256):
        self.address = address
        try:
            self.channel = grpc.insecure_channel(
                address,
                options=[
                    ('grpc.max_send_message_length', max_msg_len * 1024 * 1024),
                    ('grpc.max_receive_message_length', max_msg_len * 1024 * 1024),
                ],
            )
            grpc.channel_ready_future(self.channel).result(timeout=3)
        except (grpc.FutureTimeoutError, grpc.RpcError):
            self.channel.close()
            raise ConnectionError(f'Connection to {address} timed out, please check the address again.')
        else:
            self.stub = bff_pb2_grpc.BFFStub(self.channel)

    def __del__(self):
        self.channel.close()

    def reset_server(self):
        self.stub.ResetServer(types_pb2.CommonRequest())

    def register_service(self, services: Dict[str, Service]):
        service_info_map = bff_pb2.ServiceInfoMap()
        for id, service in services.items():
            service_info_map.services[id].type = service.type
            service_info_map.services[id].name = service.name
            service_info_map.services[id].host = service.host
            service_info_map.services[id].port = service.port
            service_info_map.services[id].desc = service.desc
        self.stub.RegisterService(service_info_map)

    def unregister_service(self, ids: List[str] = []):
        self.stub.UnRegisterService(bff_pb2.ServiceIdList(ids=ids))

    def get_service_info(self, ids: List[str] = []) -> Dict[str, Service]:
        service_info_map = self.stub.GetServiceInfo(bff_pb2.ServiceIdList(ids=ids))
        services = {}
        for id, service in service_info_map.services.items():
            services[id] = Service(
                type=service.type,
                name=service.name,
                host=service.host,
                port=service.port,
                desc=service.desc,
            )
        return services

    def set_service_info(self, services: Dict[str, Service]):
        service_info_map = bff_pb2.ServiceInfoMap()
        for id, service in services.items():
            service_info_map.services[id].type = service.type
            service_info_map.services[id].name = service.name
            service_info_map.services[id].host = service.host
            service_info_map.services[id].port = service.port
            service_info_map.services[id].desc = service.desc
        self.stub.SetServiceInfo(service_info_map)

    def reset_service(self, ids: List[str] = []):
        self.stub.ResetService(bff_pb2.ServiceIdList(ids=ids))

    def query_service(self, ids: List[str] = []) -> Dict[str, bool]:
        service_state_map = self.stub.QueryService(bff_pb2.ServiceIdList(ids=ids))
        return {id: msg.state == types_pb2.ServiceState.State.INITED for id, msg in service_state_map.states.items()}

    def get_agent_config(self, ids: List[str] = []) -> Dict[str, Agent]:
        agent_config_map = self.stub.GetAgentConfig(bff_pb2.ServiceIdList(ids=ids))
        agents = {}
        for id, agent in agent_config_map.configs.items():
            agents[id] = Agent(
                name=agent.name,
                hypers=json.loads(agent.hypers),
                training=agent.training,
                sifunc=agent.sifunc,
                oafunc=agent.oafunc,
                rewfunc=agent.rewfunc,
                hooks=[{
                    'name': hook.name,
                    'args': json.loads(hook.args)
                } for hook in agent.hooks],
            )
        return agents

    def set_agent_config(self, agents: Dict[str, Agent]):
        agent_config_map = bff_pb2.AgentConfigMap()
        for id, agent in agents.items():
            agent_config_map.configs[id].name = agent.name
            agent_config_map.configs[id].hypers = json.dumps(agent.hypers)
            agent_config_map.configs[id].training = agent.training
            agent_config_map.configs[id].sifunc = agent.sifunc
            agent_config_map.configs[id].oafunc = agent.oafunc
            agent_config_map.configs[id].rewfunc = agent.rewfunc
            for hook in agent.hooks:
                pointer = agent_config_map.configs[id].hooks.add()
                pointer.name = hook['name']
                pointer.args = json.dumps(hook['args'])
        self.stub.SetAgentConfig(agent_config_map)

    def get_agent_mode(self, ids: List[str] = []) -> Dict[str, bool]:
        agent_mode_map = self.stub.GetAgentMode(bff_pb2.ServiceIdList(ids=ids))
        return {id: bool(msg.training) for id, msg in agent_mode_map.modes.items()}

    def set_agent_mode(self, modes: Dict[str, bool]):
        agent_mode_map = bff_pb2.AgentModeMap()
        for id in modes:
            agent_mode_map.modes[id].training = modes[id]
        self.stub.SetAgentMode(agent_mode_map)

    def get_model_weights(self, ids: List[str] = []) -> AnyDict:
        model_weights_map = self.stub.GetModelWeights(bff_pb2.ServiceIdList(ids=ids))
        return {id: pickle.loads(msg.weights) for id, msg in model_weights_map.weights.items()}

    def set_model_weights(self, weights: AnyDict):
        model_weights_map = bff_pb2.ModelWeightsMap()
        for id in weights:
            model_weights_map.weights[id].weights = pickle.dumps(weights[id])
        self.stub.SetModelWeights(model_weights_map)

    def get_model_buffer(self, ids: List[str] = []) -> AnyDict:
        model_buffer_map = self.stub.GetModelBuffer(bff_pb2.ServiceIdList(ids=ids))
        return {id: pickle.loads(msg.buffer) for id, msg in model_buffer_map.buffers.items()}

    def set_model_buffer(self, buffers: AnyDict):
        model_buffer_map = bff_pb2.ModelBufferMap()
        for id in buffers:
            model_buffer_map.buffers[id].buffer = pickle.dumps(buffers[id])
        self.stub.SetModelBuffer(model_buffer_map)

    def get_model_status(self, ids: List[str] = []) -> Dict[str, AnyDict]:
        model_status_map = self.stub.GetModelStatus(bff_pb2.ServiceIdList(ids=ids))
        return {id: json.loads(msg.status) for id, msg in model_status_map.status.items()}

    def set_model_status(self, status: Dict[str, AnyDict]):
        model_status_map = bff_pb2.ModelStatusMap()
        for id in status:
            model_status_map.status[id].status = json.dumps(status[id])
        self.stub.SetModelStatus(model_status_map)

    def get_simenv_config(self, ids: List[str] = []) -> Dict[str, Simenv]:
        simenv_config_map = self.stub.GetSimenvConfig(bff_pb2.ServiceIdList(ids=ids))
        simenvs = {}
        for id, simenv in simenv_config_map.configs.items():
            simenvs[id] = Simenv(
                name=simenv.name,
                args=json.loads(simenv.args),
            )
        return simenvs

    def set_simenv_config(self, simenvs: Dict[str, Simenv]):
        simenv_config_map = bff_pb2.SimenvConfigMap()
        for id, simenv in simenvs.items():
            simenv_config_map.configs[id].name = simenv.name
            simenv_config_map.configs[id].args = json.dumps(simenv.args)
        self.stub.SetSimenvConfig(simenv_config_map)

    def sim_control(self, cmds: Dict[str, str]):
        sim_cmd_map = bff_pb2.SimCmdMap()
        for id in cmds:
            sim_cmd_map.cmds[id].type = cmds[id]
        self.stub.SimControl(sim_cmd_map)

    def sim_monitor(self, ids: List[str] = []) -> Dict[str, AnyDict]:
        sim_info_map = self.stub.SimMonitor(bff_pb2.ServiceIdList(ids=ids))
        return {
            id: {
                'state': msg.state,
                'data': json.loads(msg.data),
                'logs': json.loads(msg.logs),
            } for id, msg in sim_info_map.infos.items()
        }

    def call(self, data: Dict[str, Tuple[str, str, bytes]]) -> Dict[str, Tuple[str, str, bytes]]:
        req = bff_pb2.CallDataMap()
        for id in data:
            req.data[id].name = data[id][0]
            req.data[id].dstr = data[id][1]
            req.data[id].dbin = data[id][2]
        res = self.stub.Call(req)
        return {id: (msg.name, msg.dstr, msg.dbin) for id, msg in res.data.items()}

    def upload_custom(self, ids: List[str], path: str):
        tmp = tempfile.gettempdir()
        tgt = pathlib.Path(path)
        arch = shutil.make_archive(f'{tmp}/temp', 'zip', root_dir=tgt.parent, base_dir=tgt.name)
        with open(arch, 'rb') as f:
            file = f.read()

        data = ('@custom', '', file)
        self.call({id: data for id in ids})

    def upload_custom_model(self, path: str):
        services = self.get_service_info()
        ids = [id for id, service in services.items() if service.type == 'agent']
        self.upload_custom(ids, path)

    def upload_custom_engine(self, path: str):
        services = self.get_service_info()
        ids = [id for id, service in services.items() if service.type == 'simenv']
        self.upload_custom(ids, path)
