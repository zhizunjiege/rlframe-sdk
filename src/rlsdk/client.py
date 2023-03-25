import base64
import json
import pickle
from typing import Any, Dict, List, Tuple, Union

import grpc
from google.protobuf import json_format as jf
import numpy as np  # noqa: F401
import requests

from .protos import bff_pb2, bff_pb2_grpc
from .protos import types_pb2


class BFFClient:

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

    def register_service(self, services: Dict[str, Dict[str, Union[str, int]]]):
        service_info_map = jf.ParseDict({'services': services}, bff_pb2.ServiceInfoMap())
        self.stub.RegisterService(service_info_map)

    def unregister_service(self, ids: List[str] = []):
        service_id_list = jf.ParseDict({'ids': ids}, bff_pb2.ServiceIdList())
        self.stub.UnRegisterService(service_id_list)

    def get_service_info(self, ids: List[str] = []) -> Dict[str, Dict[str, Union[str, int]]]:
        service_info_map = self.stub.GetServiceInfo(bff_pb2.ServiceIdList(ids=ids))
        return jf.MessageToDict(service_info_map, including_default_value_fields=True)['services']

    def set_service_info(self, services: Dict[str, Dict[str, Union[str, int]]]):
        service_info_map = jf.ParseDict({'services': services}, bff_pb2.ServiceInfoMap())
        self.stub.SetServiceInfo(service_info_map)

    def get_route_config(self) -> Dict[str, Dict[str, List[str]]]:
        route_config = self.stub.GetRouteConfig(types_pb2.CommonRequest())
        routes = jf.MessageToDict(route_config, including_default_value_fields=True)['routes']
        for simenv in routes:
            configs = routes[simenv]['configs']
            for agent in configs:
                configs[agent] = configs[agent]['models']
            routes[simenv] = configs
        return routes

    def set_route_config(self, routes: Dict[str, Dict[str, List[str]]]):
        for simenv in routes:
            configs = routes[simenv]
            for agent in configs:
                configs[agent] = {'models': configs[agent]}
            routes[simenv] = {'configs': routes[simenv]}
        route_config = jf.ParseDict({'routes': routes}, bff_pb2.RouteConfig())
        self.stub.SetRouteConfig(route_config)

    def reset_service(self, ids: List[str] = []):
        service_id_list = jf.ParseDict({'ids': ids}, bff_pb2.ServiceIdList())
        self.stub.ResetService(service_id_list)

    def query_service(self, ids: List[str] = []) -> Dict[str, bool]:
        service_id_list = jf.ParseDict({'ids': ids}, bff_pb2.ServiceIdList())
        service_state_map = self.stub.QueryService(service_id_list)
        states = jf.MessageToDict(service_state_map, including_default_value_fields=True)['states']
        for id in states:
            states[id] = bool(states[id]['state'] == 'INITED')
        return states

    def get_simenv_config(self, ids: List[str] = []) -> Dict[str, Dict[str, Any]]:
        service_id_list = jf.ParseDict({'ids': ids}, bff_pb2.ServiceIdList())
        simenv_config_map = self.stub.GetSimenvConfig(service_id_list)
        configs = jf.MessageToDict(simenv_config_map, including_default_value_fields=True)['configs']
        for id in configs:
            configs[id]['args'] = json.loads(configs[id]['args'])
        return configs

    def set_simenv_config(self, configs: Dict[str, Dict[str, Any]]):
        for id in configs:
            if not isinstance(configs[id]['args'], str):
                configs[id]['args'] = json.dumps(configs[id]['args'])
        simenv_config_map = jf.ParseDict({'configs': configs}, bff_pb2.SimenvConfigMap())
        self.stub.SetSimenvConfig(simenv_config_map)

    def sim_control(self, cmds: Dict[str, Dict[str, Any]]):
        for id in cmds:
            if not isinstance(cmds[id]['params'], str):
                cmds[id]['params'] = json.dumps(cmds[id]['params'])
        sim_cmd_map = jf.ParseDict({'cmds': cmds}, bff_pb2.SimCmdMap())
        self.stub.SimControl(sim_cmd_map)

    def sim_monitor(self, ids: List[str] = []) -> Dict[str, Dict[str, Any]]:
        service_id_list = jf.ParseDict({'ids': ids}, bff_pb2.ServiceIdList())
        sim_info_map = self.stub.SimMonitor(service_id_list)
        infos = jf.MessageToDict(sim_info_map, including_default_value_fields=True)['infos']
        for id in infos:
            infos[id]['data'] = json.loads(infos[id]['data'])
            infos[id]['logs'] = json.loads(infos[id]['logs'])
        return infos

    def get_agent_config(self, ids: List[str] = []) -> Dict[str, Dict[str, Any]]:
        service_id_list = jf.ParseDict({'ids': ids}, bff_pb2.ServiceIdList())
        agent_config_map = self.stub.GetAgentConfig(service_id_list)
        configs = jf.MessageToDict(agent_config_map, including_default_value_fields=True)['configs']
        for id in configs:
            configs[id]['hypers'] = json.loads(configs[id]['hypers'])
        return configs

    def set_agent_config(self, configs: Dict[str, Dict[str, Any]]):
        for id in configs:
            if not isinstance(configs[id]['hypers'], str):
                configs[id]['hypers'] = json.dumps(configs[id]['hypers'])
        agent_config_map = jf.ParseDict({'configs': configs}, bff_pb2.AgentConfigMap())
        self.stub.SetAgentConfig(agent_config_map)

    def get_agent_mode(self, ids: List[str] = []) -> Dict[str, bool]:
        service_id_list = jf.ParseDict({'ids': ids}, bff_pb2.ServiceIdList())
        agent_mode_map = self.stub.GetAgentMode(service_id_list)
        modes = jf.MessageToDict(agent_mode_map, including_default_value_fields=True)['modes']
        for id in modes:
            modes[id] = bool(modes[id]['training'])
        return modes

    def set_agent_mode(self, modes: Dict[str, bool]):
        for id in modes:
            modes[id] = {'training': modes[id]}
        agent_mode_map = jf.ParseDict({'modes': modes}, bff_pb2.AgentModeMap())
        self.stub.SetAgentMode(agent_mode_map)

    def get_model_weights(self, ids: List[str] = []) -> Dict[str, Any]:
        service_id_list = jf.ParseDict({'ids': ids}, bff_pb2.ServiceIdList())
        model_weights_map = self.stub.GetModelWeights(service_id_list)
        weights = {id: pickle.loads(msg.weights) for id, msg in model_weights_map.weights.items()}
        return weights

    def set_model_weights(self, weights: Dict[str, Any]):
        model_weights_map = bff_pb2.ModelWeightsMap()
        for id in weights:
            model_weights_map.weights[id].weights = pickle.dumps(weights[id])
        self.stub.SetModelWeights(model_weights_map)

    def get_model_buffer(self, ids: List[str] = []) -> Dict[str, Any]:
        service_id_list = jf.ParseDict({'ids': ids}, bff_pb2.ServiceIdList())
        model_buffer_map = self.stub.GetModelBuffer(service_id_list)
        buffers = {id: pickle.loads(msg.buffer) for id, msg in model_buffer_map.buffers.items()}
        return buffers

    def set_model_buffer(self, buffers: Dict[str, Any]):
        model_buffer_map = bff_pb2.ModelBufferMap()
        for id in buffers:
            model_buffer_map.buffers[id].buffer = pickle.dumps(buffers[id])
        self.stub.SetModelBuffer(model_buffer_map)

    def get_model_status(self, ids: List[str] = []) -> Dict[str, Dict[str, Any]]:
        service_id_list = jf.ParseDict({'ids': ids}, bff_pb2.ServiceIdList())
        model_status_map = self.stub.GetModelStatus(service_id_list)
        status = jf.MessageToDict(model_status_map, including_default_value_fields=True)['status']
        for id in status:
            status[id] = json.loads(status[id]['status'])
        return status

    def set_model_status(self, status: Dict[str, Dict[str, Any]]):
        for id in status:
            if not isinstance(status[id], str):
                status[id] = {'status': json.dumps(status[id])}
        model_status_map = jf.ParseDict({'status': status}, bff_pb2.ModelStatusMap())
        self.stub.SetModelStatus(model_status_map)

    def call(self, data: Dict[str, Tuple[str, str, bytes]]) -> Dict[str, Tuple[str, str, bytes]]:
        req = bff_pb2.CallDataMap()
        for id in data:
            req.data[id].identity = data[id][0]
            req.data[id].str_data = data[id][1]
            req.data[id].bin_data = data[id][2]
        res = self.stub.Call(req)
        data = jf.MessageToDict(res, including_default_value_fields=True)['data']
        for id in data:
            data[id] = (data[id]['identity'], data[id]['str_data'], data[id]['bin_data'])
        return data


class WebClient:

    def __init__(self, address: str):
        self.address = f'http://{address}/api/db'

        try:
            self.tables = requests.get(self.address, timeout=3).json()
        except requests.RequestException:
            raise ConnectionError(f'Connection to {address} timed out, please check the address again.')

    def select(
        self,
        table: str,
        columns: List[str] = [],
        **options: Dict[str, int],
    ) -> List[Dict[str, Any]]:
        res = requests.get(f'{self.address}/{table}', params={'columns': columns, **options})
        res.raise_for_status()
        data = res.json()
        for row in data:
            for col in row:
                if row[col] is not None:
                    col_type = self.tables[table][col]['type']
                    if col_type == 'BLOB':
                        row[col] = self.b64str_to_bytes(row[col])
                    elif col_type == 'JSON':
                        row[col] = json.loads(row[col])
        return data

    def insert(
        self,
        table: str,
        data: Dict[str, Any],
    ) -> int:
        for col in data:
            if data[col] is not None:
                col_type = self.tables[table][col]['type']
                if col_type == 'BLOB':
                    data[col] = self.bytes_to_b64str(data[col])
                elif col_type == 'JSON':
                    data[col] = json.dumps(data[col])
        res = requests.post(f'{self.address}/{table}', json=data)
        res.raise_for_status()
        return res.json()['lastrowid']

    def update(
        self,
        table: str,
        id: int,
        data: Dict[str, Any],
    ) -> int:
        for col in data:
            if data[col] is not None:
                col_type = self.tables[table][col]['type']
                if col_type == 'BLOB':
                    data[col] = self.bytes_to_b64str(data[col])
                elif col_type == 'JSON':
                    data[col] = json.dumps(data[col])
        res = requests.put(f'{self.address}/{table}/{id}', json=data)
        res.raise_for_status()
        return res.json()['rowcount']

    def delete(
        self,
        table: str,
        id: int,
    ) -> int:
        res = requests.delete(f'{self.address}/{table}/{id}')
        res.raise_for_status()
        return res.json()['rowcount']

    def bytes_to_b64str(self, data):
        return base64.b64encode(data).decode('utf-8')

    def b64str_to_bytes(self, data):
        return base64.b64decode(data.encode('utf-8'))
