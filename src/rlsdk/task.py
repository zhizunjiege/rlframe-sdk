import json
import pathlib
from typing import Any, Dict, Optional

import jsonschema

from .client import BFFClient

schema_dir = pathlib.Path(__file__).parent / 'schemas'
with open(schema_dir / 'task.json', 'r') as f:
    task_schema = json.load(f)


class Task:

    def __init__(self, configs: Optional[Dict[str, Any]] = None):
        if configs is not None:
            jsonschema.validate(instance=configs, schema=task_schema)
            for _, service in configs.items():
                type1 = service['infos']['type']
                type2 = service['configs']['type']
                instance = service['configs']['args' if type1 == 'simenv' else 'hypers']
                schema_file = schema_dir / ('engines' if type1 == 'simenv' else 'models') / f'{type2}.json'
                if schema_file.exists():
                    with open(schema_file, 'r') as f:
                        schema = json.load(f)
                    jsonschema.validate(instance=instance, schema=schema)

        self.configs = configs
        self.inited = False

        self.simenvs, self.agents = [], []

    def push(self, bff_addr: str, reset=False):
        self.bff_addr = bff_addr
        self.bff_client = BFFClient(bff_addr)

        if self.configs is None:
            msg = 'Task not configured.'
            raise RuntimeError(msg)

        if reset:
            self.bff_client.reset_service()
            self.bff_client.reset_server()
        else:
            inited = False
            states = self.bff_client.query_service()
            for _, state in states.items():
                inited = inited or state
            if inited:
                msg = 'Task already inited.'
                raise RuntimeError(msg)

        self.simenvs, self.agents = [], []
        for id, srv in self.configs.items():
            if srv['infos']['type'] == 'simenv':
                self.simenvs.append(id)
            elif srv['infos']['type'] == 'agent':
                self.agents.append(id)
        self.bff_client.register_service({id: srv['infos'] for id, srv in self.configs.items()})
        self.bff_client.set_simenv_config(configs={id: self.configs[id]['configs'] for id in self.simenvs})
        self.bff_client.set_agent_config(configs={id: self.configs[id]['configs'] for id in self.agents})

        self.bff_client.sim_control(cmds={id: {'type': 'init', 'params': {}} for id in self.simenvs})
        self.inited = True

    def pull(self, bff_addr: str, reset=False):
        self.bff_addr = bff_addr
        self.bff_client = BFFClient(bff_addr)

        inited = True
        states = self.bff_client.query_service()
        for _, state in states.items():
            inited = inited and state
        if not inited:
            msg = 'Task not inited.'
            raise RuntimeError(msg)

        if reset:
            self.configs = None
        else:
            if self.configs is not None:
                msg = 'Task already configured.'
                raise RuntimeError(msg)

        simenv_configs = self.bff_client.get_simenv_config()
        agent_configs = self.bff_client.get_agent_config()
        infos = self.bff_client.get_service_info()
        configs = {**simenv_configs, **agent_configs}
        self.simenvs, self.agents = [], []
        for id, srv in infos.items():
            if srv['type'] == 'simenv':
                self.simenvs.append(id)
            elif srv['type'] == 'agent':
                self.agents.append(id)
        self.configs = {id: {'infos': infos[id], 'configs': configs[id]} for id in self.simenvs + self.agents}

        details = self.bff_client.sim_monitor(ids=self.simenvs)
        for _, detail in details.items():
            if detail['state'] != 'uninited':
                self.inited = True

    def details(self) -> Dict[str, Dict[str, Any]]:
        self.__check_inited()
        details = {}
        states = self.bff_client.query_service()
        for id in self.simenvs:
            details[id] = {'type': 'simenv', 'inited': states[id]}
            if states[id]:
                details[id]['infos'] = self.bff_client.sim_monitor(ids=[id])[id]
        for id in self.agents:
            details[id] = {'type': 'agent', 'inited': states[id]}
            if states[id]:
                details[id]['status'] = self.bff_client.get_model_status(ids=[id])[id]
        return details

    def start(self):
        self.__check_inited()
        self.bff_client.sim_control(cmds={id: {'type': 'start', 'params': {}} for id in self.simenvs})

    def pause(self):
        self.__check_inited()
        self.bff_client.sim_control(cmds={id: {'type': 'pause', 'params': {}} for id in self.simenvs})

    def resume(self):
        self.__check_inited()
        self.bff_client.sim_control(cmds={id: {'type': 'resume', 'params': {}} for id in self.simenvs})

    def stop(self):
        self.__check_inited()
        self.bff_client.sim_control(cmds={id: {'type': 'stop', 'params': {}} for id in self.simenvs})

    def monitor(self) -> Dict[str, Dict[str, Any]]:
        self.__check_inited()
        return self.bff_client.sim_monitor()

    def switch_training(self, mode: bool):
        self.__check_inited()
        modes = {id: mode for id in self.agents if self.configs[id]['configs']['training']}
        self.bff_client.set_agent_mode(modes=modes)

    def set_weights(self, id: str, weights: Any):
        self.__check_inited()
        self.bff_client.set_model_weights(weights={id: weights})

    def get_weights(self, id: str) -> Any:
        self.__check_inited()
        return self.bff_client.get_model_weights(ids=[id])[id]

    def set_buffer(self, id: str, buffer: Any):
        self.__check_inited()
        self.bff_client.set_model_buffer(buffer={id: buffer})

    def get_buffer(self, id: str) -> Any:
        self.__check_inited()
        return self.bff_client.get_model_buffer(ids=[id])[id]

    def set_status(self, id: str, status: Dict[str, Any]):
        self.__check_inited()
        self.bff_client.set_model_status(status={id: status})

    def get_status(self, id: str) -> Dict[str, Any]:
        self.__check_inited()
        return self.bff_client.get_model_status(ids=[id])[id]

    def __check_inited(self):
        if not self.inited:
            msg = 'Task not inited, call push() or pull() first.'
            raise RuntimeError(msg)
