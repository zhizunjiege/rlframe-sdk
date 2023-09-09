import json
from typing import Any, Dict

from .configs import AnyDict, Service, Agent, Simenv
from .client import Client


class Task:

    @classmethod
    def from_files(cls, path: str):
        with open(f'{path}/services.json', 'r') as f:
            configs = json.load(f)
        services, agents, simenvs = {}, {}, {}
        for id in configs:
            services[id] = Service(**configs[id])
            if services[id].type == 'agent':
                agents[id] = Agent.from_files(f'{path}/{id}')
            elif services[id].type == 'simenv':
                simenvs[id] = Simenv.from_files(f'{path}/{id}')
        return cls(services, agents, simenvs)

    def __init__(
        self,
        services: Dict[str, Service] = {},
        agents: Dict[str, Agent] = {},
        simenvs: Dict[str, Simenv] = {},
    ):
        for id in services:
            if id not in agents and id not in simenvs:
                raise ValueError(f'Service {id} not configured in agents or simenvs.')

        self.services = services
        self.agents = agents
        self.simenvs = simenvs

        self.address = ''
        self.client = None

        self.inited = False

    def push(self, address: str, reset=False):
        self.address = address
        self.client = Client(address)

        if len(self.services) == 0 or len(self.agents) == 0 and len(self.simenvs) == 0:
            raise RuntimeError('Task not configured.')

        to_register = {}
        services = self.client.get_service_info()
        for id in self.services:
            if id not in services:
                to_register[id] = self.services[id]
        if len(to_register) > 0:
            self.client.register_service(to_register)

        if reset:
            self.client.reset_service(list(self.services.keys()))
        else:
            states = self.client.query_service(list(self.services.keys()))
            for id in states:
                if states[id]:
                    raise RuntimeError(f'Service {id} already inited.')

        if len(self.agents) > 0:
            self.client.set_agent_config(self.agents)
        if len(self.simenvs) > 0:
            self.client.set_simenv_config(self.simenvs)

        self.inited = True

    def pull(self, address: str, reset=False):
        self.address = address
        self.client = Client(address)

        registered = {}
        states = self.client.query_service()
        for id in states:
            if states[id]:
                registered[id] = states[id]
        if len(registered) == 0:
            raise RuntimeError('Task not inited.')
        else:
            registered = self.client.get_service_info(list(registered.keys()))

        if reset:
            self.services = {}
            self.agents = {}
            self.simenvs = {}
        else:
            if len(self.services) > 0 or len(self.agents) > 0 or len(self.simenvs) > 0:
                raise RuntimeError('Task already configured.')

        self.services = registered
        agent_ids = [id for id in registered if registered[id].type == 'agent']
        if len(agent_ids) > 0:
            self.agents = self.client.get_agent_config(agent_ids)
        simenv_ids = [id for id in registered if registered[id].type == 'simenv']
        if len(simenv_ids) > 0:
            self.simenvs = self.client.get_simenv_config(simenv_ids)

        self.inited = True

    def details(self) -> Dict[str, AnyDict]:
        self.__check_inited()
        details = {}
        states = self.client.query_service(list(self.services.keys()))
        for id in states:
            details[id] = {'type': '', 'inited': states[id]}
        if len(self.agents) > 0:
            agents = self.client.get_model_status(list(self.agents.keys()))
            for id in agents:
                details[id]['type'] = 'agent'
                details[id]['status'] = agents[id]
        if len(self.simenvs) > 0:
            simenvs = self.client.sim_monitor(list(self.simenvs.keys()))
            for id in simenvs:
                details[id]['type'] = 'simenv'
                details[id]['infos'] = simenvs[id]
        return details

    def switch_training(self) -> bool:
        self.__check_inited()
        modes = self.client.get_agent_mode(list(self.agents.keys()))
        modes = {id: not modes[id] for id in modes if self.agents[id].training}
        self.client.set_agent_mode(modes)
        return all(modes.values())

    def set_weights(self, id: str, weights: Any):
        self.__check_inited()
        self.client.set_model_weights({id: weights})

    def get_weights(self, id: str) -> Any:
        self.__check_inited()
        return self.client.get_model_weights([id])[id]

    def set_buffer(self, id: str, buffer: Any):
        self.__check_inited()
        self.client.set_model_buffer({id: buffer})

    def get_buffer(self, id: str) -> Any:
        self.__check_inited()
        return self.client.get_model_buffer([id])[id]

    def set_status(self, id: str, status: AnyDict):
        self.__check_inited()
        self.client.set_model_status({id: status})

    def get_status(self, id: str) -> AnyDict:
        self.__check_inited()
        return self.client.get_model_status([id])[id]

    def init(self):
        self.__check_inited()
        self.client.sim_control(self.__gen_cmds('init'))

    def start(self):
        self.__check_inited()
        self.client.sim_control(self.__gen_cmds('start'))

    def pause(self):
        self.__check_inited()
        self.client.sim_control(self.__gen_cmds('pause'))

    def resume(self):
        self.__check_inited()
        self.client.sim_control(self.__gen_cmds('resume'))

    def stop(self):
        self.__check_inited()
        self.client.sim_control(self.__gen_cmds('stop'))

    def monitor(self) -> Dict[str, AnyDict]:
        self.__check_inited()
        return self.client.sim_monitor()

    def __gen_cmds(self, cmd):
        return {id: cmd for id in self.simenvs}

    def __check_inited(self):
        if not self.inited:
            raise RuntimeError('Task not inited, call push() or pull() first.')
