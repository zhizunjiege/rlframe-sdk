import json
import time
import unittest

from src.rlsdk.client import BFFClient


class BFFClientTestCase(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.client = BFFClient('localhost:10000')
        cls.task_dir = 'src/tests/examples'

    @classmethod
    def tearDownClass(cls):
        cls.client.reset_server()

    def test_00_registerservice(self):
        with open(f'{self.task_dir}/task.json', 'r') as f:
            task = json.load(f)
        services = {id: srv['infos'] for id, srv in task.items()}
        self.client.register_service(services=services)
        self.client.unregister_service(ids=[])
        self.client.register_service(services=services)

    def test_01_serviceinfo(self):
        services = self.client.get_service_info(ids=[])
        self.client.set_service_info(services=services)
        self.assertIn('simenv-main', services)
        self.assertIn('agent-main', services)

    def test_02_routeconfig(self):
        routes = {
            'simenv-main': {
                'agent-main': ['model-main']
            },
        }
        self.client.set_route_config(routes=routes)
        routes = self.client.get_route_config()
        self.assertIn('simenv-main', routes)
        self.assertIn('agent-main', routes['simenv-main'])

    def test_03_resetservice(self):
        self.client.reset_service(ids=[])

    def test_04_queryservice(self):
        states = self.client.query_service(ids=[])
        self.assertFalse(states['simenv-main'])
        self.assertFalse(states['agent-main'])

    def test_05_agentconfig(self):
        configs = {'agent-main': {}}
        with open(f'{self.task_dir}/agent/hypers.json', 'r') as f1, \
             open(f'{self.task_dir}/agent/states_inputs_func.py', 'r') as f2, \
             open(f'{self.task_dir}/agent/outputs_actions_func.py', 'r') as f3, \
             open(f'{self.task_dir}/agent/reward_func.py', 'r') as f4:
            configs['agent-main']['training'] = True
            configs['agent-main']['type'] = 'DQN'
            configs['agent-main']['hypers'] = f1.read()
            configs['agent-main']['sifunc'] = f2.read()
            configs['agent-main']['oafunc'] = f3.read()
            configs['agent-main']['rewfunc'] = f4.read()
        self.client.set_agent_config(configs=configs)
        configs = self.client.get_agent_config(ids=[])
        self.assertIn('agent-main', configs)

    def test_06_agentmode(self):
        modes = self.client.get_agent_mode(ids=[])
        self.assertIn('agent-main', modes)
        self.client.set_agent_mode(modes=modes)

    def test_07_modelweights(self):
        weights = self.client.get_model_weights(ids=[])
        self.assertIn('agent-main', weights)
        self.client.set_model_weights(weights=weights)

    def test_08_modelbuffer(self):
        buffers = self.client.get_model_buffer(ids=[])
        self.assertIn('agent-main', buffers)
        self.client.set_model_buffer(buffers=buffers)

    def test_09_modelstatus(self):
        status = self.client.get_model_status(ids=[])
        self.assertIn('agent-main', status)
        self.client.set_model_status(status=status)

    def test_10_simenvconfig(self):
        configs = {'simenv-main': {}}
        with open(f'{self.task_dir}/simenv/args.json', 'r') as f1, \
             open(f'{self.task_dir}/simenv/sim_term_func.cpp', 'r') as f2:
            configs['simenv-main']['type'] = 'CQSIM'
            configs['simenv-main']['args'] = json.load(f1)
            configs['simenv-main']['args']['proxy']['sim_term_func'] = f2.read()
        self.client.set_simenv_config(configs=configs)
        configs = self.client.get_simenv_config(ids=[])
        self.assertIn('simenv-main', configs)

    def test_11_simcontrol(self):
        cmds = {'simenv-main': {}}

        cmds['simenv-main']['type'] = 'init'
        cmds['simenv-main']['params'] = '{}'
        self.client.sim_control(cmds=cmds)

        cmds['simenv-main']['type'] = 'start'
        cmds['simenv-main']['params'] = '{}'
        self.client.sim_control(cmds=cmds)

        time.sleep(30)

        cmds['simenv-main']['type'] = 'stop'
        cmds['simenv-main']['params'] = '{}'
        self.client.sim_control(cmds=cmds)

    def test_12_simmonitor(self):
        infos = self.client.sim_monitor(ids=[])
        self.assertIn('simenv-main', infos)
