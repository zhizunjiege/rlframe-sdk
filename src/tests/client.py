import json
import time
import unittest

from src.rlsdk.configs import Service, Agent, Simenv
from src.rlsdk.client import Client


class ClientTestCase(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.client = Client('localhost:10000')
        cls.path = 'src/tests/examples'

    @classmethod
    def tearDownClass(cls):
        cls.client.reset_service()
        cls.client.reset_server()

    def test_00_registerservice(self):
        with open(f'{self.path}/services.json', 'r') as f:
            services = json.load(f)
        services = {id: Service(**services[id]) for id in services}
        self.client.register_service(services=services)
        self.client.unregister_service(ids=[])
        self.client.register_service(services=services)

    def test_01_serviceinfo(self):
        services = self.client.get_service_info(ids=[])
        self.client.set_service_info(services=services)
        self.assertIn('agent', services)
        self.assertIn('simenv', services)

    def test_02_resetservice(self):
        self.client.reset_service(ids=[])

    def test_03_queryservice(self):
        states = self.client.query_service(ids=[])
        self.assertFalse(states['agent'])
        self.assertFalse(states['simenv'])

    def test_04_agentconfig(self):
        agents = {'agent': Agent.from_files(f'{self.path}/agent')}
        self.client.set_agent_config(agents=agents)
        agents = self.client.get_agent_config(ids=[])
        self.assertIn('agent', agents)

    def test_05_agentmode(self):
        modes = self.client.get_agent_mode(ids=[])
        self.assertTrue(modes['agent'])
        self.client.set_agent_mode(modes=modes)

    def test_06_modelweights(self):
        weights = self.client.get_model_weights(ids=[])
        self.assertIn('agent', weights)
        self.client.set_model_weights(weights=weights)

    def test_07_modelbuffer(self):
        buffers = self.client.get_model_buffer(ids=[])
        self.assertIn('agent', buffers)
        self.client.set_model_buffer(buffers=buffers)

    def test_08_modelstatus(self):
        status = self.client.get_model_status(ids=[])
        self.assertIn('agent', status)
        self.client.set_model_status(status=status)

    def test_09_simenvconfig(self):
        simenvs = {'simenv': Simenv.from_files(f'{self.path}/simenv')}
        self.client.set_simenv_config(simenvs=simenvs)
        simenvs = self.client.get_simenv_config(ids=[])
        self.assertIn('simenv', simenvs)

    def test_10_simcontrol(self):
        self.client.sim_control(cmds={'simenv': 'init'})

        self.client.sim_control(cmds={'simenv': 'start'})

        time.sleep(30)

        self.client.sim_control(cmds={'simenv': 'stop'})

    def test_11_simmonitor(self):
        infos = self.client.sim_monitor(ids=[])
        self.assertIn('simenv', infos)

    def test_12_call(self):
        data = self.client.call(data={'simenv': ('test', '', b'')})
        self.assertIn('simenv', data)
        self.assertEqual(data['simenv'][0], 'test')

    def test_13_uploadmodel(self):
        self.client.upload_custom_model('src/tests/examples/agent/custom.py')

        name = 'Custom'
        hypers = {'obs_dim': 4, 'act_num': 2}
        with open('src/tests/examples/agent/states_inputs_func.py', 'r') as f1, \
             open('src/tests/examples/agent/outputs_actions_func.py', 'r') as f2, \
             open('src/tests/examples/agent/reward_func.py', 'r') as f3:
            agents = {
                'agent': Agent(
                    name=name,
                    hypers=hypers,
                    training=True,
                    sifunc=f1.read(),
                    oafunc=f2.read(),
                    rewfunc=f3.read(),
                )
            }
        self.client.set_agent_config(agents=agents)

    def test_14_uploadengine(self):
        self.client.upload_custom_engine('src/tests/examples/simenv/custom')

        name = 'Custom'
        args = {'scenario_id': 0, 'exp_design_id': 0}
        simenvs = {
            'simenv': Simenv(
                name=name,
                args=args,
            )
        }
        self.client.set_simenv_config(simenvs=simenvs)
