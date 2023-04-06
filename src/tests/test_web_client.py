import json
import unittest

from src.rlsdk.client import WebClient


class WebClientTestCase(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.client = WebClient('localhost:5000')
        cls.task_dir = 'src/tests/examples'

    @classmethod
    def tearDownClass(cls):
        cls.client = None

    def test_00_meta(self):
        self.assertIsNotNone(self.client.tables)

    def test_01_insert(self):
        with open(f'{self.task_dir}/simenv/args.json', 'r') as f1, \
             open(f'{self.task_dir}/simenv/sim_term_func.cpp', 'r') as f2:
            args = json.load(f1)
            args['proxy']['sim_term_func'] = f2.read()
            simenv_data = {
                'name': 'simenv-test',
                'description': 'simenv-test',
                'type': 'CQSim',
                'args': json.dumps(args),
            }
        rst = self.client.insert('simenv', data=simenv_data)
        simenv_rowid = rst['lastrowid']
        self.assertEqual(simenv_rowid, 1)

        with open(f'{self.task_dir}/agent/hypers.json', 'r') as f1, \
             open(f'{self.task_dir}/agent/states_inputs_func.py', 'r') as f2, \
             open(f'{self.task_dir}/agent/outputs_actions_func.py', 'r') as f3, \
             open(f'{self.task_dir}/agent/reward_func.py', 'r') as f4:
            hypers = json.load(f1)
            states_inputs_func = f2.read()
            outputs_actions_func = f3.read()
            reward_func = f4.read()
            agent_data = {
                'name': 'agent-test',
                'description': 'agent-test',
                'training': 1,
                'type': 'DQN',
                'hypers': json.dumps(hypers),
                'sifunc': states_inputs_func,
                'oafunc': outputs_actions_func,
                'rewfunc': reward_func,
                'weights': b'Helloworld!',
            }
        rst = self.client.insert('agent', data=agent_data)
        agent_rowid = rst['lastrowid']
        self.assertEqual(agent_rowid, 1)

        with open(f'{self.task_dir}/task.json', 'r') as f:
            services = json.load(f)
            services['simenv-main']['configs'] = simenv_rowid
            services['agent-main']['configs'] = agent_rowid
            task_data = {
                'name': 'task-test',
                'description': 'task-test',
                'services': json.dumps(services),
            }
        rst = self.client.insert('task', data=task_data)
        task_rowid = rst['lastrowid']
        self.assertEqual(task_rowid, 1)

    def test_02_update(self):
        rst = self.client.update('simenv', data={
            'id': 1,
            'params': '{}',
        })
        simenv_rowcount = rst['rowcount']
        self.assertEqual(simenv_rowcount, 1)

        rst = self.client.update('agent', data={
            'id': 1,
            'status': '{}',
        })
        agent_rowcount = rst['rowcount']
        self.assertEqual(agent_rowcount, 1)

        rst = self.client.update('task', data={
            'id': 1,
            'routes': '{}',
        })
        task_rowcount = rst['rowcount']
        self.assertEqual(task_rowcount, 1)

    def test_03_select(self):
        data = self.client.select('simenv', columns=[], id=1)
        self.assertEqual(len(data), 1)

        data = self.client.select('agent', columns=[], id=1)
        self.assertEqual(len(data), 1)

        data = self.client.select('task', columns=[], id=1)
        self.assertEqual(len(data), 1)

    def test_04_delete(self):
        rst = self.client.delete('simenv', ids=[1])
        simenv_rowcount = rst['rowcount']
        self.assertEqual(simenv_rowcount, 1)

        rst = self.client.delete('agent', ids=[1])
        agent_rowcount = rst['rowcount']
        self.assertEqual(agent_rowcount, 1)

        rst = self.client.delete('task', ids=[1])
        task_rowcount = rst['rowcount']
        self.assertEqual(task_rowcount, 1)
