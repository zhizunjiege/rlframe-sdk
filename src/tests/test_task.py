import json
import time
import unittest

import jsonschema

from src.rlsdk.task import Task, task_schema


class TaskTestCase(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        task_dir = 'src/tests/examples'

        with open(f'{task_dir}/task.json', 'r') as f:
            configs = json.load(f)

        with open(f'{task_dir}/simenv/args.json', 'r') as f1, \
             open(f'{task_dir}/simenv/sim_term_func.cpp', 'r') as f2:
            args = json.load(f1)
            args['proxy']['sim_term_func'] = f2.read()

        with open(f'{task_dir}/agent/hypers.json', 'r') as f1, \
             open(f'{task_dir}/agent/states_inputs_func.py', 'r') as f2, \
             open(f'{task_dir}/agent/outputs_actions_func.py', 'r') as f3, \
             open(f'{task_dir}/agent/reward_func.py', 'r') as f4:
            hypers = json.load(f1)
            sifunc = f2.read()
            oafunc = f3.read()
            rewfunc = f4.read()

        configs['simenv-main']['configs']['args'] = args
        configs['agent-main']['configs']['hypers'] = hypers
        configs['agent-main']['configs']['sifunc'] = sifunc
        configs['agent-main']['configs']['oafunc'] = oafunc
        configs['agent-main']['configs']['rewfunc'] = rewfunc

        cls.task_dir = task_dir
        cls.configs = configs
        cls.bff_addr = 'localhost:10000'

    @classmethod
    def tearDownClass(cls):
        ...

    def test_00_schema(self):
        jsonschema.validate(instance=self.configs, schema=task_schema)

    def test_01_push(self):
        task = Task(configs=self.configs)
        task.push(bff_addr=self.bff_addr, reset=True)
        self.assertTrue(task.inited)

    def test_02_pull(self):
        task = Task()
        task.pull(bff_addr=self.bff_addr, reset=True)
        self.assertTrue(task.inited)

    def test_03_details(self):
        task = Task()
        task.pull(bff_addr=self.bff_addr, reset=True)
        details = task.details()
        self.assertIn('simenv-main', details)
        self.assertIn('agent-main', details)

    def test_04_control(self):
        task = Task()
        task.pull(bff_addr=self.bff_addr, reset=True)
        task.start()
        time.sleep(5)
        task.pause()
        task.resume()
        time.sleep(5)
        task.stop()

    def test_05_monitor(self):
        task = Task()
        task.pull(bff_addr=self.bff_addr, reset=True)
        infos = task.monitor()
        self.assertIn('simenv-main', infos)

    def test_06_switch_training(self):
        task = Task()
        task.pull(bff_addr=self.bff_addr, reset=True)
        task.switch_training(mode=True)

    def test_07_weights(self):
        task = Task()
        task.pull(bff_addr=self.bff_addr, reset=True)
        weights = task.get_weights(id='agent-main')
        self.assertIsInstance(weights, dict)

    def test_08_buffer(self):
        task = Task()
        task.pull(bff_addr=self.bff_addr, reset=True)
        buffer = task.get_buffer(id='agent-main')
        self.assertIsInstance(buffer, dict)

    def test_09_status(self):
        task = Task()
        task.pull(bff_addr=self.bff_addr, reset=True)
        status = task.get_status(id='agent-main')
        self.assertIsInstance(status, dict)
