import time
import unittest

from src.rlsdk.task import Task


class TaskTestCase(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.path = 'src/tests/examples'
        cls.address = 'localhost:10000'

    @classmethod
    def tearDownClass(cls):
        ...

    def test_00_fromfiles(self):
        task = Task.from_files(path=self.path)
        self.assertIsInstance(task, Task)

    def test_01_push(self):
        task = Task.from_files(path=self.path)
        task.push(address=self.address, reset=True)
        self.assertTrue(task.inited)

    def test_02_pull(self):
        task = Task()
        task.pull(address=self.address, reset=True)
        self.assertTrue(task.inited)

    def test_03_details(self):
        task = Task()
        task.pull(address=self.address, reset=True)
        details = task.details()
        self.assertIn('agent', details)
        self.assertIn('simenv', details)

    def test_04_switch_training(self):
        task = Task()
        task.pull(address=self.address, reset=True)
        task.switch_training()
        task.switch_training()

    def test_05_weights(self):
        task = Task()
        task.pull(address=self.address, reset=True)
        weights = task.get_weights(id='agent')
        self.assertIsInstance(weights, dict)
        task.set_weights(id='agent', weights=weights)

    def test_06_buffer(self):
        task = Task()
        task.pull(address=self.address, reset=True)
        buffer = task.get_buffer(id='agent')
        self.assertIsInstance(buffer, dict)
        task.set_buffer(id='agent', buffer=buffer)

    def test_07_status(self):
        task = Task()
        task.pull(address=self.address, reset=True)
        status = task.get_status(id='agent')
        self.assertIsInstance(status, dict)
        task.set_status(id='agent', status=status)

    def test_08_control(self):
        task = Task()
        task.pull(address=self.address, reset=True)
        task.init()
        time.sleep(5)
        task.start()
        time.sleep(5)
        task.pause()
        task.resume()
        time.sleep(5)
        task.stop()

    def test_09_monitor(self):
        task = Task()
        task.pull(address=self.address, reset=True)
        infos = task.monitor()
        self.assertIn('simenv', infos)
