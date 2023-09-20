from typing import Any, Dict, List, Union

from ..base import ConfigBase


class CQSIM(ConfigBase):
    """CQSIM engine config."""

    def __init__(
        self,
        *,
        ctrl_addr='localhost:50041',
        res_addr='localhost:8001',
        x_token='Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJkYXRhc2NvcGUiOiIiLCJleHAiOjQ4MTAxOTcxNTQsImlkZW50aXR5 \
            IjoxLCJuaWNlIjoiYWRtaW4iLCJvcmlnX2lhdCI6MTY1NjU2MTE1NCwicm9sZWlkIjoxLCJyb2xla2V5IjoiYWRtaW4iLCJyb2xlbmFtZ \
            SI6Iuezu-e7n-euoeeQhuWRmCJ9.BvjGw26L1vbWHwl0n8Y1_yTF-fiFNZNmIw20iYe7ToU',
        proxy_id='',
        scenario_id=0,
        exp_design_id=0,
        repeat_times=1,
        sim_start_time=0,
        sim_duration=1,
        time_step=50,
        speed_ratio=1,
        data: Dict[str, Dict[str, Union[str, List[str], Dict[str, Any]]]] = {},
        routes: Dict[str, List[str]] = {},
        simenv_addr='localhost:10001',
        sim_step_ratio=1,
        sim_term_func='',
    ):
        """Init config.

        Args:
            ctrl_addr: control server address.
            res_addr: resource server address.
            x_token: token for resource server.
            proxy_id: proxy model ID.
            scenario_id: scenario ID.
            exp_design_id: experimental design ID.
            repeat_times: times to repeat the scenario or experiment.
            sim_start_time: start time of scenario in timestamp.
            sim_duration: simulation duration in seconds.
            time_step: time step in milliseconds.
            speed_ratio: speed ratio.
            data: input and output data needed for interaction.
            routes: routes for engine.
            simenv_addr: simenv service address.
            sim_step_ratio: number of steps to take once request for decision.
            sim_term_func: termination function written in c++.
        """
        if not proxy_id:
            raise ValueError('proxy_id must be specified')

        self.ctrl_addr = ctrl_addr
        self.res_addr = res_addr
        self.x_token = x_token
        self.proxy_id = proxy_id

        if scenario_id <= 0 and exp_design_id <= 0:
            raise ValueError('scenario_id or exp_design_id must be specified')
        if repeat_times <= 0:
            raise ValueError('repeat_times must be positive')
        if sim_start_time < 0:
            raise ValueError('sim_start_time must be non-negative')
        if sim_duration <= 0:
            raise ValueError('sim_duration must be positive')
        if time_step <= 0:
            raise ValueError('time_step must be positive')
        if speed_ratio == 0:
            raise ValueError('speed_ratio can not be zero')

        self.scenario_id = scenario_id
        self.exp_design_id = exp_design_id
        self.repeat_times = repeat_times
        self.sim_start_time = sim_start_time
        self.sim_duration = sim_duration
        self.time_step = time_step
        self.speed_ratio = speed_ratio

        for name, model in data.items():
            if 'modelid' not in model:
                raise ValueError('modelid must be specified')
            if 'inputs' not in model:
                raise ValueError('model inputs must be specified')
            if 'outputs' not in model:
                raise ValueError('model outputs must be specified')

        self.data = data

        for addr, route in routes.items():
            for name in route:
                if name not in data:
                    raise ValueError(f'model {name} not found in data')

        self.routes = routes

        if sim_step_ratio <= 0:
            raise ValueError('sim_step_ratio must be positive')
        if not sim_term_func:
            raise ValueError('sim_term_func must be specified')

        self.simenv_addr = simenv_addr
        self.sim_step_ratio = sim_step_ratio
        self.sim_term_func = sim_term_func
