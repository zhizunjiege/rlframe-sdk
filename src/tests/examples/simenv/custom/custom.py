from typing import List, Tuple

from ..base import SimEngineBase, AnyDict, CommandType


class Custom(SimEngineBase):

    def __init__(
        self,
        *,
        scenario_id=0,
        exp_design_id=0,
    ):
        super().__init__()

        self.scenario_id = scenario_id
        self.exp_design_id = exp_design_id

    def control(self, type: CommandType, params: AnyDict = {}) -> bool:
        return True

    def monitor(self) -> Tuple[AnyDict, List[str]]:
        return {}, []

    def call(self, name: str, dstr='', dbin=b'') -> Tuple[str, str, bytes]:
        return name, '', b''
