import json
from abc import ABC, abstractmethod
from typing import Tuple

from eval_agent.utils.datatypes import State


class BaseEnv(ABC):
    def __init__(
        self,
        instruction_path: str,
        icl_path: str,
        icl_format: str = "first",
        max_steps: int = 10,
        **kwargs,
    ):
        with open(instruction_path) as f:
            self.instruction = f.read()
        self.raw_icl = json.load(open(icl_path))
        # print('[DEBUG] raw_icl: ', self.raw_icl) # 正しくwebshop_icl.jsonが読み込めている
        self.icl_format = icl_format
        self.max_steps = max_steps

    @abstractmethod
    def step(self, llm_output: str) -> Tuple[str, State]:
        pass

    @abstractmethod
    def reset(self) -> Tuple[str, State]:
        pass
