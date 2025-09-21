import abc, numpy as np


class BaseAgent(abc.ABC):
    @abc.abstractmethod
    def load(self, checkpoint_path: str): ...
    @abc.abstractmethod
    def act(self, obs: np.ndarray) -> int: ...
