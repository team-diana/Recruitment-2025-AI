import torch, numpy as np
from rl.dqn import QNet
from agent_base import BaseAgent


class Agent(BaseAgent):
    def __init__(self, device="cpu"):
        self.name = "baseline_agent"
        self.device = device
        self.net = QNet().to(device)
        self.net.eval()

    def load(self, checkpoint_path):
        self.net.load_state_dict(torch.load(checkpoint_path, map_location=self.device))
        self.net.eval()

    def act(self, obs: np.ndarray) -> int:
        with torch.no_grad():
            return int(
                torch.argmax(
                    self.net(torch.tensor(obs, dtype=torch.float32).unsqueeze(0))
                ).item()
            )
