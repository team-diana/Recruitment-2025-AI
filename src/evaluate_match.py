import os, argparse, importlib.util, numpy as np, torch
from envs.tiny_pong import TinyPong
from bots import TrackingBot, RandomBot
from rl.dqn import QNet


def load_agent_from_checkpoint(path):
    class Agent:
        def __init__(self, device="cpu"):
            self.device = device
            self.net = QNet().to(device)
            self.net.eval()

        def load(self, p):
            self.net.load_state_dict(torch.load(p, map_location=self.device))
            self.net.eval()

        def act(self, obs):
            with torch.no_grad():
                return int(
                    torch.argmax(
                        self.net(torch.tensor(obs, dtype=torch.float32).unsqueeze(0))
                    ).item()
                )

    a = Agent("cpu")
    a.load(path)

    return a


def play_match(agentA, agentB, seed=0, max_points=5, step_cap=50000):
    env = TinyPong(seed=seed)
    scoreA = scoreB = 0
    steps = 0
    serve = 0

    while scoreA < max_points and scoreB < max_points and steps < step_cap:
        obs_l, obs_r = env.reset(serve_to=serve)
        serve = 1 - serve
        done = False

        while not done and steps < step_cap:
            a_l = agentA.act(obs_l)
            a_r = agentB.act(obs_r)

            (obs_l, obs_r), (rw_l, rw_r), done, info = env.step(a_l, a_r)
            steps += 1

        if info.get("point") == "left":
            scoreA += 1
        elif info.get("point") == "right":
            scoreB += 1

    return scoreA, scoreB


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--agent_a", required=True)  # path or baseline_tracking/random
    ap.add_argument("--agent_b", required=True)
    args = ap.parse_args()

    def load_any(tag):
        if tag == "baseline_tracking":
            return TrackingBot()
        if tag == "baseline_random":
            return RandomBot()
        if os.path.exists(tag):
            return load_agent_from_checkpoint(tag)
        raise ValueError(f"Unknown agent {tag}")

    A = load_any(args.agent_a)
    B = load_any(args.agent_b)
    sA, sB = play_match(A, B, seed=0, max_points=5)
    print(f"Result: A {sA} - {sB} B")


if __name__ == "__main__":
    main()
