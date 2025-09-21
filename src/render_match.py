import os, argparse, numpy as np, imageio
from envs.tiny_pong import TinyPong
from bots import TrackingBot, RandomBot
from rl.dqn import QNet
from visualize import render_frame
import importlib.util, torch

STEP_CAP = 2000

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
            import torch

            with torch.no_grad():
                return int(
                    torch.argmax(
                        self.net(torch.tensor(obs, dtype=torch.float32).unsqueeze(0))
                    ).item()
                )

    a = Agent("cpu")
    a.load(path)
    return a


def load_submission_dir(dir_path):
    agent_py = os.path.join(dir_path, "agent.py")
    ckpt = os.path.join(dir_path, "checkpoint.pt")
    spec = importlib.util.spec_from_file_location("agent_mod", agent_py)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    Agent = getattr(mod, "Agent")
    a = Agent(device="cpu")
    a.load(ckpt)
    return a


def load_any(tag):
    if tag == "baseline_tracking":
        return TrackingBot()
    if tag == "baseline_random":
        return RandomBot()
    if os.path.isdir(tag):
        return load_submission_dir(tag)
    if os.path.isfile(tag):
        return load_agent_from_checkpoint(tag)
    raise ValueError(f"Unknown agent input: {tag}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--agent_a", required=True)
    ap.add_argument("--agent_b", required=True)
    ap.add_argument("--out", default="runs/match.mp4")
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--height", type=int, default=480)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--points", type=int, default=5)
    args = ap.parse_args()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    A = load_any(args.agent_a)
    B = load_any(args.agent_b)
    env = TinyPong(seed=args.seed)
    writer = imageio.get_writer(args.out, fps=args.fps)
    scoreA = scoreB = 0
    serve = 0
    total_steps = 0

    def step_point(serve_dir):
        nonlocal scoreA, scoreB, total_steps
        obs_l, obs_r = env.reset(serve_to=serve_dir)
        done = False
        while not done:
            if total_steps >= STEP_CAP:
                return True
            print(f"Step {total_steps} Score {scoreA}-{scoreB}", end="\r")
            writer.append_data(
                render_frame(env, args.width, args.height, scoreA, scoreB)
            )
            a_l = A.act(obs_l)
            a_r = B.act(obs_r)
            (obs_l, obs_r), (rw_l, rw_r), done, info = env.step(a_l, a_r)
            total_steps += 1
        writer.append_data(render_frame(env, args.width, args.height, scoreA, scoreB))
        if info.get("point") == "left":
            scoreA += 1
        elif info.get("point") == "right":
            scoreB += 1
        return False

    while scoreA < args.points and scoreB < args.points:
        if step_point(serve):
            print(f"Reached step limit of {STEP_CAP}. Exiting.")
            break
        serve = 1 - serve

    for _ in range(args.fps // 2):
        writer.append_data(render_frame(env, args.width, args.height, scoreA, scoreB))
    writer.close()
    print("Saved", args.out)


if __name__ == "__main__":
    main()
