import os, argparse, numpy as np, torch
from tqdm import trange
from envs.tiny_pong import TinyPong
from bots import TrackingBot
from rl.dqn import QNet, Replay, select_action, train_step


def run_episode(env, policy, opp, train=False, epsilon=0.1):
    obs_l, obs_r = env.reset(serve_to=np.random.randint(0, 2))
    while True:
        a_l = policy(obs_l, epsilon) if train else policy(obs_l, 0.0)
        a_r = opp.act(obs_r)
        (nobs_l, nobs_r), (rw_l, rw_r), done, info = env.step(a_l, a_r)
        if train:
            d = 1.0 if done else 0.0
            yield obs_l, a_l, rw_l, nobs_l, d
        obs_l, obs_r = nobs_l, nobs_r
        if done:
            break


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=6000)
    ap.add_argument("--save_path", type=str, default="runs/checkpoint.pt")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.save_path), exist_ok=True)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    env = TinyPong(seed=args.seed)
    bot = TrackingBot(reaction=2, noise=0.02, seed=args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    q = QNet().to(device)
    tgt = QNet().to(device)
    tgt.load_state_dict(q.state_dict())
    opt = torch.optim.Adam(q.parameters(), lr=1e-3)
    buf = Replay(10000)

    def policy(obs, eps):
        return select_action(q, obs, eps)

    for _ in range(1000):
        for s, a, r, sp, d in run_episode(
            env, lambda o, e: np.random.randint(0, 3), bot, train=True
        ):
            buf.push(s, a, r, sp, d)

    eps_hi, eps_lo, eps_decay = 1.0, 0.05, 3000

    for ep in trange(args.episodes):
        eps = eps_lo + (eps_hi - eps_lo) * max(0.0, (eps_decay - ep) / eps_decay)
        for s, a, r, sp, d in run_episode(env, policy, bot, train=True, epsilon=eps):
            buf.push(s, a, r, sp, d)
            if len(buf) >= 512:
                train_step(q, tgt, opt, buf.sample(128), device=device)
        if ep % 100 == 0:
            tgt.load_state_dict(q.state_dict())

    torch.save(q.state_dict(), args.save_path)
    print("Saved", args.save_path)


if __name__ == "__main__":
    main()
