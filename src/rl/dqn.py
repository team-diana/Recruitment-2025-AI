import random, torch, torch.nn as nn, torch.nn.functional as F
from collections import deque
import numpy as np
class QNet(nn.Module):
    def __init__(self, obs_dim=6, hidden=128, actions=3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, actions),
        )
    def forward(self, x): return self.net(x)
class Replay:
    def __init__(self, cap=10000): self.buf=deque(maxlen=cap)
    def push(self,s,a,r,sp,d): self.buf.append((s,a,r,sp,d))
    def sample(self,bs=64):
        batch = random.sample(self.buf, bs)
        s,a,r,sp,d = map(np.array, zip(*batch))
        return s,a,r,sp,d
    def __len__(self): return len(self.buf)
def select_action(q, s, eps):
    if np.random.rand() < eps:
        return np.random.randint(0, 3)
    device = next(q.parameters()).device
    with torch.no_grad():
        state = torch.as_tensor(s, dtype=torch.float32, device=device).unsqueeze(0)
        return int(q(state).argmax(dim=1).item())
def train_step(q, tgt, opt, batch, device='cpu'):
    s,a,r,sp,d = batch
    s = torch.tensor(s, dtype=torch.float32, device=device)
    a = torch.tensor(a, dtype=torch.long, device=device)
    r = torch.tensor(r, dtype=torch.float32, device=device)
    sp= torch.tensor(sp, dtype=torch.float32, device=device)
    d = torch.tensor(d, dtype=torch.float32, device=device)
    qsa = q(s).gather(1, a.unsqueeze(1)).squeeze(1)
    with torch.no_grad(): y = r + (1-d) * 0.99 * tgt(sp).max(1)[0]
    loss = F.smooth_l1_loss(qsa, y)
    opt.zero_grad(); loss.backward(); opt.step()
    return float(loss.item())
