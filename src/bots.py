import numpy as np


class TrackingBot:
    def __init__(self, reaction=2, noise=0.02, seed=0):
        self.reaction = reaction
        self.noise = noise
        self.buf = []
        self.rng = np.random.RandomState(seed)

    def act(self, obs):
        self.buf.append(obs.copy())
        target = obs[1]

        if len(self.buf) > self.reaction:
            target = self.buf.pop(0)[1]

        target += self.rng.uniform(-self.noise, self.noise)
        my = obs[4]

        if target > my + 0.02:
            return 2
        if target < my - 0.02:
            return 0

        return 1


class RandomBot:
    def __init__(self, seed=0):
        self.rng = np.random.RandomState(seed)

    def act(self, obs):
        return int(self.rng.randint(0, 3))
