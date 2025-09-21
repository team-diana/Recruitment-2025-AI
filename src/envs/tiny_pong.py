import math, numpy as np


class TinyPong:
    def __init__(
        self,
        paddle_h=0.25,
        paddle_speed=0.06,
        ball_speed=0.035,
        spin=0.015,
        max_steps=2000,
        seed=0,
    ):
        self.rng = np.random.RandomState(seed)
        self.paddle_h = paddle_h
        self.paddle_speed = paddle_speed
        self.ball_speed = ball_speed
        self.spin = spin
        self.max_steps = max_steps
        self.reset(serve_to=self.rng.choice([0, 1]))

    def seed(self, s):
        self.rng = np.random.RandomState(s)

    def reset(self, serve_to=0):
        self.ball_x = 0.0
        self.ball_y = 0.0
        ang = self.rng.uniform(-0.5, 0.5)
        self.ball_vx = (
            (1 if serve_to == 0 else -1)
            * self.ball_speed
            * (1.0 + self.rng.uniform(-0.05, 0.05))
        )
        self.ball_vy = self.ball_speed * math.sin(ang)
        self.yl = 0.0
        self.yr = 0.0
        self.steps = 0
        return self._obs_left(), self._obs_right()

    def _obs_left(self):
        return np.array(
            [self.ball_x, self.ball_y, self.ball_vx, self.ball_vy, self.yl, self.yr],
            dtype=np.float32,
        )

    def _obs_right(self):
        return np.array(
            [-self.ball_x, self.ball_y, -self.ball_vx, self.ball_vy, self.yr, self.yl],
            dtype=np.float32,
        )

    def _clamp(self, y):
        return float(np.clip(y, -1.0 + self.paddle_h / 2, 1.0 - self.paddle_h / 2))

    def step(self, a_left: int, a_right: int):
        self.steps += 1
        d = {0: -self.paddle_speed, 1: 0.0, 2: self.paddle_speed}
        self.yl = self._clamp(self.yl + d.get(a_left, 0.0))
        self.yr = self._clamp(self.yr + d.get(a_right, 0.0))

        self.ball_x += self.ball_vx
        self.ball_y += self.ball_vy

        if self.ball_y >= 1.0 and self.ball_vy > 0:
            self.ball_y = 1.0
            self.ball_vy *= -1
        if self.ball_y <= -1.0 and self.ball_vy < 0:
            self.ball_y = -1.0
            self.ball_vy *= -1

        if self.ball_x <= -0.98 and self.ball_vx < 0:
            if abs(self.ball_y - self.yl) <= self.paddle_h / 2:
                self.ball_x = -0.98
                self.ball_vx = abs(self.ball_vx)
                off = (self.ball_y - self.yl) / (self.paddle_h / 2)
                self.ball_vy += off * self.spin
                self._limit_speed()
            else:
                obs_l, obs_r = self.reset(serve_to=0)
                return (obs_l, obs_r), (-1.0, +1.0), True, {"point": "right"}

        if self.ball_x >= 0.98 and self.ball_vx > 0:
            if abs(self.ball_y - self.yr) <= self.paddle_h / 2:
                self.ball_x = 0.98
                self.ball_vx = -abs(self.ball_vx)
                off = (self.ball_y - self.yr) / (self.paddle_h / 2)
                self.ball_vy += off * self.spin
                self._limit_speed()
            else:
                obs_l, obs_r = self.reset(serve_to=1)
                return (obs_l, obs_r), (+1.0, -1.0), True, {"point": "left"}

        done = self.steps >= self.max_steps
        return (self._obs_left(), self._obs_right()), (0.0, 0.0), done, {}

    def _limit_speed(self):
        max_vy = self.ball_speed * 1.5
        self.ball_vy = float(np.clip(self.ball_vy, -max_vy, max_vy))
        s = math.hypot(self.ball_vx, self.ball_vy)
        if s == 0:
            self.ball_vx = self.ball_speed
            return
        sc = self.ball_speed / s
        self.ball_vx *= sc
        self.ball_vy *= sc
