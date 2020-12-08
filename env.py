import numpy as np
import random

class Env():
    def __init__(self):
        self.dt = 1.0
        self.max_steps = 150
        self.state_size = 20
        self.action_size = 2
        self.state = np.zeros(self.state_size, dtype=np.float32)

    def step(self, action):
        self.state = np.random.rand(self.state_size).astype(np.float32)
        return random.random(), False

    def reset(self):
        self.state = np.random.rand(self.state_size).astype(np.float32)

    