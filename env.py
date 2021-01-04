import numpy as np
import random

class Env():
    def __init__(self, n_parallel):
        self.dt = 1.0
        self.max_steps = 100
        self.state_size = 20
        self.action_size = 2
        self.n_parallel = n_parallel
        self.state = np.random.rand(self.n_parallel, self.state_size).astype(np.float32)

    def step(self, actions):
        self.state = np.random.rand(self.n_parallel, self.state_size).astype(np.float32)
        rewards = [random.random() + actions[i,0] for i in range(self.n_parallel)]
        dones = [False for _ in range(self.n_parallel)]
        return rewards, dones

    def reset(self, idx):
        self.state[idx, :] = np.random.rand(self.state_size).astype(np.float32)

    