import numpy as np
import random
from config import *

class Env():
    def __init__(self):
        self.state = np.random.rand(n_parallel, state_size).astype(np.float32)

    def step(self, actions):
        self.state = np.random.rand(n_parallel, state_size).astype(np.float32)
        rewards = [random.random() + actions[i,0] for i in range(n_parallel)]
        dones = [False for _ in range(n_parallel)]
        return rewards, dones

    def reset(self, idx):
        self.state[idx, :] = np.random.rand(state_size).astype(np.float32)

    