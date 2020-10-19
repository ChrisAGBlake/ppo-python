import torch


class Env():
    def __init__(self):
        self.state_size = 20
        self.action_size = 2
        self.max_steps = 150
        self.timestep = 1.0
        self.state = torch.zeros(self.state_size)
        self.norm_state = torch.zeros(self.state_size)

