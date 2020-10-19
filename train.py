import torch
from prestart_env import Env
import math
import time

class PPOActorCritic(torch.nn.Module):
    def __init__(self, state_size, action_size):
        super(PPOActorCritic, self).__init__()

        self.l1 = torch.nn.Linear(state_size, 256)
        self.l2 = torch.nn.Linear(256, 512)
        self.l3 = torch.nn.Linear(512, 512)
        self.l4 = torch.nn.Linear(512, 256)
        self.l5 = torch.nn.Linear(256, action_size + 1)
        log_std = torch.ones(action_size) * -0.5
        self.log_std = torch.nn.Parameter(log_std)

    def forward(self, obs, act=None):

        x = torch.nn.tanh(self.l1(obs))
        x = torch.nn.tanh(self.l2(x))
        x = torch.nn.tanh(self.l3(x))
        x = torch.nn.tanh(self.l4(x))
        out = self.l5(x)

        return out

def run_episode():


def get_batch(env, model, gamma_arr, gamma_lam_arr, batch_size):

    # setup batch data buffers
    states_buf = torch.zeros(batch_size, env.state_size)
    actions_buf = torch.zeros(batch_size, env.action_size)
    log_probs_buf = torch.zeros(batch_size)
    r2g_buf = torch.zeros(batch_size)
    adv_buf = torch.zeros(batch_size)
    val_buf = torch.zeros(batch_size)

    # setup episode data buffers
    states = torch.zeros(env.max_steps, env.state_size)
    actions = torch.zeros(env.max_steps, env.action_size)
    values = torch.zeros(env.max_steps + 1)
    combined_actions = torch.zeros(env.action_size * 2)
    log_probs = torch.zeros(env.max_steps)
    rewards = torch.zeros(env.max_steps)
    state = torch.zeros(env.state_size)
    norm_state = torch.zeros(env.state_size)
    row_buffer = torch.zeros(int(2 / env.timestep))
    i = 0
    while i < batch_size:

        # run an episode

        # calculate the rewards to go

        # calculate the advantage estimates

        # update the buffers

    # normalise the advantage estimates

    return

def train():

    # set hyper parameters
    gamma = 0.99
    lam = 0.95
    lr = 3e-4
    batch_size = 200000
    n_epochs = int(1e7 / batch_size)

    # setup the environment and model
    env = Env()
    model = PPOActorCritic(env.state_size, env.action_size)
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    # setup arrays to speed up the calculation of advantage estimates
    gamma_arr = torch.zeros(env.max_steps)
    gamma_lam_arr = torch.zeros(env.max_steps)
    for i in range(env.max_steps):
        gamma_arr[i] = math.pow(gamma, i-1)
        gamma_lam_arr[i] = math.pow(gamma * lam, i-1)

    # run training
    while True:
        ct = time.time()
        for i in range(n_epochs):
            st = time.time()

            # get a batch of data to train on
            states, actions, adv, log_probs, r2g, values = get_batch(env, model, gamma_arr, gamma_lam_arr, batch_size)

            # update the actor critic network

            # timing
            print(f"time: {(time.time() - st):.3f}    ", end="\r")

        print(f"time for checkpoint: {(time.time() - ct):.3f}    ")
        break
