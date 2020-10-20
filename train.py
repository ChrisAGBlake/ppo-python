import torch
import numpy as np
from prestart_env import Env
import math
import time
import random

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
        self.action_size = action_size

    def forward(self, obs, act=None):

        x = torch.tanh(self.l1(obs))
        x = torch.tanh(self.l2(x))
        x = torch.tanh(self.l3(x))
        x = torch.tanh(self.l4(x))
        out = self.l5(x)
        mu = out[:, :self.action_size]
        std = torch.exp(self.log_std)
        pi = torch.distributions.normal.Normal(mu, std)
        value = out[:, self.action_size]
        if act is None:
            act = pi.sample()
        log_prob = pi.log_prob(act).sum(axis=-1)

        return act, log_prob, value


def run_episode(env, model, states, actions, values, log_probs, rewards):

    env.reset()
    i = 0
    while i < env.max_steps:

        # add the state to the buffer
        states[i, :] = torch.from_numpy(env.state[:])

        # get action, log prob and value estimate
        action, log_prob, value = model(torch.unsqueeze(torch.from_numpy(env.norm_state), 0))

        # add to the buffers
        actions[i, :] = action[:]
        log_probs[i] = log_prob
        values[i] = value

        # get the opponent action
        combined_actions = np.zeros(env.action_size * 2, dtype=np.float32)
        combined_actions[2] = random.random() * 2 - 1
        combined_actions[3] = random.random()

        # combine boat 1 and boat 2 actions
        combined_actions[:env.action_size] = action.numpy()

        # update the environment
        reward, done = env.step(combined_actions)

        # add reward to the buffer
        rewards[i] = reward

        # check for the episode ending
        if done:
            break

        i += 1

    # set the final value to 0
    i += 1
    values[i] = 0

    return i


def discount_cumsum(values, discount_array, sz):
    res = torch.zeros(sz)
    for i in range(sz):
        res[i] = torch.sum(values[i:sz] * discount_array[:sz-i])
    return res


def get_batch(env, model, gamma, gamma_arr, gamma_lam_arr, batch_size):

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
    log_probs = torch.zeros(env.max_steps)
    rewards = torch.zeros(env.max_steps)
    i = 0
    while i < batch_size:

        # run an episode
        sz = run_episode(env, model, states, actions, values, log_probs, rewards)

        # calculate the rewards to go
        r2g = discount_cumsum(rewards[:sz], gamma_arr, sz)

        # calculate the advantage estimates
        delta = rewards[:sz] + gamma * values[1:sz+1] - values[:sz]
        adv_est = discount_cumsum(delta, gamma_lam_arr, sz)

        # update the buffers
        sz = min(sz, batch_size - i)
        states_buf[i:i+sz, :] = states[:sz,:]
        actions_buf[i:i+sz, :] = actions[:sz,:]
        log_probs_buf[i:i+sz] = log_probs[:sz]
        r2g_buf[i:i+sz] = r2g[:sz]
        adv_buf[i:i+sz] = adv_est[:sz]
        val_buf[i:i+sz] = values[:sz]
        i += sz

    # normalise the advantage estimates
    m = torch.mean(adv_buf)
    s = torch.std(adv_buf)
    adv_buf = (adv_buf - m) / s

    return states_buf, actions_buf, log_probs_buf, r2g_buf, adv_buf, val_buf


def compute_loss(model, states, actions, log_probs_old, r2g, adv_est, values_old, epsilon, ent_coef, vf_coef):

    # get updated log_probs and values for the current states and actions
    _, log_probs, values = model(states, act=actions)

    # calculate policy loss
    ratio = torch.exp(log_probs - log_probs_old)
    clip_adv = torch.clamp(ratio, 1-epsilon, 1+epsilon) * adv_est
    p_loss = -torch.mean(torch.min(ratio * adv_est, clip_adv))

    # calculate entropy
    entropy = torch.sum(model.log_std + 0.5 * math.log(2 * math.pi * math.e))
    entropy *= ent_coef

    # calculate the value function loss
    v_clip = values_old + torch.clamp(values - values_old, -epsilon, epsilon)
    v_loss1 = torch.pow(values - r2g, 2)
    v_loss2 = torch.pow(v_clip - r2g, 2)
    v_loss = vf_coef * torch.mean(torch.max(v_loss1, v_loss2))

    return p_loss + v_loss - entropy


def update(model, opt, states, actions, log_probs, r2g, adv, values, epsilon, ent_coef, vf_coef):

    for i in range(10):
        opt.zero_grad()
        loss = compute_loss(model, states, actions, log_probs, r2g, adv, values, epsilon, ent_coef, vf_coef)
        loss.backward()
        opt.step()
    return


def train():

    # set hyper parameters
    gamma = 0.99
    lam = 0.95
    lr = 3e-4
    epsilon = 0.2
    vf_coef = 0.25
    ent_coef = 0.01
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
            states, actions, log_probs, r2g, adv, values = get_batch(env, model, gamma, gamma_arr, gamma_lam_arr, batch_size)
            print("batch time: ", time.time() - st)

            # update the actor critic network
            update(model, opt, states, actions, log_probs, r2g, adv, values, epsilon, ent_coef, vf_coef)

            # timing
            print(f"time: {(time.time() - st):.3f}    ", end="\r")

        print(f"time for checkpoint: {(time.time() - ct):.3f}    ")
        break

    return

if __name__ == "__main__":
    train()

