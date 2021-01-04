import time
import torch
import torch.nn as nn
from torch.distributions.normal import Normal
from env import Env
import numpy as np

class Actor(nn.Module):
    def __init__(self, state_size, action_size, hidden_size):
        super(Actor, self).__init__()
        self.l1 = nn.Linear(state_size, hidden_size)
        self.l2 = nn.Linear(hidden_size, hidden_size)
        self.l3 = nn.Linear(hidden_size, hidden_size)
        self.l4 = nn.Linear(hidden_size, hidden_size)
        self.l5 = nn.Linear(hidden_size, hidden_size)
        self.l6 = nn.Linear(hidden_size, hidden_size)
        self.l7 = nn.Linear(hidden_size, action_size)
        self.log_std = nn.Parameter(-0.5 * torch.ones(action_size, dtype=torch.float32))

    def forward(self, x):
        x = torch.relu(self.l1(x))
        x = torch.relu(self.l2(x))
        x = torch.relu(self.l3(x))
        x = torch.relu(self.l4(x))
        x = torch.relu(self.l5(x))
        x = torch.relu(self.l6(x))
        mu = self.l7(x)
        return mu

    def dist(self, mu):
        pi = Normal(mu, torch.exp(self.log_std))
        return pi

    def log_prob(self, pi, action):
        return pi.log_prob(action).sum(axis=-1)

class Critic(nn.Module):
    def __init__(self, state_size,  hidden_size):
        super(Critic, self).__init__()
        self.l1 = nn.Linear(state_size, hidden_size)
        self.l2 = nn.Linear(hidden_size, hidden_size)
        self.l3 = nn.Linear(hidden_size, hidden_size)
        self.l4 = nn.Linear(hidden_size, hidden_size)
        self.l5 = nn.Linear(hidden_size, hidden_size)
        self.l6 = nn.Linear(hidden_size, hidden_size)
        self.l7 = nn.Linear(hidden_size, 1)

    def forward(self, x):
        x = torch.relu(self.l1(x))
        x = torch.relu(self.l2(x))
        x = torch.relu(self.l3(x))
        x = torch.relu(self.l4(x))
        x = torch.relu(self.l5(x))
        x = torch.relu(self.l6(x))
        x = self.l7(x)
        return x

class Buffer():
    def __init__(self, size, state_size, action_size):
        self.size = size
        self.states = np.zeros((size, state_size), dtype=np.float32)
        self.actions = np.zeros((size, action_size), dtype=np.float32)
        self.log_probs = np.zeros(size, dtype=np.float32)
        self.rewards_to_go = np.zeros(size, dtype=np.float32)
        self.advantages = np.zeros(size, dtype=np.float32)
        self.values = np.zeros(size, dtype=np.float32)

    def normalise_advantages(self):
        mu = np.mean(self.advantages)
        sigma = np.std(self.advantages)
        self.advantages = (self.advantages - mu) / sigma
        return

class EpisodeBuffer():
    def __init__(self, n_parallel, size, state_size, action_size):
        self.size = size
        self.states = np.zeros((n_parallel, size, state_size), dtype=np.float32)
        self.actions = np.zeros((n_parallel, size, action_size), dtype=np.float32)
        self.log_probs = np.zeros((n_parallel, size), dtype=np.float32)
        self.values = np.zeros((n_parallel, size + 1), dtype=np.float32)
        self.rewards = np.zeros((n_parallel, size), dtype=np.float32)
        self.rewards_to_go = np.zeros((n_parallel, size), dtype=np.float32)
        self.advantages = np.zeros((n_parallel, size), dtype=np.float32)
        self.idxs = np.zeros(n_parallel, dtype=np.int)

class PPO():

    def __init__(self):
        # initialise variables
        self.batch_size = 20000
        self.lr = 1e-4
        self.gamma = 0.99
        self.lam = 0.95
        self.n_actor_updates = 10
        self.n_critic_updates = 10
        self.epsilon = 0.2
        self.n_parallel = 100

        # setup the environment
        self.env = Env(self.n_parallel)

        # setup actor critic nn models
        self.actor = Actor(self.env.state_size, self.env.action_size, 256).cuda()
        self.critic = Critic(self.env.state_size, 256).cuda()

        # setup gamm lambda arrays to speed up calculation of advantage estimates
        self.gamma_arr = np.zeros(self.env.max_steps, dtype=np.float32)
        self.gamma_lam_arr = np.zeros(self.env.max_steps, dtype=np.float32)
        for i in range(self.env.max_steps):
            self.gamma_arr[i] = pow(self.gamma, i)
            self.gamma_lam_arr[i] = pow(self.gamma * self.lam, i)

        # setup optimisers
        self.actor_opt = torch.optim.Adam(self.actor.parameters(), self.lr)
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), self.lr)


    # this section is 10x faster in Julia, try to speed it up using cython
    def get_batch(self):

        # set up date buffers for the batch and for individual episodes
        data_batch = Buffer(self.batch_size, self.env.state_size, self.env.action_size)
        data_episode = EpisodeBuffer(self.n_parallel, self.env.max_steps, self.env.state_size, self.env.action_size)

        # reset the states
        for i in range(self.n_parallel):
            self.env.reset(i)

        # collect a batch of data
        n = 0
        while n < self.batch_size:

            # normalise the states and add to the episode buffer of states
            # not needed for this env
            
            # get the actions, log_prob and value estimates
            state = torch.from_numpy(self.env.state).cuda()
            mu = self.actor(state)
            pi = self.actor.dist(mu)
            action = pi.sample()
            log_prob = self.actor.log_prob(pi, action)
            value = self.critic(state)
            action = action.cpu()
            log_prob = log_prob.cpu()
            value = value.cpu()

            # add to the episode buffers
            for i in range(self.n_parallel):
                data_episode.states[i, data_episode.idxs[i], :] = self.env.state[i, :]
                data_episode.actions[i, data_episode.idxs[i], :] = action[i, :]
                data_episode.values[i, data_episode.idxs[i]] = value[i, 0]
                data_episode.log_probs[i, data_episode.idxs[i]] = log_prob[i]
            

            # update the states
            rewards, dones = self.env.step(action)
            for i in range(self.n_parallel):

                # add the rewards to the episode buffer
                data_episode.rewards[i, data_episode.idxs[i]] = rewards[i]

                # check for the episode happening
                if dones[i] or data_episode.idxs[i] == self.env.max_steps - 1:
                    # add a final 0 value
                    sz = data_episode.idxs[i] + 1
                    data_episode.values[i, sz] = 0

                    # calculate rewards to go
                    for j in range(sz):
                        data_episode.rewards_to_go[i, j] = np.sum(data_episode.rewards[i, j:sz] * self.gamma_arr[:sz-j])

                    # calculate the advantage estimates
                    delta = data_episode.rewards[i, :sz] + self.gamma * data_episode.values[i, 1:sz+1] - data_episode.values[i, :sz]
                    for j in range(sz):
                        data_episode.advantages[i, j] = np.sum(delta[j:sz] * self.gamma_lam_arr[:sz-j])

                    # update the buffers
                    s = n
                    sz = min(sz, self.batch_size - n)
                    n += sz
                    data_batch.states[s:n, :] = data_episode.states[i, :sz, :]
                    data_batch.actions[s:n, :] = data_episode.actions[i, :sz, :]
                    data_batch.log_probs[s:n] = data_episode.log_probs[i, :sz]
                    data_batch.rewards_to_go[s:n] = data_episode.rewards_to_go[i, :sz]
                    data_batch.advantages[s:n] = data_episode.advantages[i, :sz]
                    data_batch.values[s:n] = data_episode.values[i, :sz]

                    # reset the episode index
                    data_episode.idxs[i] = 0

                    # reset this state
                    self.env.reset(i)

                else:
                     data_episode.idxs[i] += 1
            

        # normalise the advantage estimates
        data_batch.normalise_advantages()

        # change to torch tensors
        data_batch.states = torch.as_tensor(data_batch.states)
        data_batch.actions = torch.as_tensor(data_batch.actions)
        data_batch.log_probs = torch.as_tensor(data_batch.log_probs)
        data_batch.rewards_to_go = torch.as_tensor(data_batch.rewards_to_go)
        data_batch.advantages = torch.as_tensor(data_batch.advantages)
        data_batch.values = torch.as_tensor(data_batch.values)

        return data_batch


    def actor_loss(self, data):
        mu = self.actor(data.states)
        pi = self.actor.dist(mu)
        log_probs = self.actor.log_prob(pi, data.actions)
        ratio = torch.exp(log_probs - data.log_probs)
        clip_advantage = torch.clamp(ratio, 1-self.epsilon, 1+self.epsilon) * data.advantages
        loss = -(torch.min(ratio * data.advantages, clip_advantage)).mean()
        return loss


    def critic_loss(self, data):
        values = self.critic(data.states)[:,0]
        v_clip = data.values + torch.clamp(values - data.values, -self.epsilon, self.epsilon)
        v_loss1 = torch.pow(values - data.rewards_to_go, 2)
        v_loss2 = torch.pow(v_clip - data.rewards_to_go, 2)
        loss = torch.max(v_loss1, v_loss2).mean()
        return loss


    def update(self, data):

        # shift to the gpu
        data.states = data.states.cuda()
        data.actions = data.actions.cuda()
        data.log_probs = data.log_probs.cuda()
        data.advantages = data.advantages.cuda()
        data.values = data.values.cuda()
        data.rewards_to_go = data.rewards_to_go.cuda()

        # update the actor
        for i in range(self.n_actor_updates):
            self.actor_opt.zero_grad()
            loss_pi = self.actor_loss(data)
            loss_pi.backward()
            self.actor_opt.step()

        # update the critic
        for i in range(self.n_critic_updates):
            self.critic_opt.zero_grad()
            loss_v = self.critic_loss(data)
            loss_v.backward()
            self.critic_opt.step()

        return


    def train(self):
        
        updates_per_checkpoint = int(1e7 / self.batch_size)
        while True:
            for _ in range(updates_per_checkpoint):
                
                # get a batch of data to train with
                st = time.time()
                data = self.get_batch()
                bt = time.time() - st

                # update the actor critic networks
                st = time.time()
                self.update(data)
                ut = time.time() - st

                # timing for performance tracking
                print(f"batch time: {bt}, update time: {ut}")
            
            # checkpoint the model
            

if __name__ == "__main__":
    ppo = PPO()
    ppo.train()

