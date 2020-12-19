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
        self.l3 = nn.Linear(hidden_size, action_size)
        self.log_std = nn.Parameter(-0.5 * torch.ones(action_size, dtype=torch.float32))

    def forward(self, x):
        x = torch.tanh(self.l1(x))
        x = torch.tanh(self.l2(x))
        mu = self.l3(x)
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
        self.l3 = nn.Linear(hidden_size, 1) 

    def forward(self, x):
        x = torch.tanh(self.l1(x))
        x = torch.tanh(self.l2(x))
        x = self.l3(x)
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
        self.rewards = np.zeros(size, dtype=np.float32)

    def normalise_advantages(self):
        mu = np.mean(self.advantages)
        sigma = np.std(self.advantages)
        self.advantages = (self.advantages - mu) / sigma
        return

class PPO():

    def __init__(self):
        # initialise variables
        self.batch_size = 20000
        self.lr = 1e-4
        self.gamma = 0.99
        self.lam = 0.95
        self.n_actor_updates = 1
        self.n_critic_updates = 1
        self.epsilon = 0.2

        # setup the environment
        self.env = Env()

        # setup actor critic nn models
        self.actor = Actor(self.env.state_size, self.env.action_size, 256)
        self.critic = Critic(self.env.state_size, 256)

        # setup gamm lambda arrays to speed up calculation of advantage estimates
        self.gamma_arr = np.zeros(self.env.max_steps, dtype=np.float32)
        self.gamma_lam_arr = np.zeros(self.env.max_steps, dtype=np.float32)
        for i in range(self.env.max_steps):
            self.gamma_arr[i] = pow(self.gamma, i)
            self.gamma_lam_arr[i] = pow(self.gamma * self.lam, i)

        # setup optimisers
        self.actor_opt = torch.optim.Adam(self.actor.parameters(), self.lr)
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), self.lr)


    def episode(self, data):
        
        # reset the state
        self.env.reset()

        # run the episode
        n_steps = self.env.max_steps
        for i in range(self.env.max_steps):

            # add state to the array of episode states
            data.states[i, :] = self.env.state[:]

            # get the action
            mu = self.actor(torch.reshape(torch.from_numpy(self.env.state), (1,-1)))
            pi = self.actor.dist(mu)
            action = pi.sample()
            log_prob = self.actor.log_prob(pi, action)

            # add to the arrays of episode values
            data.actions[i, :] = action.numpy()[:]
            data.log_probs[i] = log_prob

            # update the environment
            reward, done = self.env.step(action)

            # add the reward to the array of episode rewards
            data.rewards[i] = reward

            # check for the episode ending
            if done:
                n_steps = i+1
                break

        return n_steps


    def get_batch(self):

        # set up date buffers for the batch and for individual episodes
        data_batch = Buffer(self.batch_size, self.env.state_size, self.env.action_size)
        data_episode = Buffer(self.env.max_steps, self.env.state_size, self.env.action_size)

        n = 0
        while n < self.batch_size:

            # run an episode, sz is the number of states in the episode, not +1 for the extra value
            sz = self.episode(data_episode)

            # calculate the rewards to go
            for i in range(sz):
                data_episode.rewards_to_go[i] = np.sum(data_episode.rewards[i:sz] * self.gamma_arr[:sz-i])

            # calculate the advantage estimates
            values = self.critic(torch.as_tensor(data_episode.states[:sz, :])).detach().numpy()
            values = np.append(values, np.zeros((1,1), dtype=np.float32), axis=0)
            delta = data_episode.rewards[:sz] + self.gamma * values[1:sz+1,0] - values[:sz,0]
            for i in range(sz):
                data_episode.advantages[i] = np.sum(delta[i:sz] * self.gamma_lam_arr[:sz-i])

            # update the batch data arrays
            s = n
            sz = min(sz, self.batch_size - n)
            n += sz
            data_batch.states[s:n, :] = data_episode.states[:sz, :]
            data_batch.actions[s:n, :] = data_episode.actions[:sz, :]
            data_batch.log_probs[s:n] = data_episode.log_probs[:sz]
            data_batch.rewards_to_go[s:n] = data_episode.rewards_to_go[:sz]
            data_batch.advantages[s:n] = data_episode.advantages[:sz]
            data_batch.values[s:n] = values[:sz,0]

        # normalise the advantage estimates
        data_batch.normalise_advantages()

        # change to torch tensors
        data_batch.states = torch.as_tensor(data_batch.states)
        data_batch.actions = torch.as_tensor(data_batch.actions)
        data_batch.log_probs = torch.as_tensor(data_batch.log_probs)
        data_batch.rewards_to_go = torch.as_tensor(data_batch.rewards_to_go)
        data_batch.advantages = torch.as_tensor(data_batch.advantages)
        data_batch.values = torch.as_tensor(data_batch.values)
        data_batch.rewards = torch.as_tensor(data_batch.rewards)

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

        # # shift to the gpu
        # self.actor = self.actor.cuda()
        # self.critic = self.critic.cuda()
        # self.states = self.states.cuda()
        # self.actions = self.actions.cuda()
        # self.log_probs = self.log_probs.cuda()
        # self.advantages = self.advantages.cuda()
        # self.values = self.values.cuda()
        # self.rewards_to_go = self.rewards_to_go.cuda()

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

        # # shift back to the cpu
        # self.actor = self.actor.cpu()
        # self.critic = self.critic.cpu()
        # self.states = self.states.cpu()
        # self.actions = self.actions.cpu()
        # self.log_probs = self.log_probs.cpu()
        # self.advantages = self.advantages.cpu()
        # self.values = self.values.cpu()
        # self.rewards_to_go = self.rewards_to_go.cpu()

        return


    def train(self):
        
        updates_per_checkpoint = int(1e7 / self.batch_size)
        while True:
            for i in range(updates_per_checkpoint):
                st = time.time()

                # get a batch of data to train with
                print("get batch")
                data = self.get_batch()

                # update the actor critic networks
                print("update")
                self.update(data)

                print("batch update time: ", time.time() - st)
            break
            # checkpoint the model

if __name__ == "__main__":
    ppo = PPO()
    ppo.train()