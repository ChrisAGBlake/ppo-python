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


    def get_batch(self):

        # setup data buffers for the batch of episodes
        batch_states = np.zeros((self.batch_size, self.env.state_size), dtype=np.float32)
        batch_actions = np.zeros((self.batch_size, self.env.action_size), dtype=np.float32)
        batch_log_probs = np.zeros(self.batch_size, dtype=np.float32)
        batch_rewards_to_go = np.zeros(self.batch_size, dtype=np.float32)
        batch_advantages = np.zeros(self.batch_size, dtype=np.float32)
        batch_values = np.zeros(self.batch_size, dtype=np.float32)

        # setup data buffers for the indiviual episodes
        episode_states = np.zeros((self.n_parallel, self.env.max_steps, self.env.state_size), dtype=np.float32)
        episode_actions = np.zeros((self.n_parallel, self.env.max_steps, self.env.action_size), dtype=np.float32)
        episode_log_probs = np.zeros((self.n_parallel, self.env.max_steps), dtype=np.float32)
        episode_values = np.zeros((self.n_parallel, self.env.max_steps + 1), dtype=np.float32)
        episode_rewards = np.zeros((self.n_parallel, self.env.max_steps), dtype=np.float32)
        episode_rewards_to_go = np.zeros((self.n_parallel, self.env.max_steps), dtype=np.float32)
        episode_advantages = np.zeros((self.n_parallel, self.env.max_steps), dtype=np.float32)
        episode_idxs = np.zeros(self.n_parallel, dtype=np.int)

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
                episode_states[i, episode_idxs[i], :] = self.env.state[i, :]
                episode_actions[i, episode_idxs[i], :] = action[i, :]
                episode_values[i, episode_idxs[i]] = value[i, 0]
                episode_log_probs[i, episode_idxs[i]] = log_prob[i]
            

            # update the states
            rewards, dones = self.env.step(action)
            for i in range(self.n_parallel):

                # add the rewards to the episode buffer
                episode_rewards[i, episode_idxs[i]] = rewards[i]

                # check for the episode happening
                if dones[i] or episode_idxs[i] == self.env.max_steps - 1:
                    # add a final 0 value
                    sz = episode_idxs[i] + 1
                    episode_values[i, sz] = 0

                    # calculate rewards to go
                    for j in range(sz):
                        episode_rewards_to_go[i, j] = np.sum(episode_rewards[i, j:sz] * self.gamma_arr[:sz-j])

                    # calculate the advantage estimates
                    delta = episode_rewards[i, :sz] + self.gamma * episode_values[i, 1:sz+1] - episode_values[i, :sz]
                    for j in range(sz):
                        episode_advantages[i, j] = np.sum(delta[j:sz] * self.gamma_lam_arr[:sz-j])

                    # update the buffers
                    s = n
                    sz = min(sz, self.batch_size - n)
                    n += sz
                    batch_states[s:n, :] = episode_states[i, :sz, :]
                    batch_actions[s:n, :] = episode_actions[i, :sz, :]
                    batch_log_probs[s:n] = episode_log_probs[i, :sz]
                    batch_rewards_to_go[s:n] = episode_rewards_to_go[i, :sz]
                    batch_advantages[s:n] = episode_advantages[i, :sz]
                    batch_values[s:n] = episode_values[i, :sz]

                    # reset the episode index
                    episode_idxs[i] = 0

                    # reset this state
                    self.env.reset(i)

                else:
                    episode_idxs[i] += 1
            

        # normalise the advantage estimates
        mu = np.mean(batch_advantages)
        sigma = np.std(batch_advantages)
        batch_advantages = (batch_advantages - mu) / sigma

        # change to torch tensors
        batch_states = torch.as_tensor(batch_states)
        batch_actions = torch.as_tensor(batch_actions)
        batch_log_probs = torch.as_tensor(batch_log_probs)
        batch_rewards_to_go = torch.as_tensor(batch_rewards_to_go)
        batch_advantages = torch.as_tensor(batch_advantages)
        batch_values = torch.as_tensor(batch_values)

        return batch_states, batch_actions, batch_log_probs, batch_rewards_to_go, batch_advantages, batch_values


    def actor_loss(self, states, actions, log_probs, advantages):
        mu = self.actor(states)
        pi = self.actor.dist(mu)
        new_log_probs = self.actor.log_prob(pi, actions)
        ratio = torch.exp(new_log_probs - log_probs)
        clip_advantage = torch.clamp(ratio, 1-self.epsilon, 1+self.epsilon) * advantages
        loss = -(torch.min(ratio * advantages, clip_advantage)).mean()
        return loss


    def critic_loss(self, states, rewards_to_go, values):
        new_values = self.critic(states)[:,0]
        v_clip = values + torch.clamp(new_values - values, -self.epsilon, self.epsilon)
        v_loss1 = torch.pow(new_values - rewards_to_go, 2)
        v_loss2 = torch.pow(v_clip - rewards_to_go, 2)
        loss = torch.max(v_loss1, v_loss2).mean()
        return loss


    def update(self, states, actions, log_probs, rewards_to_go, advantages, values):

        # shift to the gpu
        states = states.cuda()
        actions = actions.cuda()
        log_probs = log_probs.cuda()
        advantages = advantages.cuda()
        values = values.cuda()
        rewards_to_go = rewards_to_go.cuda()

        # update the actor
        for i in range(self.n_actor_updates):
            self.actor_opt.zero_grad()
            loss_pi = self.actor_loss(states, actions, log_probs, advantages)
            loss_pi.backward()
            self.actor_opt.step()

        # update the critic
        for i in range(self.n_critic_updates):
            self.critic_opt.zero_grad()
            loss_v = self.critic_loss(states, rewards_to_go, values)
            loss_v.backward()
            self.critic_opt.step()

        return


    def train(self):
        
        updates_per_checkpoint = int(1e7 / self.batch_size)
        while True:
            for _ in range(updates_per_checkpoint):
                
                # get a batch of data to train with
                st = time.time()
                states, actions, log_probs, rewards_to_go, advantages, values = self.get_batch()
                bt = time.time() - st

                # update the actor critic networks
                st = time.time()
                self.update(states, actions, log_probs, rewards_to_go, advantages, values)
                ut = time.time() - st

                # timing for performance tracking
                print(f"batch time: {bt}, update time: {ut}")
            
            # checkpoint the model
            

if __name__ == "__main__":
    ppo = PPO()
    ppo.train()

