import time
import torch
import torch.nn as nn
from torch.distributions.normal import Normal
from cy_ppo import env_reset, update_state_and_buffers
from config import *
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

        # setup actor critic nn models
        self.actor = Actor(state_size, action_size, 256).cuda()
        self.critic = Critic(state_size, 256).cuda()

        # setup gamm lambda arrays to speed up calculation of advantage estimates
        self.gamma_arr = np.zeros(max_steps, dtype=np.float32)
        self.gamma_lam_arr = np.zeros(max_steps, dtype=np.float32)
        for i in range(max_steps):
            self.gamma_arr[i] = pow(gamma, i)
            self.gamma_lam_arr[i] = pow(gamma * lam, i)

        # setup optimisers
        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr)
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr)

    def act(self, env_state):
        state = torch.from_numpy(env_state).cuda()
        mu = self.actor(state)
        pi = self.actor.dist(mu)
        action = pi.sample()
        log_prob = self.actor.log_prob(pi, action)
        value = self.critic(state)
        action = action.cpu().numpy()
        log_prob = log_prob.cpu().detach().numpy()
        value = value.cpu().detach().numpy()
        return action, log_prob, value

    def get_batch(self):

        # setup data buffers for the batch of episodes
        batch_states = np.zeros((batch_size, state_size), dtype=np.float32)
        batch_actions = np.zeros((batch_size, action_size), dtype=np.float32)
        batch_log_probs = np.zeros(batch_size, dtype=np.float32)
        batch_rewards_to_go = np.zeros(batch_size, dtype=np.float32)
        batch_advantages = np.zeros(batch_size, dtype=np.float32)
        batch_values = np.zeros(batch_size, dtype=np.float32)

        # setup data buffers for the indiviual episodes
        episode_states = np.zeros((n_parallel, max_steps, state_size), dtype=np.float32)
        episode_actions = np.zeros((n_parallel, max_steps, action_size), dtype=np.float32)
        episode_log_probs = np.zeros((n_parallel, max_steps), dtype=np.float32)
        episode_values = np.zeros((n_parallel, max_steps + 1), dtype=np.float32)
        episode_rewards = np.zeros((n_parallel, max_steps), dtype=np.float32)
        episode_rewards_to_go = np.zeros((n_parallel, max_steps), dtype=np.float32)
        episode_advantages = np.zeros((n_parallel, max_steps), dtype=np.float32)
        idxs = np.zeros(n_parallel, dtype=np.int32)
        states = np.zeros((n_parallel, state_size), dtype=np.float32)
        rewards = np.zeros(n_parallel, dtype=np.float32)
        dones = np.zeros(n_parallel, dtype=np.int32)
        delta = np.zeros(max_steps, dtype=np.float32)

        # reset the states
        for i in range(n_parallel):
            env_reset(states[i,:])

        # collect a batch of data
        n = 0
        while n < batch_size:

            # normalise the states and add to the episode buffer of states
            # N.B. not needed for this env
            
            # get the actions, log_prob and value estimates
            actions, log_probs, values = self.act(states)

            # update the state and buffers
            n = update_state_and_buffers(
                states, 
                actions, 
                rewards, 
                dones, 
                values,
                log_probs,
                episode_states, 
                episode_actions, 
                episode_values, 
                episode_log_probs, 
                episode_rewards,
                episode_rewards_to_go,
                episode_advantages,
                batch_states, 
                batch_actions, 
                batch_values, 
                batch_log_probs, 
                batch_rewards_to_go,
                batch_advantages,
                delta,
                gamma,
                self.gamma_arr, 
                self.gamma_lam_arr, 
                idxs, 
                n
            )
            
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
        clip_advantage = torch.clamp(ratio, 1-epsilon, 1+epsilon) * advantages
        loss = -(torch.min(ratio * advantages, clip_advantage)).mean()
        return loss

    def critic_loss(self, states, rewards_to_go, values):
        new_values = self.critic(states)[:,0]
        v_clip = values + torch.clamp(new_values - values, -epsilon, epsilon)
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
        for i in range(n_actor_updates):
            self.actor_opt.zero_grad()
            loss_pi = self.actor_loss(states, actions, log_probs, advantages)
            loss_pi.backward()
            self.actor_opt.step()

        # update the critic
        for i in range(n_critic_updates):
            self.critic_opt.zero_grad()
            loss_v = self.critic_loss(states, rewards_to_go, values)
            loss_v.backward()
            self.critic_opt.step()

        return

    def train(self):
        
        updates_per_checkpoint = int(1e7 / batch_size)
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

