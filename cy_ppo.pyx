from libc.stdlib cimport rand
cdef extern from "stdlib.h":
    int RAND_MAX
import config as cfg
import numpy as np

cdef int n_parallel = cfg.n_parallel
cdef int state_size = cfg.state_size
cdef int action_size = cfg.action_size
cdef int batch_size = cfg.batch_size
cdef int max_steps = cfg.max_steps

cdef float rand_float():
    cdef float f = <float>rand() / <float>RAND_MAX - 0.5
    return f

def env_reset(float[::] state):
    cdef int i
    for i in range(state_size):
        state[i] = rand_float()
    return

cdef env_step(float[:,::] states, float[:,::] actions, float[::] rewards, int[::] dones):
    cdef int i, j
    for i in range(n_parallel):
        for j in range(state_size):
            states[i,j] = rand_float()
        rewards[i] = rand_float()
        dones[i] = 0
    return

def update_state_and_buffers(
    float[:,::] states, 
    float[:,::] actions, 
    float[::] rewards, 
    int[::] dones, 
    float[:,::] values,
    float[::] log_probs,
    float[:,:,::] episode_states, 
    float[:,:,::] episode_actions, 
    float[:,::] episode_values, 
    float[:,::] episode_log_probs, 
    float[:,::] episode_rewards,
    float[:,::] episode_rewards_to_go,
    float[:,::] episode_advantages,
    float[:,::] batch_states, 
    float[:,::] batch_actions, 
    float[::] batch_values, 
    float[::] batch_log_probs, 
    float[::] batch_rewards_to_go,
    float[::] batch_advantages,
    float[::] delta,
    float gamma,
    float[::] gamma_arr, 
    float[::] gamma_lam_arr, 
    int[::] idxs, 
    int n):
    cdef int i, j, k, s, sz
    cdef float v

    # add to the episode buffers
    for i in range(n_parallel):
        episode_states[i, idxs[i], :] = states[i, :]
        episode_actions[i, idxs[i], :] = actions[i, :]
        episode_values[i, idxs[i]] = values[i, 0]
        episode_log_probs[i, idxs[i]] = log_probs[i]

    # update the states
    env_step(states, actions, rewards, dones)
    for i in range(n_parallel):

        # add the rewards to the episode buffer
        episode_rewards[i, idxs[i]] = rewards[i]

        # check for the episode happening
        if dones[i] > 0 or idxs[i] == max_steps - 1:
            # add a final 0 value
            sz = idxs[i] + 1
            episode_values[i, sz] = 0

            # calculate rewards to go
            for j in range(sz):
                # episode_rewards_to_go[i, j] = np.sum(episode_rewards[i, j:sz] * gamma_arr[:sz-j])
                v = 0
                for k in range(sz-j):
                    v += episode_rewards[i, j+k] * gamma_arr[k]
                episode_rewards_to_go[i,j] = v

            # calculate the advantage estimates
            # delta = episode_rewards[i, :sz] + gamma * episode_values[i, 1:sz+1] - episode_values[i, :sz]
            # for j in range(sz):
            #     episode_advantages[i, j] = np.sum(delta[j:sz] * gamma_lam_arr[:sz-j])
            for j in range(sz):
                delta[j] = episode_rewards[i, j] + gamma * episode_values[i, j+1] - episode_values[i, j]
            for j in range(sz):
                v = 0
                for k in range(sz-j):
                    v += delta[j+k] * gamma_lam_arr[k]
                episode_advantages[i, j] = v

            # update the buffers
            s = n
            sz = min(sz, batch_size - n)
            n += sz
            batch_states[s:n, :] = episode_states[i, :sz, :]
            batch_actions[s:n, :] = episode_actions[i, :sz, :]
            batch_log_probs[s:n] = episode_log_probs[i, :sz]
            batch_rewards_to_go[s:n] = episode_rewards_to_go[i, :sz]
            batch_advantages[s:n] = episode_advantages[i, :sz]
            batch_values[s:n] = episode_values[i, :sz]

            # reset the episode index
            idxs[i] = 0

            # reset this state
            env_reset(states[i,:])

        else:
            idxs[i] += 1

    return n