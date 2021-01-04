from libc.stdlib cimport rand
cdef extern from "stdlib.h":
    int RAND_MAX
import config as cfg

cdef int n_parallel = cfg.n_parallel
cdef int state_size = cfg.state_size
cdef int action_size = cfg.action_size
cdef int batch_size = cfg.batch_size
cdef int max_steps = cfg.max_steps

cdef float rand_float():
    cdef float f = <float>rand() / <float>RAND_MAX - 0.5
    return f

cpdef env_reset(float[::1] state):
    cdef int i
    for i in range(state_size):
        state[i] = rand_float()
    return

cdef env_step(float[:,::1] states, float[:,::1] actions, float[::1] rewards, int[::1] dones):
    cdef int i, j
    for i in range(n_parallel):
        for j in range(state_size):
            states[i,j] = rand_float()
        rewards[i] = rand_float()
        dones[i] = 0
    return

def update_state_and_buffers(
    float[:,::1] states, 
    float[:,::1] actions, 
    float[::1] rewards, 
    int[::1] dones, 
    float[:,::1] values,
    float[::1] log_probs,
    float[:,:,::1] episode_states, 
    float[:,:,::1] episode_actions, 
    float[:,::1] episode_values, 
    float[:,::1] episode_log_probs, 
    float[:,::1] episode_rewards,
    float[:,::1] episode_rewards_to_go,
    float[:,::1] episode_advantages,
    float[:,::1] batch_states, 
    float[:,::1] batch_actions, 
    float[::1] batch_values, 
    float[::1] batch_log_probs, 
    float[::1] batch_rewards_to_go,
    float[::1] batch_advantages,
    float[::1] delta,
    float gamma,
    float[::1] gamma_arr, 
    float[::1] gamma_lam_arr, 
    int[::1] idxs, 
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
                v = 0
                for k in range(sz-j):
                    v += episode_rewards[i, j+k] * gamma_arr[k]
                episode_rewards_to_go[i,j] = v

            # calculate the advantage estimates
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