from libc.stdlib cimport rand
cdef extern from "stdlib.h":
    int RAND_MAX
import config as cfg

cdef int n_parallel = cfg.n_parallel
cdef int state_size = cfg.state_size
cdef int action_size = cfg.action_size

cdef float rand_float():
    cdef float f = <float>rand() / <float>RAND_MAX - 0.5
    return f

def env_reset(float[::] state):
    cdef int i
    for i in range(state_size):
        state[i] = rand_float()
    return

def env_step(float[:,::] states, float[:,::] actions, float[::] rewards, int[::] dones):
    cdef int i, j
    for i in range(n_parallel):
        for j in range(state_size):
            states[i,j] = rand_float()
        rewards[i] = rand_float()
        dones[i] = 0
    return
