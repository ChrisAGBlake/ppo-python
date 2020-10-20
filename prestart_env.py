import numpy as np
import gym_cython

class Env():
    def __init__(self):
        self.state_size = 23
        self.action_size = 2
        self.max_steps = 150
        self.timestep = 1.0
        self.state = np.zeros(self.state_size, dtype=np.float32)
        self.norm_state = np.zeros(self.state_size, dtype=np.float32)
        self.row_buffer = np.zeros(int(2 / self.timestep), dtype=np.float32)

    def step(self, actions):

        reward, done, info, row = gym_cython.step(
            self.state,
            self.state.copy(),
            actions,
            self.row_buffer[-1],
            self.timestep,
            120.0,
            1,
            6.5,
        )
        self.state[21] = self.row_buffer[-1]
        n = int(1 / self.timestep) - 1
        self.state[20] = self.row_buffer[n]
        self.state[19] = row

        # update the row buffer
        for i in reversed(range(self.row_buffer.shape[0] - 1)):
            self.row_buffer[i + 1] = self.row_buffer[i]
        self.row_buffer[0] = row

        # normalise the state
        self._normalise()
        return reward, done

    def reset(self):
        gym_cython.reset(self.state, 250.0, 0.0, True)
        for i in range(self.row_buffer.shape[0]):
            self.row_buffer[i] = self.state[19]

        # normalise the state
        self._normalise()
        return

    def _normalise(self):
        self.norm_state[:] = self.state[:]
        gym_cython.normalise(self.norm_state, 120.0)
        return