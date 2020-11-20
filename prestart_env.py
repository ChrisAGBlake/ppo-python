from gym import spaces
import numpy as np
import gym_cython
import gym
import random


class GameEnv(gym.Env):
    metadata = {"render.modes": ["human"]}

    def __init__(self, args=None):
        if args is None:
            args = {}
        self.timestep = 1.0
        self.prestart_duration = 120.0
        self.randomise_line = False
        self.line_len = 250.0
        self.line_skew = 0.0
        self.entry_side = True
        self.tws = 12.0 * 1852.0 / 3600.0
        self.action_space = spaces.Box(
            np.array([-1, 0]), np.array([1, 1]), dtype=np.float32
        )
        self.actions = np.zeros(4, dtype=np.float32)
        self.observation_space = spaces.Box(
            np.array([-np.inf] * 23),
            np.array([np.inf] * 23),
            dtype=np.float32,
        )
        self.state = np.zeros(23, dtype=np.float32)
        self.norm_state = np.zeros(23, dtype=np.float32)
        self.episode_buffer = np.zeros(
            (
                int(
                    (self.prestart_duration + 20.0)
                    / self.timestep
                    + 1
                ),
                23,
            ),
            dtype=np.float32,
        )
        self.row_buffer = np.zeros(int(2 / self.timestep), dtype=np.int32)
        self.step_count = 0
        self.reset()

    def get_norm_state(self):
        self.norm_state[:] = self.state[:]
        gym_cython.normalise(self.norm_state, self.prestart_duration)
        return self.norm_state

    def step(self, action):
        
        # copy the previous state into the buffer
        self.episode_buffer[self.step_count, :] = self.state[:]
        self.step_count += 1

        # set the action
        self.actions[:2] = action
        self.actions[2] = random.random() * 2 - 1

        reward, done, info, row = gym_cython.step(
            self.state,
            self.state.copy(),
            self.actions,
            self.row_buffer[-1],
            self.timestep,
            self.prestart_duration,
            self.tws
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
        self.get_norm_state()

        return (
            self.norm_state,
            reward,
            done,
            info,
        )

    def reset(self, reset_timestamp=None):
        if reset_timestamp is not None:
            self.step_count = int(reset_timestamp / self.timestep)
            self.state[:] = self.episode_buffer[self.step_count, :]
            self.tws = self.state[22]
        else:
            if self.randomise_line:
                self.line_len = (
                    200
                    + (random.random() - 0.5) * 100
                )
                self.line_skew = (
                    0
                    + (random.random() - 0.5) * 0.5
                )
            gym_cython.game_reset(
                self.state, self.line_len, self.line_skew, self.entry_side
            )
            self.step_count = 0
            self.state[22] = self.tws

        # reset the right of way buffer
        for i in range(self.row_buffer.shape[0]):
            self.row_buffer[i] = self.state[19]

        # normalise the state
        self.get_norm_state()

        return self.norm_state

    def render(self, mode="human", close=False):
        pass