import ppo
import gym
from prestart_env import GameEnv

gym.envs.registration.register(
    id="prestart-v0", 
    entry_point=GameEnv
)
env = lambda: gym.make("prestart-v0")

ppo.ppo(env)

