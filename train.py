import ppo
import gym
from prestart_env import GameEnv

gym.envs.registration.register(
    id="prestart-v0", 
    entry_point=GameEnv
)
env = lambda: gym.make("prestart-v0")
kwargs = dict(hidden_sizes=(256,512,512,256))
ppo.ppo(env, ac_kwargs=kwargs, epochs=3, steps_per_epoch=20000, pi_lr=3e-4, vf_lr=3e-4, train_pi_iters=10, train_v_iters=10)

# in this configuration the julia version with no multithreading takes around 10s for the episode collection and 0.9s for the update (using the GPU)
# the current version using spinningup and cython based env takes around 30s for episode collection and 15s for the update (not using the GPU)
# from profiling the code, all the time is spent inside pytorch functions and can't be optimised much.
# it looks like the julia version is a much better option.