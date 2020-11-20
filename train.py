import ppo
import gym
from prestart_env import GameEnv

gym.envs.registration.register(
    id="prestart-v0", 
    entry_point=GameEnv
)
env = lambda: gym.make("prestart-v0")
kwargs = dict(hidden_sizes=(128,256,256,256,128))
ppo.ppo(env, ac_kwargs=kwargs, epochs=50, steps_per_epoch=20000, pi_lr=1e-4, vf_lr=1e-4, train_pi_iters=10, train_v_iters=10)

