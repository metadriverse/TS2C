import mujoco_py
import gym
import numpy as np

def relu(x):
    return (np.abs(x) + x) / 2


def expert_action_prob(obs, weights, deterministic=False, algo="ppo"):
    obs = obs.reshape(1, -1)
    if algo == "ppo":
        x = np.matmul(obs, weights["default_policy/fc_1/kernel"]) + weights["default_policy/fc_1/bias"]
        x = np.tanh(x)
        x = np.matmul(x, weights["default_policy/fc_2/kernel"]) + weights["default_policy/fc_2/bias"]
        x = np.tanh(x)
        x = np.matmul(x, weights["default_policy/fc_out/kernel"]) + weights["default_policy/fc_out/bias"]
    elif algo == "sac":
        x = np.matmul(obs, weights["default_policy/sequential/action_1/kernel"]) + weights["default_policy/sequential/action_1/bias"]
        x = relu(x)
        x = np.matmul(x, weights["default_policy/sequential/action_2/kernel"]) + weights["default_policy/sequential/action_2/bias"]
        x = relu(x)
        x = np.matmul(x, weights["default_policy/sequential/action_out/kernel"]) + weights["default_policy/sequential/action_out/bias"]
        x = np.tanh(x)
    x = x.reshape(-1)
    mean, log_std = np.split(x, 2)
    std = np.exp(log_std)
    expert_action = np.random.normal(mean, std) if not deterministic else mean
    return expert_action

env = gym.make("Hopper-v2")
env.render()
o = env.reset()
weight = np.load("hopper_egpo_weights/160.npz")
r_tot = 0
while True:
	a = expert_action_prob(o, weight, algo='sac', deterministic=True)
	o, r, d, _ = env.step(a)
	r_tot += r
	env.render()
	if d:
		print(r_tot)
		r_tot = 0
		o = env.reset()
    