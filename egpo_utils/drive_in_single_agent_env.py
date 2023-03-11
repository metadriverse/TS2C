"""
Please feel free to run this script to enjoy a journey by keyboard!
Remember to press H to see help message!

Note: This script require rendering, please following the installation instruction to setup a proper
environment that allows popping up an window.
"""
import argparse
import random

import numpy as np

from metadrive import MetaDriveEnv, Sa
from metadrive.constants import HELP_MESSAGE
import math


def normpdf(x, mean, sd):
    var = float(sd) ** 2
    denom = (2 * math.pi * var) ** .5
    num = math.exp(-(float(x) - float(mean)) ** 2 / (2 * var + 1e-6))
    return num / (denom + 1e-6)

def relu(x):
    return (np.abs(x) + x) / 2

def load_weights(path: str):
    """
    Load NN weights
    :param path: weights file path path
    :return: NN weights object
    """
    # try:
    model = np.load(path)
    return model

def expert_action_prob(action, obs, weights, deterministic=False, algo="ppo", std_multi=0):
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
    elif algo == "bc":
        x = np.matmul(obs, weights["action_model.action_0._model.0.weight"].T) + weights["action_model.action_0._model.0.bias"].T
        x = relu(x)
        x = np.matmul(x, weights["action_model.action_1._model.0.weight"].T) + weights["action_model.action_1._model.0.bias"].T
        x = relu(x)
        x = np.matmul(x, weights["action_model.action_out._model.0.weight"].T) + weights["action_model.action_out._model.0.bias"]
        x = np.tanh(x)
    else:
        assert False
    x = x.reshape(-1)
    mean, log_std = np.split(x, 2)
    std = np.exp(log_std) + std_multi
    if action is not None:
        a_0_p = normpdf(action[0], mean[0], std[0])
        a_1_p = normpdf(action[1], mean[1], std[1])
    else:
        a_0_p, a_1_p = 0, 0
    expert_action = np.random.normal(mean, std) if not deterministic else mean
    return expert_action, a_0_p, a_1_p

weights = load_weights(path)
algo = "sac"

if __name__ == "__main__":
    config = dict(
        # controller="joystick",
        use_render=True,
        manual_control=False,
        traffic_density=0.1,
        environment_num=100,
        random_agent_model=True,
        random_lane_width=True,
        random_lane_num=True,
        show_skybox=True, 
        map=4,  # seven block
        start_seed=random.randint(0, 1000)
    )
    parser = argparse.ArgumentParser()
    parser.add_argument("--observation", type=str, default="lidar", choices=["lidar", "rgb_camera"])
    args = parser.parse_args()
    if args.observation == "rgb_camera":
        config.update(dict(offscreen_render=True))
    env = MetaDriveEnv(config)
    # try:
    o = env.reset()
    print(HELP_MESSAGE)
    env.vehicle.expert_takeover = True
    if args.observation == "rgb_camera":
        action = expert_action_prob(None, o, weights, True, algo=algo)
        assert isinstance(o, dict)
        print("The observation is a dict with numpy arrays as values: ", {k: v.shape for k, v in o.items()})
    else:
        assert isinstance(o, np.ndarray)
        print("The observation is an numpy array with shape: ", o.shape)
    for i in range(1, 1000000000):
        o, r, d, info = env.step([0, 0])
        env.render(
            text={
                "Auto-Drive (Switch mode: T)": "on" if env.current_track_vehicle.expert_takeover else "off",
            }
        )
        if d and info["arrive_dest"]:
            env.reset()
            env.current_track_vehicle.expert_takeover = True
    # except:
    #     pass
    # finally:
    #     env.close()
