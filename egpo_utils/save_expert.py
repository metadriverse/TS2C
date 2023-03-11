import pickle
import math
import numpy as np


def compress_model(ckpt_path, path="safe_expert.npz", remove_value_network=False):
    with open(ckpt_path, "rb") as f:
        data = f.read()
    unpickled = pickle.loads(data)
    worker = pickle.loads(unpickled.pop("worker"))
    if "_optimizer_variables" in worker["state"]["default_policy"]:
        worker["state"]["default_policy"].pop("_optimizer_variables")
    pickled_worker = pickle.dumps(worker)
    weights = worker["state"]["default_policy"]
    for i in weights.keys():
        print(i)
        print(weights[i].shape)
    if remove_value_network:
        weights = {k: v for k, v in weights.items() if "value" not in k}
    np.savez_compressed(path, **weights)
    print("Numpy agent weight is saved at: {}!".format(path))

if __name__ == "__main__":
    ckpt = "../training_script/PPO/PPO_Walker2dTS2CEnv_0bd33_00000_0_seed=0_2022-11-16_15-26-54/checkpoint_1000/checkpoint-1000"
    compress_model(ckpt, path="saved_expert/mujoco/walker/ppo1000.npz")
