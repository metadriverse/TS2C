import pickle
import numpy as np
from egpo_utils.common import expert_action_prob, evaluation_config
from egpo_utils.expert_guided_env import ExpertGuidedEnv, load_weights

def compress_model(ckpt_path, remove_value_network=False):
	with open(ckpt_path, "rb") as f:
		data = f.read()
	unpickled = pickle.loads(data)
	worker = pickle.loads(unpickled.pop("worker"))
	weights = worker["state"]["default_policy"]
	if remove_value_network:
		weights = {k: v for k, v in weights.items() if "value" not in k}
	return weights

def eval(env, weights, algo="sac"):
	o = env.reset()
	epi_num = 0

	total_cost = 0
	total_reward = 0
	success_rate = 0
	ep_cost = 0
	ep_reward = 0
	success_flag = False
	horizon = 2000
	step = 0
	EPISODE_NUM=100
	while True:
		# action_to_send = compute_actions(w, [o], deterministic=False)[0]
		step += 1
		if len(weights) > 1:
			action_to_send = np.array([expert_action_prob(None, o, weight, deterministic=False, algo=algo)[0] for weight in weights])
			# print(action_to_send)
			action_to_send = np.average(action_to_send, axis=0)
			# print(action_to_send)
			# print()
		else:
			action_to_send =expert_action_prob(None, o, weights[0], deterministic=False, algo=algo)[0]
		o, r, d, info = env.step(action_to_send)
		total_reward += r
		ep_reward += r
		total_cost += info["cost"]
		ep_cost += info["cost"]
		if d or step > horizon:
			if info["arrive_dest"]:
				success_rate += 1
				success_flag = True
			epi_num += 1
			if epi_num % 30 == 0:
				print(epi_num)
			if epi_num > EPISODE_NUM:
				break
			else:
				o = env.reset()

			# super_data[ckpt].append({"reward": ep_reward, "success": success_flag, "cost": ep_cost})

			ep_cost = 0.0
			ep_reward = 0.0
			success_flag = False
			step = 0

	print(
		"success_rate:{}, mean_episode_reward:{}, mean_episode_cost:{}".format(success_rate / EPISODE_NUM,
																							total_reward / EPISODE_NUM,
																							total_cost / EPISODE_NUM))

paths = [
	# "./ckpts/stu_r1/EGPOTrainer_ExpertGuidedEnv_108f4_00000_0_seed=0_2021-12-08_22-49-37/checkpoint_40/checkpoint-40",
	# "./ckpts/stu_r1/EGPOTrainer_ExpertGuidedEnv_108f4_00001_1_seed=100_2021-12-08_22-49-37/checkpoint_70/checkpoint-70",
	# "./ckpts/stu_r1/EGPOTrainer_ExpertGuidedEnv_108f4_00001_1_seed=100_2021-12-08_22-49-37/checkpoint_80/checkpoint-80",
	# "./ckpts/stu_r1/EGPOTrainer_ExpertGuidedEnv_108f4_00002_2_seed=200_2021-12-08_22-49-37/checkpoint_60/checkpoint-60",
	# "./ckpts/stu_r2/EGPOTrainer_ExpertGuidedEnv_37612_00001_1_seed=100_2021-12-16_01-05-34-3/checkpoint_60/checkpoint-60",
	# "./ckpts/stu_r2/EGPOTrainer_ExpertGuidedEnv_37612_00001_1_seed=100_2021-12-16_01-05-34-3/checkpoint_70/checkpoint-70",
	# "./ckpts/stu_r2/EGPOTrainer_ExpertGuidedEnv_37612_00001_1_seed=100_2021-12-16_01-05-34-3/checkpoint_80/checkpoint-80",
	# "./ckpts/stu_r2/r3_checkpoint_30/checkpoint-30"
    # "./ckpts/checkpoint_90/checkpoint-90"
	# "../SAC_baseline/ensembleQ_ExpertGuidedEnv_55666_00001_1_seed=100_2021-12-21_11-51-52/checkpoint_210/checkpoint-210",
	# "../SAC_baseline/ensembleQ_ExpertGuidedEnv_55666_00001_1_seed=100_2021-12-21_11-51-52/checkpoint_220/checkpoint-220",
	# "../SAC_baseline/ensembleQ_ExpertGuidedEnv_55666_00001_1_seed=100_2021-12-21_11-51-52/checkpoint_230/checkpoint-230",
	# "../SAC_baseline/ensembleQ_ExpertGuidedEnv_55666_00001_1_seed=100_2021-12-21_11-51-52/checkpoint_240/checkpoint-240",
	# "../SAC_baseline/ensembleQ_ExpertGuidedEnv_55666_00001_1_seed=100_2021-12-21_11-51-52/checkpoint_250/checkpoint-250",
	# "../../results/sub-sac/140/EGPOTrainer_ExpertGuidedEnv_fd1ba_00002_2_seed=200_2021-12-22_10-50-57/checkpoint_{}/checkpoint-{}" \
		# .format(i, i) for i in range(120, 200, 10)
	"../../egpo_utils/saved_expert/BC/130.npz"
]
print(paths)
env = ExpertGuidedEnv(evaluation_config["env_config"])
# weights = [compress_model(path) for path in paths]
weights = [load_weights(path) for path in paths]
eval(env, weights, "bc")
