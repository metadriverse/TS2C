from panda3d.core import PNMImage 
import pygame
from numpy.random import randn
import os.path as osp
from os import scandir

import gym
import numpy as np
from metadrive.envs.safe_metadrive_env import SafeMetaDriveEnv
from metadrive.utils.config import Config
from egpo_utils.common import expert_action_prob, ExpertObservation, expert_q_value, ensemble_q_value
import math
from egpo_utils.save_ppo_expert import compress_model

def normpdf(x, mean, sd):
    var = float(sd) ** 2
    denom = (2 * math.pi * var) ** .5
    num = math.exp(-(float(x) - float(mean)) ** 2 / (2 * var))
    return num / denom


def load_weights(path: str):
    """
    Load NN weights
    :param path: weights file path path
    :return: NN weights object
    """
    # try:
    model = np.load(path)
    return model
    # except FileNotFoundError:
    # print("Can not find {}, didn't load anything".format(path))
    # return None


class ExpertGuidedEnv(SafeMetaDriveEnv):

    steps = 0

    def default_config(self) -> Config:
        """
        Train/Test set both contain 10 maps
        :return: PGConfig
        """
        config = super(ExpertGuidedEnv, self).default_config()
        config.update(dict(
            environment_num=100,
            start_seed=100,
            safe_rl_env_v2=False,  # If True, then DO NOT done even out of the road!
            # _disable_detector_mask=True,  # default False to acc Lidar detection

            # traffic setting
            random_traffic=False,
            # traffic_density=0.1,

            # special setting
            rule_takeover=False,
            value_takeover=False,
            dagger_takeover=False,
            maxmin_takeover=False,
            uncertainty_takeover=False,
            ensemble=False,
            value_takeover_threshold=1.5,
            var_threshold=2.0,
            expert_policy_type="ppo",
            old_obs=False,
            warmup_ts=10000,
            warmup_noise=0.5,
            takeover_cost=1,
            eval=False,
            cost_info="native",  # or takeover
            random_spawn=False,  # used to collect dataset
            cost_to_reward=True,  # for egpo, it accesses the ENV reward by penalty
            horizon=1000,

            crash_vehicle_penalty=1.,
            crash_object_penalty=0.5,
            out_of_road_penalty=1.,
            update_value_freq=1000,
            exp_path=None,

            vehicle_config=dict(  # saver config, free_level = expert
                use_saver=False,
                free_level=100,
                expert_deterministic=False,
                release_threshold=100,  # the save will be released when level < this threshold
                overtake_stat=False),  # set to True only when evaluate

            expert_policy_weights=osp.join(osp.dirname(__file__), "expert.npz"),
            expert_value_weights="default",
            value_fn_path=None,
            value_from_scratch=False,
        ), allow_add_new_key=True)
        return config

    def __init__(self, config):
        # if ("safe_rl_env" in config) and (not config["safe_rl_env"]):
        #     raise ValueError("You should always set safe_rl_env to True!")
        # config["safe_rl_env"] = True
        if config.get("safe_rl_env_v2", False):
            config["out_of_road_penalty"] = 0
        super(ExpertGuidedEnv, self).__init__(config)
        self.expert_observation = ExpertObservation(self.config["vehicle_config"])
        assert self.config["expert_policy_weights"] is not None
        self.total_takeover_cost = 0
        self.total_native_cost = 0
        self.state_value = 0
        self.expert_policy_weights = load_weights(self.config["expert_policy_weights"])
        self.expert_weights = self.expert_policy_weights
        value_weight_path = osp.join(osp.dirname(__file__), 
                                               "saved_expert", 
                                               "ensemble_sac_expert_350.npz") \
                            if self.config["expert_value_weights"] == "default" \
                            else self.config["expert_value_weights"]
        self.expert_value_weights = load_weights(value_weight_path) if value_weight_path is not None else None
        self.old_obs = self.config["old_obs"]
        self.ensemble = self.config["ensemble"]
        if self.config["cost_to_reward"]:
            self.config["out_of_road_penalty"] = self.config["out_of_road_cost"]
            self.config["crash_vehicle_penalty"] = self.config["crash_vehicle_cost"]
            self.config["crash_object_penalty"] = self.config["crash_object_cost"]
        self.latest_idx = 0 # for loading latest value weights
        self.max_idx_for_value = 50

    def expert_observe(self):
        return self.expert_observation.observe(self.vehicle)

    def get_expert_action(self, v_id="default_agent", deterministic=False, std_multi=0):
        if self.old_obs:
            # ppo policy with specific expert observation
            obs = self.expert_observation.observe(self.vehicle)
        else:
            obs = self.observations[v_id].observe(self.vehicle)

        return expert_action_prob([0, 0], obs, self.expert_policy_weights, std_multi=std_multi,
                                  deterministic=deterministic, algo=self.config["expert_policy_type"])[0]

    def get_q_value(self, v_id, policy="expert", action=None, twin=False, pessimistic=False, ensemble=False):
        obs = self.observations[v_id].observe(self.vehicle)
        # currently use expert weight
        weight = self.expert_value_weights
        if policy=="expert":
            action = self.get_expert_action()
        else:
            assert action is not None
        if ensemble:
            return ensemble_q_value(action, obs, weight, twin, pessimistic)
        return expert_q_value(action, obs, weight, twin, pessimistic)

    def _get_reset_return(self):
        assert self.num_agents == 1
        self.total_takeover_cost = 0
        self.total_native_cost = 0
        if self.config["vehicle_config"]["free_level"] < 1e-3:
            # 1.0 full takeover
            self.vehicle.takeover_start = True
        return super(ExpertGuidedEnv, self)._get_reset_return()

    def step(self, actions):
        if not self.config["eval"]:
            if ExpertGuidedEnv.steps % self.config["update_value_freq"] == 0 and self.config["value_from_scratch"]:
                self.load_latest_value_weights()
        actions, saver_info = self.expert_takeover("default_agent", actions)
        obs, r, d, info, = super(ExpertGuidedEnv, self).step(actions)
        saver_info.update(info)
        info = self.extra_step_info(saver_info)
        if not self.config["eval"]:
            ExpertGuidedEnv.steps += 1
        return obs, r, d, info

    def extra_step_info(self, step_info):
        # step_info = step_infos[self.DEFAULT_AGENT]

        step_info["native_cost"] = step_info["cost"]
        # if step_info["out_of_road"] and not step_info["arrive_dest"]:
        # out of road will be done now
        step_info["high_speed"] = True if self.vehicle.speed >= 50 else False
        step_info["takeover_cost"] = self.config["takeover_cost"] if step_info["takeover_start"] else 0
        self.total_takeover_cost += step_info["takeover_cost"]
        self.total_native_cost += step_info["native_cost"]
        step_info["total_takeover_cost"] = self.total_takeover_cost
        step_info["total_native_cost"] = self.total_native_cost

        if self.config["cost_info"] == "native":
            step_info["cost"] = step_info["native_cost"]
            step_info["total_cost"] = self.total_native_cost
        elif self.config["cost_info"] == "takeover":
            step_info["cost"] = step_info["takeover_cost"]
            step_info["total_cost"] = self.total_takeover_cost
        else:
            raise ValueError
        return step_info

    def done_function(self, v_id):
        """This function is a little bit different compared to the SafePGDriveEnv in PGDrive!"""
        done, done_info = super(ExpertGuidedEnv, self).done_function(v_id)
        if self.config["safe_rl_env_v2"]:
            assert self.config["out_of_road_cost"] > 0
            if done_info["out_of_road"]:
                done = False
        return done, done_info

    def _is_out_of_road(self, vehicle):
        return vehicle.out_of_route

    def load_latest_value_weights(self):
        if self.config["value_fn_path"] is not None:
            self.expert_value_weights = compress_model(self.config["value_fn_path"])
        else:
            exp_dir = self.config["exp_path"]
            subfolders = [ f.path for f in scandir(exp_dir) if f.is_dir() ]
            # to find exactly the exp dir, number of seeds should be set to 1, i.e. only one dir in exp dir
            assert len(subfolders) == 1
            ckpt_dir = subfolders[0]
            ckpt_idx = [int(i.split("_")[-1]) for i in [f.path for f in scandir(ckpt_dir) if f.is_dir() ]]
            if len(ckpt_idx) == 0:
                return
            latest_idx = max(ckpt_idx)
            if self.max_idx_for_value > latest_idx >= self.latest_idx:
                self.latest_idx = latest_idx
                self.expert_value_weights = compress_model(osp.join(exp_dir, ckpt_dir, "checkpoint_%d"%latest_idx, "checkpoint-%d"%latest_idx))

    def value_takeover(self, v_id, actions):
        """
        Takeover if V_agent_policy is far lower than V_expert_policy
        """
        vehicle = self.vehicles[v_id]
        action = actions
        steering = action[0]
        throttle = action[1]
        threshold = self.config["value_takeover_threshold"]
        takeover = False
        warmup_ts = self.config["warmup_ts"]
        if ExpertGuidedEnv.steps < warmup_ts and not self.config["eval"]:
            takeover = True
            warmup = True
        else:
            warmup = False
            if vehicle.config["use_saver"] or vehicle.expert_takeover:
                # saver can be used for human or another AI
                if not self.ensemble:
                    expert_value = self.get_q_value(v_id)
                    agent_value = self.get_q_value(v_id, policy="agent", action=action, pessimistic=True)
                    if agent_value < expert_value - threshold:
                        takeover = True
                else:
                    if self.config["maxmin_takeover"]:
                        sampled_actions = [self.get_expert_action() for _ in range(10)]
                        q_values = [self.get_q_value(v_id, action=action, pessimistic=False) for action in sampled_actions]
                        maxmin_diff = np.max(q_values) - np.min(q_values)
                        if maxmin_diff > threshold:
                            takeover = True
                    elif self.config["uncertainty_takeover"]:
                        expert_ensemble_values = self.get_q_value(v_id, ensemble=True)
                        expert_var = np.var(expert_ensemble_values, action=action)
                        if expert_var > self.config["var_threshold"]:
                            takeover = True 
                    else:
                        expert_ensemble_values = self.get_q_value(v_id, ensemble=True)
                        expert_var = np.var(expert_ensemble_values)
                        agent_ensemble_values = self.get_q_value(v_id, policy="agent", action=action, ensemble=True)
                        agent_var = np.var(agent_ensemble_values)
                        diff = np.array(expert_ensemble_values) - np.array(agent_ensemble_values)
                        diff_mean, diff_var = np.average(diff), np.var(diff)
                        if diff_mean > threshold or agent_var > 2 * self.config["var_threshold"]:
                            takeover = True

                    # print(expert_ensemble_values)
                    # print("expert variance: ", np.var(expert_ensemble_values))
                    # print(agent_ensemble_values)
                    # print("agent variance: ", np.var(agent_ensemble_values))
                    # print(diff)
                    # print("diff variance: ", np.var(diff))
                    # print()
        if takeover:
            expert_action = self.get_expert_action(deterministic=vehicle.config["expert_deterministic"]) + \
                            self.config["warmup_noise"]* randn(2)
            steering = expert_action[0]
            throttle = expert_action[1]

        # indicate if current frame is takeover step
        pre_save = vehicle.takeover
        vehicle.takeover = takeover
        saver_info = {
            "takeover_start": True if not pre_save and vehicle.takeover else False,
            "takeover_end": True if pre_save and not vehicle.takeover else False,
            "takeover": vehicle.takeover if pre_save else False,
            "warmup": warmup,
        }
        return (steering, throttle) if saver_info["takeover"] else action, saver_info

    def expert_takeover(self, v_id: str, actions):
        """
        Action prob takeover
        """
        if self.config["rule_takeover"]:
            return self.rule_takeover(v_id, actions)
        elif self.config["value_takeover"] or self.config["maxmin_takeover"]:
            return self.value_takeover(v_id, actions)

        vehicle = self.vehicles[v_id]
        action = actions
        steering = action[0]
        throttle = action[1]
        self.state_value = 0
        pre_save = vehicle.takeover
        if vehicle.config["use_saver"] or vehicle.expert_takeover:
            # saver can be used for human or another AI
            free_level = vehicle.config["free_level"] if not vehicle.expert_takeover else 1.0
            if not self.old_obs:
                obs = self.observations[v_id].observe(self.vehicle)
            else:
                obs = self.expert_observation.observe(vehicle)
            try:
                saver_a, a_0_p, a_1_p = expert_action_prob(action, obs, self.expert_policy_weights, algo=self.config["expert_policy_type"],
                                                           deterministic=vehicle.config["expert_deterministic"])
            except ValueError:
                print("Expert can not takeover, due to observation space mismathing!")
                assert False
                saver_a = action
            else:
                if free_level <= 1e-3:
                    steering = saver_a[0]
                    throttle = saver_a[1]
                elif free_level > 1e-3:
                    if a_0_p * a_1_p < 1 - vehicle.config["free_level"]:
                        steering, throttle = saver_a[0], saver_a[1]

        # indicate if current frame is takeover step
        vehicle.takeover = True if action[0] != steering or action[1] != throttle else False
        saver_info = {
            "takeover_start": True if not pre_save and vehicle.takeover else False,
            "takeover_end": True if pre_save and not vehicle.takeover else False,
            "takeover": vehicle.takeover if pre_save else False
        }
        if saver_info["takeover"]:
            saver_info["raw_action"] = [steering, throttle]
        return (steering, throttle) if saver_info["takeover"] else action, saver_info

    def rule_takeover(self, v_id, actions):
        vehicle = self.vehicles[v_id]
        action = actions[v_id]
        steering = action[0]
        throttle = action[1]
        if vehicle.config["use_saver"] or vehicle.expert_takeover:
            # saver can be used for human or another AI
            save_level = vehicle.config["save_level"] if not vehicle.expert_takeover else 1.0
            obs = self.observations[v_id].observe(vehicle)
            try:
                saver_a, a_0_p, a_1_p = expert_action_prob(action, obs, self.expert_policy_weights,
                                                           deterministic=vehicle.config["expert_deterministic"])
            except ValueError:
                print("Expert can not takeover, due to observation space mismathing!")
            else:
                if save_level > 0.9:
                    steering = saver_a[0]
                    throttle = saver_a[1]
                elif save_level > 1e-3:
                    heading_diff = vehicle.heading_diff(vehicle.lane) - 0.5
                    f = min(1 + abs(heading_diff) * vehicle.speed * vehicle.max_speed, save_level * 10)
                    # for out of road
                    if (obs[0] < 0.04 * f and heading_diff < 0) or (obs[1] < 0.04 * f and heading_diff > 0) or obs[
                        0] <= 1e-3 or \
                            obs[
                                1] <= 1e-3:
                        steering = saver_a[0]
                        throttle = saver_a[1]
                        if vehicle.speed < 5:
                            throttle = 0.5
                    # if saver_a[1] * vehicle.speed < -40 and action[1] > 0:
                    #     throttle = saver_a[1]

                    # for collision
                    lidar_p = vehicle.lidar.get_cloud_points()
                    left = int(vehicle.lidar.num_lasers / 4)
                    right = int(vehicle.lidar.num_lasers / 4 * 3)
                    if min(lidar_p[left - 4:left + 6]) < (save_level + 0.1) / 10 or min(lidar_p[right - 4:right + 6]
                                                                                        ) < (save_level + 0.1) / 10:
                        # lateral safe distance 2.0m
                        steering = saver_a[0]
                    if action[1] >= 0 and saver_a[1] <= 0 and min(min(lidar_p[0:10]), min(lidar_p[-10:])) < save_level:
                        # longitude safe distance 15 m
                        throttle = saver_a[1]

        # indicate if current frame is takeover step
        pre_save = vehicle.takeover
        vehicle.takeover = True if action[0] != steering or action[1] != throttle else False
        saver_info = {
            "takeover_start": True if not pre_save and vehicle.takeover else False,
            "takeover_end": True if pre_save and not vehicle.takeover else False,
            "takeover": vehicle.takeover if pre_save else False
        }
        return (steering, throttle) if saver_info["takeover"] else action, saver_info

if __name__ == "__main__":

    render=True
    algo="ppo"
    if algo == "ppo":
        weight = "ckpts/ppo_weights/ppo-expert-180.npz"
    else:
        weight = "ckpts/ts2c_weights/ts2c-expert-175.npz"
        

    env = ExpertGuidedEnv(dict(
        show_interface=False,
        show_terrain=True,
        show_skybox=True,
        show_logo=False,
        show_mouse=False,
        show_fps=False,
        environment_num=1,
        # start_seed=42,
        start_seed=440,
        vehicle_config=dict(
            use_saver=True,
            free_level=0.,
            expert_deterministic=True,
            show_navi_mark=False
          ),
        cull_scene=True,
        map_config={
            "config": 5,
        },
        cost_to_reward=True,
        expert_policy_type=algo,
        expert_policy_weights=weight,
        safe_rl_env=True,
        use_render=render,
        old_obs=False,
        manual_control=False))

    def _save(env):
        env.vehicle.vehicle_config["use_saver"]= not env.vehicle.vehicle_config["use_saver"]

    eval_reward = []
    done_num=0
    o = env.reset()
    env.engine.accept("p",env.capture)
    env.engine.accept("u", _save, extraArgs=[env])
    max_s = 0
    max_t = 0
    start = 0
    total_r = 0
    for i in range(1, 30000):
        o_to_evaluate = o
        o, r, d, info = env.step(env.action_space.sample())
        # frame = env.render(mode="top_down", num_stack=100, )
        # pygame.image.save(frame, "{}/{:05d}.png".format(full_name, i))
        # record pic
        # if i > 150:
            # img = PNMImage()
            # env.engine.win.getScreenshot(img)
            # img.write("pics/{}/{}.png".format(full_name, i))
        
        total_r += r
        max_s = max(max_s, info["raw_action"][0])
        max_t = max(max_t, info["raw_action"][1])

        # assert not info["takeover_start"]
        if env.config["cost_info"] == "native":
            assert info["cost"] == info["native_cost"]
            assert info["total_cost"] == info["total_native_cost"]
        elif env.config["cost_info"] == "takeover":
            assert info["cost"] == info["takeover_cost"]
            assert info["total_cost"] == info["total_takeover_cost"]
        else:
            raise ValueError
        if d:
            eval_reward.append(total_r)
            done_num+=1
            if done_num > 100:
                break
            print(info["out_of_road"])
            print("done_cost:{}".format(info["cost"]))
            print("done_reward:{}".format(r))
            print("total_takeover_cost:{}".format(info["total_takeover_cost"]))
            takeover_cost = 0
            native_cost = 0
            total_r = 0
            print("episode_len:", i - start)
            env.reset()
            start = i
    import numpy as np
    print(np.mean(eval_reward),np.std(sorted(eval_reward)))
    env.close()
