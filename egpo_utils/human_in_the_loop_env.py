from metadrive.envs.safe_metadrive_env import SafeMetaDriveEnv
from metadrive.policy.manual_control_policy import TakeoverPolicy
from metadrive.engine.core.onscreen_message import ScreenMessage
from metadrive.engine.core.manual_controller import KeyboardController
from metadrive.engine.engine_utils import get_global_config

ScreenMessage.SCALE = 0.1

class KeyboardTakeoverController(KeyboardController):

    def __init__(self):
        super(KeyboardController, self).__init__()
        self.inputs.watchWithModifiers('takeover', ']')

class KeyboardTakeoverPolicy(TakeoverPolicy):
    """
    Record the takeover signal
    """
    def __init__(self):
        super(TakeoverPolicy, self).__init__()
        config = get_global_config()
        if config["manual_control"] and config["use_render"]:
            if config["controller"] == "keyboard":
                self.controller = KeyboardTakeoverController()
            else:
                raise ValueError("Takeover policy can only be activated with Keyboard")
        self.takeover = False

    def act(self, agent_id):
        print(1)
        agent_action = super(TakeoverPolicy, self).act(agent_id)
        if self.engine.global_config["manual_control"] and self.engine.agent_manager.get_agent(
                agent_id) is self.engine.current_track_vehicle and not self.engine.main_camera.is_bird_view_camera():
            expert_action = self.controller.process_input(self.engine.current_track_vehicle)
            if self.controller.inputs.isSet("takeover"):
                # if expert_action[0]*agent_action[0]< 0 or expert_action[1]*agent_action[1] < 0:
                self.takeover = True
                return expert_action
        self.takeover = False
        return agent_action

class HumanInTheLoopEnv(SafeMetaDriveEnv):
    """
    This Env depends on the new version of MetaDrive
    """

    def default_config(self):
        config = super(HumanInTheLoopEnv, self).default_config()
        config.update(
            {
                "environment_num": 1,
                "start_seed": 10,
                "map": "Cr",
                "cost_to_reward": True,
                "manual_control": True,
                "controller": "keyboard",
                "agent_policy": KeyboardTakeoverPolicy
            },
            allow_add_new_key=True
        )
        return config

    def reset(self, *args, **kwargs):
        self.t_o = False
        self.total_takeover_cost = 0
        return super(HumanInTheLoopEnv, self).reset(*args, **kwargs)

    def _get_step_return(self, actions, step_infos):
        o, r, d, step_infos = super(HumanInTheLoopEnv, self)._get_step_return(actions, step_infos)
        controller = self.engine.get_policy(self.vehicle.id)
        last_t_o = self.t_o
        self.t_o = controller.takeover if hasattr(controller, "takeover") else False
        step_infos["takeover"] = self.t_o
        if step_infos["takeover"] and not last_t_o:
            self.total_takeover_cost += 1
        step_infos["takeover_cost"] = 1 if step_infos["takeover"] else 0
        step_infos["total_takeover_cost"] = self.total_takeover_cost
        step_infos["native_cost"] = step_infos["cost"]
        step_infos["total_native_cost"] = self.episode_cost
        return o, r, d, step_infos

    def step(self, actions):
        ret = super(HumanInTheLoopEnv, self).step(actions)
        if self.config["use_render"]:
            super(HumanInTheLoopEnv, self).render(text={
                "Total Cost": self.episode_cost,
                "Total Takeover Cost": self.total_takeover_cost,
                "Takeover": self.t_o
            })
        return ret


if __name__ == "__main__":
    env = HumanInTheLoopEnv(
        {
            "manual_control": True,
            "use_render": True,
        }

    )

    o = env.reset()
    total_cost = 0
    for i in range(1, 100000):
        o, r, d, info = env.step(env.action_space.sample())
        total_cost += info["cost"]
        # env.render(
        #     text={
        #         "cost": total_cost,
        #         "seed": env.current_seed,
        #         "reward": r,
        #         "total_cost": info["total_cost"],
        #         "total_takeover_cost": info["total_takeover_cost"],
        #         "takeover": info["takeover"]
        #     }
        # )
        if info["crash_vehicle"]:
            print("crash_vehicle:cost {}, reward {}".format(info["cost"], r))
        if info["crash_object"]:
            print("crash_object:cost {}, reward {}".format(info["cost"], r))

        if d:
            total_cost = 0
            print("done_cost:{}".format(info["cost"]), "done_reward;{}".format(r))
            print("Reset")
            env.reset()
    env.close()
