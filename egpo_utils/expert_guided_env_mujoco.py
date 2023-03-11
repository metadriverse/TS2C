from gym.envs.mujoco import AntEnv, HopperEnv, Walker2dEnv, HalfCheetahEnv, HumanoidEnv
from egpo_utils.common import expert_action_prob, ExpertObservation, expert_q_value, ensemble_q_value
from numpy.linalg import norm
from numpy import load, array, average, var, zeros, clip
from numpy.random import randn
from copy import deepcopy
from os import scandir, getcwd
from egpo_utils.save_ppo_expert import compress_model
import os.path as osp


cur_warmup_ts = 0

def load_latest_value_weights(self):
    exp_dir = self.exp_path
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
        self.value_weights = compress_model(osp.join(exp_dir, ckpt_dir, "checkpoint_%d"%latest_idx, "checkpoint-%d"%latest_idx))


def get_q_value(self, a=None):
    if self.value_weights is None or self.o is None:
        return zeros(5)
    if a is None:
        a, *_ = expert_action_prob([0, 0],
                               self.o,
                               self.expert_weights,
                               algo=self.algo,
                               deterministic=True)
    return ensemble_q_value(a, self.o, self.value_weights)

def expert_takeover(self, a):
    takeover = False
    warmup = False
    saver_info = {'raw_action': a, 
                  'takeover': False, 
                  'warmup': False,
                  'takeover_cost': 0, 
                  'native_cost': 0, 
                  'total_takeover_cost': self.total_takeover}
    if self.eval:
        return a, saver_info
    if self.o is None:
        # during init
        return a, saver_info
    return_a = a
    if self.takeover_mode == 'action' and self.o is not None:
        saver_a, *_ = expert_action_prob(None, 
                                         self.o, 
                                         self.expert_weights, 
                                         algo=self.algo, 
                                         deterministic=True)
        # print("===========norm============")
        # print(norm(a-saver_a, ord=1))
        # print("=========norm============")
        if norm(a-saver_a, ord=1) > self.threshold:
            takeover = True

    elif self.takeover_mode == 'value':
        global cur_warmup_ts
        if cur_warmup_ts % self.update_value_fraq == 1:
            load_latest_value_weights(self)

        cur_warmup_ts += 1
        if cur_warmup_ts < self.tot_warmup_ts and not self.eval:
            takeover = True
            warmup = True
        else:
            expert_values = get_q_value(self)
            agent_values = get_q_value(self, a)
            diff = array(expert_values) - array(agent_values)
            diff_mean = average(diff)
            agent_var = var(agent_values)
            # print("====================action stats================")
            # print(diff_mean)
            # print(agent_var)
            # print("====================action stats================")
            if diff_mean > self.threshold or agent_var > self.var_threshold:
                takeover = True

        if takeover:
            saver_a, *_ = expert_action_prob(None, 
                                            self.o, 
                                            self.expert_weights, 
                                            algo=self.algo,
                                            deterministic=False,
                                                )
            
            if warmup:
                saver_a += self.warmup_noise * randn(self.act_dim)
                saver_a = clip(saver_a, -1, 1)
    
    if takeover:
        saver_info["takeover"] = takeover
        saver_info['takeover_cost'] = 1
        saver_info['warmup'] = warmup
        return_a = saver_a
        
    return return_a, saver_info

def common_init(self, config):
    self.takeover_mode = config.get('takeover_mode', None)
    self.expert_weights_path = config.get('expert_weights', None)
    self.expert_weights = None
    if self.expert_weights_path:
        self.expert_weights = load(self.expert_weights_path)

    self.exp_path = config.get('exp_path', None)
    self.threshold = config.get('threshold', 2.)
    self.var_threshold = config.get('var_threshold', 2.)
    self.o = None
    self.total_takeover = 0
    self.tot_warmup_ts = config.get('warmup_ts', None)
    self.warmup_noise = config.get('warmup_noise', 0.5)
    self.algo = config.get('expert_policy_type', 'ppo')
    self.act_dim = config.get('act_dim', 11)
    self.obs_dim = config.get('obs_dim', 11)
    self.eval = config.get('evaluate')
    self.value_weights = None
    self.update_value_fraq = 1000
    self.latest_idx = 0
    self.max_idx_for_value = config.get('max_index', None)

def common_takeover(self, a):
    a, saver_info = expert_takeover(self, a)
    if saver_info['takeover']:
        self.total_takeover += 1
    return a, saver_info

class HopperTS2CEnv(HopperEnv):
    def __init__(self, 
                 config,
        ):
        common_init(self, config)
        super(HopperTS2CEnv, self).__init__()

    def reset(self):
        self.o = super(HopperTS2CEnv, self).reset()
        self.total_takeover = 0
        return self.o

    def step(self, a):
        a, saver_info = common_takeover(self, a)
        self.o, r, d, info = super(HopperTS2CEnv, self).step(a)
        info.update(saver_info)
        return self.o, r, d, info

class Walker2dTS2CEnv(Walker2dEnv):
    def __init__(self, 
                 config,
        ):
        common_init(self, config)
        super(Walker2dTS2CEnv, self).__init__()

    def reset(self):
        self.o = super(Walker2dTS2CEnv, self).reset()
        self.total_takeover = 0
        return self.o

    def step(self, a):
        a, saver_info = common_takeover(self, a)
        # print("===========probe action==========")
        # print(a)
        # print("===========probe action==========")
        # print()
        self.o, r, d, info = super(Walker2dTS2CEnv, self).step(a)
        info.update(saver_info)
        return self.o, r, d, info

class HalfCheetahTS2CEnv(HalfCheetahEnv):
    def __init__(self, 
                 config,
        ):
        common_init(self, config)
        super(HalfCheetahTS2CEnv, self).__init__()

    def reset(self):
        self.o = super(HalfCheetahTS2CEnv, self).reset()
        self.total_takeover = 0
        return self.o

    def step(self, a):
        a, saver_info = common_takeover(self, a)
        self.o, r, d, info = super(HalfCheetahTS2CEnv, self).step(a)
        info.update(saver_info)
        return self.o, r, d, info


class AntTS2CEnv(AntEnv):
    def __init__(self, 
                 config,
        ):
        common_init(self, config)
        super(AntTS2CEnv, self).__init__()

    def reset(self):
        self.o = super(AntTS2CEnv, self).reset()
        self.total_takeover = 0
        return self.o

    def step(self, a):
        a, saver_info = common_takeover(self, a)
        self.o, r, d, info = super(AntTS2CEnv, self).step(a)
        info.update(saver_info)
        return self.o, r, d, info
