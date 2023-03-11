from ray import tune
from egpo_utils.egpo.egpo import EGPOTrainer
from egpo_utils.egpo.egpo_ensemble import EGPOEnsembleTrainer
from egpo_utils.expert_guided_env import ExpertGuidedEnv
from egpo_utils.common import EGPOCallbacks, evaluation_config
from egpo_utils.train import train, get_train_parser
from os.path import join
import pathlib

if __name__ == '__main__':
    args = get_train_parser().parse_args()

    exp_name = args.exp_name or "EGPO" 
    stop = {"timesteps_total": args.tsk * 1000}

    path_prefix = pathlib.Path(__file__).resolve().parents[1]

    # set expert weights
    if not args.expert_level and not args.expert_name:
        expert_policy_weights = "egpo_utils/saved_expert/sac_expert.npz" if args.expert_policy_type=="sac" else \
                               "egpo_utils/expert.npz"
    else:
        if not args.sub_experts_prefix:
            expert_policy_weights = "egpo_utils/saved_expert/ppo_expert_{}.npz".format(args.expert_level)
        else:
            if not args.expert_name:
                expert_policy_weights = "egpo_utils/saved_expert/{}/ppo_expert_{}.npz".format(args.sub_experts_prefix, args.expert_level)
            else:
                expert_policy_weights = "egpo_utils/saved_expert/{}/{}.npz".format(args.sub_experts_prefix, args.expert_name)
    
    expert_policy_weights = join(str(path_prefix), expert_policy_weights)

    config = dict(
        env=ExpertGuidedEnv,
        env_config=dict(
            vehicle_config=dict(
                use_saver=True,
                free_level=args.free_level,
                expert_deterministic=args.expert_det,
            ),
            cost_to_reward=True,
            value_takeover=args.value_takeover,
            dagger_takeover=args.dagger_takeover,
            maxmin_takeover=args.maxmin_takeover,
            uncertainty_takeover=args.uncertainty_takeover,
            value_takeover_threshold=args.value_takeover_threshold,
            var_threshold=args.var_threshold,
            expert_policy_type=args.expert_policy_type,
            expert_policy_weights=expert_policy_weights,
            expert_value_weights=args.expert_value_weights if not args.value_from_scratch else None,
            old_obs=False if not args.old_obs else args.expert_policy_type=="ppo",
            warmup_ts=args.warmup_ts,
            warmup_noise=args.warmup_noise,
            ensemble=args.ensemble,
            value_from_scratch=args.value_from_scratch,
            value_fn_path=args.value_fn_path,
            exp_path=join(args.local_dir, exp_name),
            safe_rl_env=True),

        # ===== Evaluation =====
        evaluation_interval=1,
        evaluation_num_episodes=30,
        evaluation_config=evaluation_config,
        evaluation_num_workers=2,
        metrics_smoothing_episodes=20,

        # ===== Training =====
        takeover_data_discard=False,
        alpha=3.0,
        recent_episode_num=5,
        normalize=True,
        twin_cost_q=True,
        k_i=0.01,
        k_p=5,
        # search > 0
        k_d=0.1,
        il_agent_coef=args.il_agent_coef,
        il_expert_coef=args.il_expert_coef,
        no_cql=args.no_cql,
        fix_lambda=args.fix_lambda,
        lambda_init=args.lambda_init,
        no_cost_minimization=args.no_cost_minimization,

        # expected max takeover num
        cost_limit=2,
        optimization=dict(actor_learning_rate=1e-4, critic_learning_rate=1e-4, entropy_learning_rate=1e-4),
        prioritized_replay=False,
        horizon=2000,
        target_network_update_freq=1,
        timesteps_per_iteration=1000,
        learning_starts=0,
        clip_actions=False,
        normalize_actions=True,
        num_cpus_for_driver=0.5,
        # No extra worker used for learning. But this config impact the evaluation workers.
        num_cpus_per_worker=0.1,
        # num_gpus_per_worker=0.1 if args.num_gpus != 0 else 0,
        num_gpus=0.2 if args.num_gpus != 0 else 0,
    )

    train(
        EGPOTrainer if not args.egpo_ensemble else EGPOEnsembleTrainer,
        exp_name=exp_name,
        keep_checkpoints_num=None,
        checkpoint_freq=args.ckpt_freq,
        stop=stop,
        config=config,
        num_gpus=args.num_gpus,
        # num_seeds=2,
        num_seeds=args.num_seeds,
        start_seed=args.start_seed,
        custom_callback=EGPOCallbacks,
        test_mode=False,
        local_dir=args.local_dir,
        # local_mode=True
    )
