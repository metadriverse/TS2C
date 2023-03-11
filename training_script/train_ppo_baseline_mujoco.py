from ray.rllib.agents.ppo.ppo import PPOTrainer
from egpo_utils.train import train, get_train_parser
from egpo_utils.expert_guided_env_mujoco import HopperTS2CEnv, Walker2dTS2CEnv, HalfCheetahTS2CEnv, AntTS2CEnv
# from egpo_utils.common import EGPOCallbacks
from ray.rllib.agents.callbacks import DefaultCallbacks
env_dict = {
    "hopper": HopperTS2CEnv,
    "walker": Walker2dTS2CEnv,
    "cheetah": HalfCheetahTS2CEnv,
    "ant": AntTS2CEnv,
}

if __name__ == '__main__':
    args = get_train_parser().parse_args()

    exp_name = args.exp_name or "PPO" 
    stop = {"timesteps_total": 20_000_000}

    config = dict(
        env=env_dict[args.env],
        env_config=dict(
        ),

        # ===== Evaluation =====
        evaluation_interval=1,
        evaluation_num_episodes=30,
        evaluation_num_workers=2,
        metrics_smoothing_episodes=20,

        # ===== Training =====
        horizon=1000,
        num_sgd_iter=20,
        lr=5e-5,
        grad_clip=10.0,
        rollout_fragment_length=200,
        sgd_minibatch_size=100,
        train_batch_size=4000,
        num_gpus=args.num_gpus,
        num_cpus_per_worker=0.1,
        num_cpus_for_driver=0.5,
        num_workers=8,
        clip_actions=False
    )

    train(
        PPOTrainer,
        exp_name=exp_name,
        keep_checkpoints_num=None,
        stop=stop,
        config=config,
        num_gpus=args.num_gpus,
        custom_callback=DefaultCallbacks,
        num_seeds=args.num_seeds,
        start_seed=0,
        checkpoint_freq=10,
        local_dir=args.local_dir,
        # test_mode=True,
        # local_mode=True
    )
