import argparse
import logging
import os

import ray


def initialize_ray(local_mode=False, num_gpus=None, test_mode=False, **kwargs):
    os.environ['OMP_NUM_THREADS'] = '1'

    if ray.__version__.split(".")[0] == "1":  # 1.0 version Ray
        if "redis_password" in kwargs:
            redis_password = kwargs.pop("redis_password")
            kwargs["_redis_password"] = redis_password

    ray.init(
        logging_level=logging.ERROR if not test_mode else logging.DEBUG,
        log_to_driver=test_mode,
        local_mode=local_mode,
        num_gpus=num_gpus,
        ignore_reinit_error=True,
        **kwargs
    )
    print("Successfully initialize Ray!")
    try:
        print("Available resources: ", ray.available_resources())
    except Exception:
        pass


def get_train_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp-name", type=str, default="")
    parser.add_argument("--local-dir", type=str, default="./")
    parser.add_argument("--env", type=str)
    parser.add_argument("--num-gpus", type=int, default=0)
    parser.add_argument("--num-seeds", type=int, default=1)
    parser.add_argument("--start-seed", type=int, default=0)
    parser.add_argument("--num-cpus-per-worker", type=float, default=0.5)
    parser.add_argument("--num-gpus-per-trial", type=float, default=0.25)
    parser.add_argument("--il-expert-coef", type=float, default=0.)
    parser.add_argument("--il-agent-coef", type=float, default=0.)
    parser.add_argument("--free-level", type=float, default=0.95)
    parser.add_argument("--value-takeover-threshold", type=float, default=1.2)
    parser.add_argument("--var-threshold", type=float, default=2.5)
    parser.add_argument("--test", action="store_true")
    parser.add_argument("--value-takeover", action="store_true")
    parser.add_argument("--dagger-takeover", action="store_true")
    parser.add_argument("--maxmin-takeover", action="store_true")
    parser.add_argument("--uncertainty-takeover", action="store_true")
    # parser.add_argument("--sac-policy", action="store_true")
    parser.add_argument("--expert-policy-type", type=str, default="ppo")
    parser.add_argument("--takeover-mode", type=str, default="action")
    parser.add_argument("--expert-level", type=int, default=None)
    parser.add_argument("--sub-experts-prefix", type=str, default=None)
    parser.add_argument("--expert-name", type=str, default=None)
    parser.add_argument("--no-cost-minimization", type=bool, default=True)
    parser.add_argument("--expert-value-weights", type=str, default="default")
    # tsk: timestep in k
    parser.add_argument("--tsk", type=int, default=200)
    parser.add_argument("--warmup-ts", type=int, default=0)
    parser.add_argument("--max-index", type=int, default=60)
    parser.add_argument("--warmup-noise", type=float, default=0.5)
    parser.add_argument("--ckpt-freq", type=int, default=10)
    parser.add_argument("--old-obs", action="store_true")
    parser.add_argument("--value-from-scratch", action="store_true")
    parser.add_argument("--no-cql", action="store_true")
    parser.add_argument("--ensemble", action="store_true")
    parser.add_argument("--egpo-ensemble", action="store_true")
    parser.add_argument("--expert-det", action="store_true")
    parser.add_argument("--fix-lambda", action="store_true")
    parser.add_argument("--late-learning-start", action="store_true")
    parser.add_argument("--lambda-init", type=float, default=200.)
    parser.add_argument("--value-fn-path", type=str, default=None)
    return parser

def setup_logger(debug=False):
    import logging
    logging.basicConfig(
        level=logging.DEBUG if debug else logging.WARNING,
        format='%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s'
    )
