# Guarded Policy Optimization with Imperfect Online Demonstrations

[**Webpage**](https://metadriverse.github.io/TS2C/) | 
[**Code**](https://github.com/metadriverse/TS2C) |
[**Paper**](https://openreview.net/forum?id=O5rKg7IRQIO)

# Installation

## Create virtual environment
```
conda create -n ts2c python=3.7
conda activate ts2c
```

## Install basic dependency
```
pip install -e .
pip install opencv-python-headless==4.1.2.30
pip uninstall nvidia_cublas_cu11
```
The last two command help to avoid some bugs. You also need to install mujoco_py if you would like to do experiments on the MuJoCo Simulator. Please perform the setup instructions here: https://github.com/openai/mujoco-py/.

# Training TS2C
## Training on Metadrive
First, change directory to training script.
```bash
cd ts2c/training_script/
```
You can train the teacher policy from scratch:
```bash
python train_ppo_baseline.py
python train_sac_baseline.py
```
Then use the teacher policy checkpoint and `egpo_utils/parse_expert.py` to generate policy weight file. But you can also use the pretrained teacher policies in `egpo_utils/saved_expert`.

Finally, train the student policy with with teacher-student shared control.
```
python train_egpo.py --exp-name ts2c --local-dir PATH_TO_SAVE_DIR --expert-policy-type ppo --expert-level 30 --value-takeover --value-from-scratch --ensemble --egpo-ensemble --start-seed 0 --warmup-ts 50000 --ckpt-freq 10 --no-cql --warmup-noise 0.3 --num-gpus 1
```

## Training on MuJoCo
TBD

# Code Navigation
`egpo_utils/expert_guided_env.py` is the core file to implement the value-based intervention in TS2C. You can check line 259-266 for detailed implementation.