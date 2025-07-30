import argparse
import os
import pickle
import shutil
from importlib import metadata

try:
    try:
        if metadata.version("rsl-rl"):
            raise ImportError
    except metadata.PackageNotFoundError:
        if metadata.version("rsl-rl-lib") != "2.2.4":
            raise ImportError
except (metadata.PackageNotFoundError, ImportError) as e:
    raise ImportError("Please uninstall 'rsl_rl' and install 'rsl-rl-lib==2.2.4'.") from e
from rsl_rl.runners import OnPolicyRunner

import genesis as gs

from ur3_env import UREnv


def get_train_cfg(exp_name, max_iterations):
    train_cfg_dict = {
        "algorithm": {
            "class_name": "PPO",
            "clip_param": 0.2,
            "desired_kl": 0.01,
            "entropy_coef": 0.01,
            "gamma": 0.99,
            "lam": 0.95,
            "learning_rate": 0.001,
            "max_grad_norm": 1.0,
            "num_learning_epochs": 5,
            "num_mini_batches": 4,
            "schedule": "adaptive",
            "use_clipped_value_loss": True,
            "value_loss_coef": 1.0,
        },
        "init_member_classes": {},
        "policy": {
            "activation": "elu",
            "actor_hidden_dims": [512, 256, 128],
            "critic_hidden_dims": [512, 256, 128],
            "init_noise_std": 1.0,
            "class_name": "ActorCritic",
        },
        "runner": {
            "checkpoint": -1,
            "experiment_name": exp_name,
            "load_run": -1,
            "log_interval": 1,
            "max_iterations": max_iterations,
            "record_interval": -1,
            "resume": False,
            "resume_path": None,
            "run_name": "",
        },
        "runner_class_name": "OnPolicyRunner",
        "num_steps_per_env": 24,
        "save_interval": 100,
        "empirical_normalization": None,
        "seed": 1,
    }

    return train_cfg_dict


def get_cfgs():
    env_cfg = {
        "num_actions": 7,   # 6 for UR3 joints, 1 for gripper
        # joint/link names
        "default_joint_angles": {  # [rad]
            'shoulder_pan_joint' : -0.0,
            'shoulder_lift_joint' : -0.9,
            'elbow_joint'  : -0.5,
            'wrist_1_joint' :  -1.4,
            'wrist_2_joint' : -1.3,
            'wrist_3_joint' : -0.3,
            'finger_joint' : 0.04,
        },
        "joint_names": [
            'shoulder_pan_joint',
            'shoulder_lift_joint',
            'elbow_joint',
            'wrist_1_joint',
            'wrist_2_joint',
            'wrist_3_joint',
            'finger_joint',
        ],
        # PD
        "kp": [4500, 4500, 3500, 3500, 2000, 2000, 100,],
        "kd": [450,   450,  350,  350,  200,  200, 10,],
        "force_limit_l": [-87, -87, -87, -87, -87, -87, -12,],
        "force_limit_u": [ 87,  87,  87,  87,  87,  87,  12,],
        # termination
        #"termination_if_roll_greater_than": 10,  # degree
        #"termination_if_pitch_greater_than": 10,
        # base pose
        "base_init_pos": [  -0, -0.9,  -0.5,  -1.4,  -1.3,  -0.3, 0.04],
        "base_init_quat": [0.0, 1, 0.0, 0.0],
        "episode_length_s": 4.0,
        "resampling_time_s": 4.0,
        "action_scale": 0.05,
        "simulate_action_latency": True,
        "clip_actions": 1.0,
    }
    obs_cfg = {
        "num_obs": 21,  # 3 for EE postion, 4 for EE quaternion, 7 for joint positions, 7 for action
        "obs_scales": {
            "lin_vel": 2.0,
            "ang_vel": 0.25,
            "dof_pos": 1.0,
            "dof_vel": 0.05,
        },
    }
    reward_cfg = {
        "target_pos" : [0.65, 0.0, 0.15],
        "target_quat": [0.0, 1.0, 0.0, 0.0],
        "orientation_reward_scale": 5.0,
        "reward_scales": {
            "reach_target": 1.0,  # ターゲットに到達
            "ee_quat": 0.8,  # エンドエフェクタのクォータニオン
            "grasp_success": 0.0,  # 把持成功
            "action_smoothness": -0.1,  # アクションの滑らかさ
            "collision_penalty": -1.0,  # 衝突ペナルティ
        },
    }
    command_cfg = {
        "num_commands": 6,
    }

    return env_cfg, obs_cfg, reward_cfg, command_cfg


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--exp_name", type=str, default="ur-pick")
    parser.add_argument("-B", "--num_envs", type=int, default=4096)
    parser.add_argument("-I", "--max_iterations", type=int, default=101)
    args = parser.parse_args()

    gs.init(logging_level="warning")

    log_dir = f"logs/{args.exp_name}"
    env_cfg, obs_cfg, reward_cfg, command_cfg = get_cfgs()
    train_cfg = get_train_cfg(args.exp_name, args.max_iterations)

    if os.path.exists(log_dir):
        shutil.rmtree(log_dir)
    os.makedirs(log_dir, exist_ok=True)

    pickle.dump(
        [env_cfg, obs_cfg, reward_cfg, command_cfg, train_cfg],
        open(f"{log_dir}/cfgs.pkl", "wb"),
    )

    env = UREnv(
        num_envs=args.num_envs, env_cfg=env_cfg, obs_cfg=obs_cfg, reward_cfg=reward_cfg, command_cfg=command_cfg
    )

    runner = OnPolicyRunner(env, train_cfg, log_dir, device=gs.device)

    runner.learn(num_learning_iterations=args.max_iterations, init_at_random_ep_len=True)


if __name__ == "__main__":
    main()

"""
# training
python examples/locomotion/go2_train.py
"""
