import torch
import numpy as np
import math
import genesis as gs
from genesis.utils.geom import quat_to_xyz, transform_by_quat, inv_quat, transform_quat_by_quat


def gs_rand_float(lower, upper, shape, device):
    return (upper - lower) * torch.rand(size=shape, device=device) + lower


class UREnv:
    def __init__(self, num_envs, env_cfg, obs_cfg, reward_cfg, command_cfg, show_viewer=False):
        self.num_envs = num_envs
        self.num_obs = obs_cfg["num_obs"]
        self.num_privileged_obs = None
        self.num_actions = env_cfg["num_actions"]
        self.num_commands = command_cfg["num_commands"]
        self.device = gs.device

        self.simulate_action_latency = True  # there is a 1 step latency on real robot
        self.dt = 0.02  # control frequency on real robot is 50hz
        self.max_episode_length = math.ceil(env_cfg["episode_length_s"] / self.dt)

        self.env_cfg = env_cfg
        self.obs_cfg = obs_cfg
        self.reward_cfg = reward_cfg
        self.command_cfg = command_cfg

        self.obs_scales = obs_cfg["obs_scales"]
        self.reward_scales = reward_cfg["reward_scales"]
        self.reached_goal = torch.tensor([False] * self.num_envs) # flag to indicate if the goal is reached in any environment


        # create scene
        self.scene = gs.Scene(
            sim_options=gs.options.SimOptions(dt=self.dt, substeps=2),
            viewer_options=gs.options.ViewerOptions(
                max_FPS=int(0.5 / self.dt),
                camera_pos=(1.5, -1.5, 1),
                camera_lookat=(0.0, 0.0, 0.3),
                camera_fov=40,
            ),
            vis_options=gs.options.VisOptions(rendered_envs_idx=list(range(1))),
            rigid_options=gs.options.RigidOptions(
                dt=self.dt,
                constraint_solver=gs.constraint_solver.Newton,
                enable_collision=True,
                enable_joint_limit=True,
            ),
            show_viewer=show_viewer,
        )

        # add plain
        self.scene.add_entity(gs.morphs.URDF(file="urdf/plane/plane.urdf", fixed=True))

        # add robot
        self.base_init_pos = torch.tensor(self.env_cfg["base_init_pos"], device=gs.device)
        self.base_init_quat = torch.tensor(self.env_cfg["base_init_quat"], device=gs.device)
        self.inv_base_init_quat = inv_quat(self.base_init_quat)
        self.robot = self.scene.add_entity(
            gs.morphs.URDF(
                file="../ur3_with_robotiq_2f_140_gripper.urdf", 
                fixed=True,
            ),
        )
        
        # Add a soft cube for grasping
        self.target_object = self.scene.add_entity(  
            gs.morphs.Box(  
                size=(0.04, 0.04, 0.04),  
                pos=(0.4, 0.0, 0.02),
            ),
        )
        
        self.cam = self.scene.add_camera(
            res=(640, 480), 
            pos=(2.0, 2.0, 2.5),
            lookat=(0.0, 0.0, 0.5),
            fov=40,
            GUI=False,
        )
        

        # build
        self.scene.build(n_envs=num_envs)

        # names to indices
        self.motors_dof_idx = list(np.arange(7))
        self.all_dof_idx = list(np.arange(7))

        # PD control parameters
        self.robot.set_dofs_kp(self.env_cfg["kp"], self.all_dof_idx)
        self.robot.set_dofs_kv(self.env_cfg["kd"], self.all_dof_idx)

        qpos = self.env_cfg["base_init_pos"]
        # qpos[0:12]を [num_envs, 12] の形にコピーしながら変形する。
        num_envs = self.num_envs
        qpos = torch.tensor(qpos, device=gs.device, dtype=gs.tc_float).repeat(num_envs, 1)
        self.robot.set_dofs_position(
            position=qpos[:, :len(self.motors_dof_idx)],
            dofs_idx_local=self.motors_dof_idx,
            zero_velocity=True,
            envs_idx=list(range(num_envs)),
        )
        
        # prepare reward functions and multiply reward scales by dt
        self.reward_functions, self.episode_sums = dict(), dict()
        for name in self.reward_scales.keys():
            self.reward_scales[name] *= self.dt
            self.reward_functions[name] = getattr(self, "_reward_" + name)
            self.episode_sums[name] = torch.zeros((self.num_envs,), device=gs.device, dtype=gs.tc_float)

        # initialize buffers
        self.obs_buf = torch.zeros((self.num_envs, self.num_obs), device=gs.device, dtype=gs.tc_float)
        self.rew_buf = torch.zeros((self.num_envs,), device=gs.device, dtype=gs.tc_float)
        self.reset_buf = torch.ones((self.num_envs,), device=gs.device, dtype=gs.tc_int)
        self.episode_length_buf = torch.zeros((self.num_envs,), device=gs.device, dtype=gs.tc_int)
        self.commands = torch.zeros((self.num_envs, self.num_commands), device=gs.device, dtype=gs.tc_float)
        self.commands_scale = torch.tensor(
            [1.0] * self.num_commands,
            device=gs.device,
            dtype=gs.tc_float,
        )
        self.actions = torch.zeros((self.num_envs, self.num_actions), device=gs.device, dtype=gs.tc_float)
        self.last_actions = torch.zeros_like(self.actions)
        self.dof_pos = torch.zeros_like(self.actions)
        #self.dof_vel = torch.zeros_like(self.actions)
        #self.last_dof_vel = torch.zeros_like(self.actions)
        self.default_dof_pos = torch.tensor(
            [self.env_cfg["default_joint_angles"][name] for name in self.env_cfg["joint_names"]],
            device=gs.device,
            dtype=gs.tc_float,
        )
        self.extras = dict()  # extra information for logging
        self.extras["observations"] = dict()

    def _resample_commands(self, envs_idx):
        #self.commands[envs_idx, 0] = gs_rand_float(*self.command_cfg["lin_vel_x_range"], (len(envs_idx),), gs.device)
        #self.commands[envs_idx, 1] = gs_rand_float(*self.command_cfg["lin_vel_y_range"], (len(envs_idx),), gs.device)
        #self.commands[envs_idx, 2] = gs_rand_float(*self.command_cfg["ang_vel_range"], (len(envs_idx),), gs.device)
        pass
        
    def step(self, actions):
        qpos_all = self.robot.get_qpos(self.all_dof_idx)
        qpos = qpos_all[:, self.motors_dof_idx]  # get only the first 7 joints (motors)
        self.actions = torch.clip(actions, -self.env_cfg["clip_actions"], self.env_cfg["clip_actions"])
        target_dof_pos = self.actions * self.env_cfg["action_scale"] + qpos
        
        
        self.robot.control_dofs_position(target_dof_pos, self.motors_dof_idx)
        self.scene.step()

        # update buffers
        self.episode_length_buf += 1
        self.dof_pos[:] = self.robot.get_dofs_position(self.motors_dof_idx)
        #self.dof_vel[:] = self.robot.get_dofs_velocity(self.motors_dof_idx)

        # resample commands
        envs_idx = (
            (self.episode_length_buf % int(self.env_cfg["resampling_time_s"] / self.dt) == 0)
            .nonzero(as_tuple=False)
            .reshape((-1,))
        )
        #self._resample_commands(envs_idx)

        # check termination and reset
        self.reset_buf = self.episode_length_buf > self.max_episode_length
        self.reset_buf |= self.reached_goal
        
        time_out_idx = (self.episode_length_buf > self.max_episode_length).nonzero(as_tuple=False).reshape((-1,))
        self.extras["time_outs"] = torch.zeros_like(self.reset_buf, device=gs.device, dtype=gs.tc_float)
        self.extras["time_outs"][time_out_idx] = 1.0

        self.reset_idx(self.reset_buf.nonzero(as_tuple=False).reshape((-1,)))

        # compute reward
        self.rew_buf[:] = 0.0
        for name, reward_func in self.reward_functions.items():
            rew = reward_func() * self.reward_scales[name]
            self.rew_buf += rew
            self.episode_sums[name] += rew
            
        qpos_all = self.robot.get_dofs_position(self.motors_dof_idx)
        qpos = qpos_all[:, :7]  # get only the first 7 joints (motors)
        links_pos = self.robot.get_links_pos()
        links_quat = self.robot.get_links_quat()
        eepos = links_pos[:, 5, :3]  # end effector position
        eequat = links_quat[:, 5, :4]  # end effector quaternion

        # compute observations
        self.obs_buf = torch.cat(
            [
                eepos, # 3
                eequat, # 4
                qpos, # 7
                self.actions, # 7
            ],
            axis=-1,
        )

        self.last_actions[:] = self.actions[:]
        #self.last_dof_vel[:] = self.dof_vel[:]

        self.extras["observations"]["critic"] = self.obs_buf

        return self.obs_buf, self.rew_buf, self.reset_buf, self.extras

    def get_observations(self):
        self.extras["observations"]["critic"] = self.obs_buf
        return self.obs_buf, self.extras

    def get_privileged_observations(self):
        return None

    def reset_idx(self, envs_idx):
        if len(envs_idx) == 0:
            return

        # reset dofs
        qpos = self.env_cfg["base_init_pos"]
        # qpos[0:12]を [num_envs, 12] の形にコピーしながら変形する。
        num_envs = self.num_envs
        qpos = torch.tensor(qpos, device=gs.device, dtype=gs.tc_float).repeat(num_envs, 1)
        self.robot.set_dofs_position(
            position=qpos[:, :len(self.motors_dof_idx)],
            dofs_idx_local=self.motors_dof_idx,
            zero_velocity=True,
            envs_idx=list(range(num_envs)),
        )
        
        # reset buffers
        self.last_actions[envs_idx] = 0.0
        #self.last_dof_vel[envs_idx] = 0.0
        self.episode_length_buf[envs_idx] = 0
        self.reset_buf[envs_idx] = True

        # fill extras
        self.extras["episode"] = {}
        for key in self.episode_sums.keys():
            self.extras["episode"]["rew_" + key] = (
                torch.mean(self.episode_sums[key][envs_idx]).item() / self.env_cfg["episode_length_s"]
            )
            self.episode_sums[key][envs_idx] = 0.0

        self._resample_commands(envs_idx)

    def reset(self):
        self.reset_buf[:] = True
        self.reset_idx(torch.arange(self.num_envs, device=gs.device))
        self.reached_goal = False
        return self.obs_buf, None

    # ------------ reward functions----------------
    def _reward_reach_target(self):
        links_pos = self.robot.get_links_pos()
        eepos = links_pos[:, 5, :3]  # end effector position
        target_pos = torch.tensor(self.reward_cfg["target_pos"], device=self.device)
        target_pos_broadcasted = target_pos.unsqueeze(0).repeat(self.num_envs, 1)
        #target_quat = np.array(self.reward_cfg["target_quat"])
        # エンドエフェクタとターゲットの距離
        distance = torch.norm(eepos - target_pos_broadcasted, dim=1) # torch.normを正しく使用
        return torch.exp(-distance * 10.0)  # 距離に基づく報酬

    def _reward_grasp_success(self):
        links_pos = self.robot.get_links_pos()
        eepos = links_pos[:, 5, :3]  # end effector position
        target_pos = torch.tensor(self.reward_cfg["target_pos"], device=self.device)
        target_pos_broadcasted = target_pos.unsqueeze(0).repeat(self.num_envs, 1)      
        # 把持成功の判定（距離とグリッパーの状態）
        distance = torch.norm(eepos - target_pos_broadcasted, dim=1) # 修正後
        # グリッパーが閉じていて、オブジェクトが近くにある場合
        grasp_threshold = 0.02  # 2cm
        return (distance < grasp_threshold).float()

    def _reward_action_smoothness(self):
        # アクションの変化を小さくする
        return torch.sum(torch.square(self.last_actions - self.actions), dim=1)
