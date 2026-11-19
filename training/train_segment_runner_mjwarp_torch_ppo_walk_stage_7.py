from __future__ import annotations

"""Stable full-toe-contact walk stage-7 MJWarp + PyTorch PPO trainer.

This trainer starts from the stage-6 wave-stable checkpoint, keeps the same
half-cycle toe/COM wave reference, and adds stronger full toe-pair contact at
the start and end of each step. It also lengthens the gait cycle and penalizes
fore-aft velocity oscillation so the torso advances with less back-and-forth
sway.

Use this file after train_segment_runner_mjwarp_torch_ppo_walk_stage_6.py has
produced a wave-stable checkpoint.
"""

import argparse
import math
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.distributions import Normal


DEFAULT_XML = "robot_segment_trainable.xml"
MODEL_BASENAME = "segment_runner_mjwarp"
DEFAULT_STAGE_6_CHECKPOINT = "artifacts/mjwarp_walk_stage_6/segment_runner_mjwarp.pt"
FORWARD_PROGRESS_SIGMA = 0.060

NOMINAL_JOINT_POSE = np.array(
    [-0.04, 0.04, -0.16, -0.16, 0.18, 0.18, -0.20, -0.20, -0.20, -0.20],
    dtype=np.float32,
)
POSITION_ACTION_SCALE = np.array(
    [1.10, 1.10, 0.94, 0.94, 0.69, 0.69, 0.195, 0.195, 0.195, 0.195],
    dtype=np.float32,
)
PD_KP = np.array([82.0, 82.0, 80.0, 80.0, 70.0, 70.0, 22.0, 22.0, 22.0, 22.0], dtype=np.float32)
PD_KD = np.array([2.7, 2.7, 2.6, 2.6, 1.9, 1.9, 0.7, 0.7, 0.7, 0.7], dtype=np.float32)
JOINT_CONTROL_COST_WEIGHTS = np.array(
    [0.24, 0.24, 0.28, 0.28, 0.36, 0.36, 1.10, 1.10, 1.10, 1.10],
    dtype=np.float32,
)
GAIT_CYCLE_SECONDS = 1.08
WEIGHT_SHIFT_STAGE_START = 0.08
MARCH_STAGE_START = 0.28
WALK_STAGE_START = 0.50
LATERAL_SHIFT_TARGET = 0.022
COM_WAVE_AMPLITUDE = 0.011
COM_SQUARE_WAVE_LATERAL_SIGMA = 0.020
MARCH_STEP_TARGET = 0.030
WALK_STEP_TARGET_BASE = 0.040
WALK_STEP_TARGET_SPEED_GAIN = 0.060
WALK_STEP_TARGET_CURRICULUM_GAIN = 0.025
LANDING_TARGET_FORWARD_BASE = 0.030
LANDING_TARGET_FORWARD_GAIN = 0.65
LANDING_TARGET_LATERAL_TARGET = 0.125
MARCH_SWING_HEIGHT_TARGET = 0.020
WALK_SWING_HEIGHT_TARGET = 0.050
STEP_AIRTIME_TARGET = 0.38
STEP_AIRTIME_TARGET_GAIN = 0.04
STEP_AIRTIME_PHASE_SIGMA = 0.080
STEP_AIRTIME_WINDOW_SIGMA = 0.090
STEP_CLEARANCE_TARGET = 0.020
MIN_PROGRESS_TOUCHDOWN_FORWARD = 0.030
MIN_WALK_STEP_REWARD_RATIO = 0.48
BIG_STEP_BONUS_RATIO = 0.98
NO_TOUCHDOWN_TIMEOUT_SECONDS = 1.35
EARLY_TOUCHDOWN_AIRTIME_RATIO = 0.72
EARLY_TOUCHDOWN_PROGRESS_RATIO = 0.70
ABORTED_LIFTOFF_TIMEOUT_SECONDS = 0.42
ABORTED_LIFTOFF_FORWARD_THRESHOLD = 0.015
TOE_PATH_FORWARD_SIGMA = 0.055
TOE_PATH_LATERAL_SIGMA = 0.125
TOE_PATH_HEIGHT_SIGMA = 0.040
STANCE_FOOT_CLEARANCE_LIMIT = 0.040
SUPPORT_CLEARANCE_THRESHOLD = 0.045
FULL_TOE_CONTACT_THRESHOLD = 0.065
FULL_CONTACT_PHASE_MARGIN = 0.16
COMMAND_SPEED_SIGMA = 0.12
SAFE_TILT_FOR_FORWARD_DEG = 10.0
MIRROR_ACTION_SIGMA = 0.55
DELAYED_MIRROR_ACTION_SIGMA = 0.65
PHASE_MIRROR_ACTION_BLEND = 0.25
STEP_SYMMETRY_FORWARD_SIGMA = 0.035
STEP_SYMMETRY_AIRTIME_SIGMA = 0.070
MAX_STABLE_STEP_RATIO = 1.25
FORWARD_ACCEL_SIGMA = 0.22
BACKWARD_VELOCITY_SIGMA = 0.08
LATERAL_VELOCITY_SIGMA = 0.08
LATERAL_ACCEL_SIGMA = 0.18

JOINT_NAMES = [
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
    "left_leg_3_2_joint",
    "left_leg_3_3_joint",
    "right_leg_3_2_joint",
    "right_leg_3_3_joint",
]
EXPECTED_ACTUATORS = [
    "left_hip_torque",
    "right_hip_torque",
    "left_knee_torque",
    "right_knee_torque",
    "left_ankle_torque",
    "right_ankle_torque",
    "left_leg_3_2_torque",
    "left_leg_3_3_torque",
    "right_leg_3_2_torque",
    "right_leg_3_3_torque",
]


def resolve_xml(xml_file: str) -> Path:
    candidates = [
        Path(xml_file),
        Path.cwd() / xml_file,
        Path.cwd().parent / "project_2" / xml_file,
        Path.cwd().parent / "project_2_modified" / xml_file,
    ]
    for candidate in candidates:
        path = candidate.expanduser().resolve()
        if path.exists():
            return path
    raise FileNotFoundError(
        f"Could not find {xml_file}. Pass --xml-file with the path to robot_segment_trainable.xml."
    )


def explained_variance(y_pred: torch.Tensor, y_true: torch.Tensor) -> float:
    var_y = torch.var(y_true)
    if var_y.item() == 0.0:
        return float("nan")
    return float((1.0 - torch.var(y_true - y_pred) / var_y).detach().cpu())


class RunningMeanStd:
    def __init__(self, shape: tuple[int, ...], device: torch.device, eps: float = 1e-4) -> None:
        self.mean = torch.zeros(shape, device=device)
        self.var = torch.ones(shape, device=device)
        self.count = torch.tensor(eps, device=device)

    @torch.no_grad()
    def update(self, x: torch.Tensor) -> None:
        batch_mean = x.mean(dim=0)
        batch_var = x.var(dim=0, unbiased=False)
        batch_count = torch.tensor(float(x.shape[0]), device=x.device)

        delta = batch_mean - self.mean
        total_count = self.count + batch_count
        new_mean = self.mean + delta * batch_count / total_count

        m_a = self.var * self.count
        m_b = batch_var * batch_count
        m_2 = m_a + m_b + delta.square() * self.count * batch_count / total_count

        self.mean = new_mean
        self.var = m_2 / total_count
        self.count = total_count

    def normalize(self, x: torch.Tensor, clip: float = 10.0) -> torch.Tensor:
        return torch.clamp((x - self.mean) / torch.sqrt(self.var + 1e-8), -clip, clip)

    def state_dict(self) -> dict[str, torch.Tensor]:
        return {"mean": self.mean, "var": self.var, "count": self.count}

    def load_state_dict(self, state: dict[str, torch.Tensor]) -> None:
        self.mean.copy_(state["mean"].to(self.mean.device))
        self.var.copy_(state["var"].to(self.var.device))
        self.count.copy_(state["count"].to(self.count.device))


class ActorCritic(nn.Module):
    def __init__(self, obs_dim: int, action_dim: int) -> None:
        super().__init__()
        self.policy = nn.Sequential(
            nn.Linear(obs_dim, 400),
            nn.ReLU(),
            nn.Linear(400, 300),
            nn.ReLU(),
            nn.Linear(300, action_dim),
        )
        self.value = nn.Sequential(
            nn.Linear(obs_dim, 400),
            nn.ReLU(),
            nn.Linear(400, 300),
            nn.ReLU(),
            nn.Linear(300, 1),
        )
        self.log_std = nn.Parameter(torch.full((action_dim,), -1.2))

    def distribution(self, obs: torch.Tensor) -> Normal:
        mean = self.policy(obs)
        std = torch.exp(self.log_std).expand_as(mean)
        return Normal(mean, std)

    def act(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        dist = self.distribution(obs)
        raw_action = dist.rsample()
        log_prob = dist.log_prob(raw_action).sum(dim=-1)
        value = self.value(obs).squeeze(-1)
        return raw_action, log_prob, value

    def evaluate(self, obs: torch.Tensor, raw_action: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        dist = self.distribution(obs)
        log_prob = dist.log_prob(raw_action).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)
        value = self.value(obs).squeeze(-1)
        return log_prob, entropy, value


class MjWarpSegmentRunnerEnv:
    """Batched MJWarp environment for the segmented walker.

    This is intentionally smaller than the original Gymnasium environment: it keeps the
    same XML, actuator order, nominal pose, and PD action interpretation, but uses a
    simpler locomotion reward that can run over many GPU worlds.
    """

    def __init__(
        self,
        xml_path: Path,
        n_envs: int,
        device: torch.device,
        frame_skip: int,
        episode_length: int,
        target_forward_speed: float,
        reset_noise_scale: float,
        nconmax: int,
        njmax: int,
        use_cuda_graph: bool,
    ) -> None:
        try:
            import mujoco
            import mujoco_warp as mjw
            import warp as wp
        except ImportError as exc:
            raise ImportError(
                "This trainer requires mujoco, mujoco-warp, and warp-lang. "
                "Install those packages before running it."
            ) from exc

        self.mujoco = mujoco
        self.mjw = mjw
        self.wp = wp
        self.device = device
        self.n_envs = n_envs
        self.frame_skip = frame_skip
        self.episode_length = episode_length
        self.target_forward_speed = target_forward_speed
        self.curriculum_progress = 0.0
        self.reset_noise_scale = reset_noise_scale

        if device.type != "cuda":
            raise ValueError("MJWarp training should use a CUDA torch device, for example --device cuda:0.")
        wp.init()
        wp.set_device(str(device))

        self.mj_model = mujoco.MjModel.from_xml_path(str(xml_path))
        self.model = mjw.put_model(self.mj_model)
        self.data = mjw.make_data(self.mj_model, nworld=n_envs, nconmax=nconmax, njmax=njmax)

        self.qpos = wp.to_torch(self.data.qpos)
        self.qvel = wp.to_torch(self.data.qvel)
        self.ctrl = wp.to_torch(self.data.ctrl)
        self.qfrc_actuator = wp.to_torch(self.data.qfrc_actuator)
        self.xpos = wp.to_torch(self.data.xpos)
        self.geom_xpos = wp.to_torch(self.data.geom_xpos)

        action_names = [
            mujoco.mj_id2name(self.mj_model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
            for i in range(self.mj_model.nu)
        ]
        if action_names != EXPECTED_ACTUATORS:
            raise ValueError(f"Unexpected actuator order: {action_names}. Expected {EXPECTED_ACTUATORS}.")

        joint_ids = [
            mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_JOINT, name)
            for name in JOINT_NAMES
        ]
        self.joint_qpos_adr = torch.as_tensor(self.mj_model.jnt_qposadr[joint_ids], dtype=torch.long, device=device)
        self.joint_dof_adr = torch.as_tensor(self.mj_model.jnt_dofadr[joint_ids], dtype=torch.long, device=device)

        joint_bounds = torch.as_tensor(self.mj_model.jnt_range[joint_ids], dtype=torch.float32, device=device)
        self.joint_low = joint_bounds[:, 0]
        self.joint_high = joint_bounds[:, 1]

        ctrl_range = torch.as_tensor(self.mj_model.actuator_ctrlrange, dtype=torch.float32, device=device)
        self.ctrl_low = ctrl_range[:, 0]
        self.ctrl_high = ctrl_range[:, 1]

        self.nominal_joint_pose = torch.as_tensor(NOMINAL_JOINT_POSE, dtype=torch.float32, device=device)
        self.position_action_scale = torch.as_tensor(POSITION_ACTION_SCALE, dtype=torch.float32, device=device)
        self.pd_kp = torch.as_tensor(PD_KP, dtype=torch.float32, device=device)
        self.pd_kd = torch.as_tensor(PD_KD, dtype=torch.float32, device=device)
        self.control_cost_weights = torch.as_tensor(JOINT_CONTROL_COST_WEIGHTS, dtype=torch.float32, device=device)
        self.leg_mirror_weights = torch.as_tensor([1.0, 0.85, 0.85, 0.55, 0.55], dtype=torch.float32, device=device)

        self.torso_body_id = mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_BODY, "torso")
        self.left_toe_body_ids = [
            mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_BODY, "left_leg_3_2"),
            mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_BODY, "left_leg_3_3"),
        ]
        self.right_toe_body_ids = [
            mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_BODY, "right_leg_3_2"),
            mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_BODY, "right_leg_3_3"),
        ]
        self.left_toe_geom_ids = [
            mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_GEOM, "left_leg_3_2_collision"),
            mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_GEOM, "left_leg_3_3_collision"),
        ]
        self.right_toe_geom_ids = [
            mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_GEOM, "right_leg_3_2_collision"),
            mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_GEOM, "right_leg_3_3_collision"),
        ]
        nominal_data = mujoco.MjData(self.mj_model)
        nominal_data.qpos[:] = self.mj_model.qpos0
        nominal_data.qpos[self.mj_model.jnt_qposadr[joint_ids]] = NOMINAL_JOINT_POSE
        mujoco.mj_forward(self.mj_model, nominal_data)
        nominal_left_foot = nominal_data.xpos[self.left_toe_body_ids].mean(axis=0)
        nominal_right_foot = nominal_data.xpos[self.right_toe_body_ids].mean(axis=0)
        self.nominal_left_foot_height = torch.tensor(float(nominal_left_foot[2]), dtype=torch.float32, device=device)
        self.nominal_right_foot_height = torch.tensor(float(nominal_right_foot[2]), dtype=torch.float32, device=device)

        self.qpos0 = torch.as_tensor(self.mj_model.qpos0, dtype=torch.float32, device=device)
        self.qpos0[self.joint_qpos_adr] = self.nominal_joint_pose

        self.dt = float(self.mj_model.opt.timestep * frame_skip)
        self.prev_action = torch.zeros((n_envs, self.mj_model.nu), dtype=torch.float32, device=device)
        self.mirror_delay_steps = max(int(round(0.5 * GAIT_CYCLE_SECONDS / self.dt)), 1)
        self.action_history_len = self.mirror_delay_steps + 1
        self.action_history = torch.zeros(
            (self.action_history_len, n_envs, self.mj_model.nu),
            dtype=torch.float32,
            device=device,
        )
        self.action_history_index = 0
        self.prev_target = self.nominal_joint_pose.repeat(n_envs, 1)
        self.episode_step = torch.zeros(n_envs, dtype=torch.long, device=device)
        self.prev_x = torch.zeros(n_envs, dtype=torch.float32, device=device)
        self.prev_y = torch.zeros(n_envs, dtype=torch.float32, device=device)
        self.prev_lateral_velocity = torch.zeros(n_envs, dtype=torch.float32, device=device)
        self.prev_forward_velocity = torch.zeros(n_envs, dtype=torch.float32, device=device)
        self.spawn_xy = torch.zeros((n_envs, 2), dtype=torch.float32, device=device)
        self.prev_left_foot = torch.zeros((n_envs, 3), dtype=torch.float32, device=device)
        self.prev_right_foot = torch.zeros((n_envs, 3), dtype=torch.float32, device=device)
        self.prev_left_supporting = torch.ones(n_envs, dtype=torch.bool, device=device)
        self.prev_right_supporting = torch.ones(n_envs, dtype=torch.bool, device=device)
        self.left_swing_start = torch.zeros((n_envs, 3), dtype=torch.float32, device=device)
        self.right_swing_start = torch.zeros((n_envs, 3), dtype=torch.float32, device=device)
        self.left_swing_airtime = torch.zeros(n_envs, dtype=torch.float32, device=device)
        self.right_swing_airtime = torch.zeros(n_envs, dtype=torch.float32, device=device)
        self.left_swing_max_clearance = torch.zeros(n_envs, dtype=torch.float32, device=device)
        self.right_swing_max_clearance = torch.zeros(n_envs, dtype=torch.float32, device=device)
        self.left_abort_penalized = torch.zeros(n_envs, dtype=torch.bool, device=device)
        self.right_abort_penalized = torch.zeros(n_envs, dtype=torch.bool, device=device)
        self.steps_since_valid_touchdown = torch.zeros(n_envs, dtype=torch.long, device=device)
        self.last_touchdown_side = torch.zeros(n_envs, dtype=torch.long, device=device)
        self.last_left_step_forward = torch.zeros(n_envs, dtype=torch.float32, device=device)
        self.last_right_step_forward = torch.zeros(n_envs, dtype=torch.float32, device=device)
        self.last_left_step_airtime = torch.zeros(n_envs, dtype=torch.float32, device=device)
        self.last_right_step_airtime = torch.zeros(n_envs, dtype=torch.float32, device=device)
        self.last_left_touchdown_valid = torch.zeros(n_envs, dtype=torch.bool, device=device)
        self.last_right_touchdown_valid = torch.zeros(n_envs, dtype=torch.bool, device=device)
        self.episode_return = torch.zeros(n_envs, dtype=torch.float32, device=device)
        self.episode_length_buf = torch.zeros(n_envs, dtype=torch.float32, device=device)
        self.action_dim = int(self.mj_model.nu)
        self.obs_dim = 7 + self.mj_model.nv + 10 + 10 + 10 + 4

        self.step_graph = None
        self.reset(torch.ones(n_envs, dtype=torch.bool, device=device))
        if use_cuda_graph:
            with wp.ScopedCapture() as capture:
                for _ in range(self.frame_skip):
                    mjw.step(self.model, self.data)
            self.step_graph = capture.graph

    def _foot_center(self, body_ids: list[int]) -> torch.Tensor:
        return self.xpos[:, body_ids, :].mean(dim=1)

    def _toe_floor_clearance(self, geom_ids: list[int]) -> torch.Tensor:
        return torch.clamp(self.geom_xpos[:, geom_ids, 2].min(dim=1).values, min=0.0)

    def _toe_floor_clearances(self, geom_ids: list[int]) -> torch.Tensor:
        return torch.clamp(self.geom_xpos[:, geom_ids, 2], min=0.0)

    @staticmethod
    def _toe_wave_reference(phase: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        swing_mask = (phase < 0.5).float()
        swing_progress = torch.clamp(phase / 0.5, 0.0, 1.0)
        forward_profile = (0.5 - 0.5 * torch.cos(math.pi * swing_progress)) * swing_mask
        height_profile = torch.sin(math.pi * swing_progress).clamp_min(0.0) * swing_mask
        return swing_mask, swing_progress, forward_profile, height_profile

    def set_curriculum(self, progress: float) -> None:
        self.curriculum_progress = float(np.clip(progress, 0.0, 1.0))

    def _curriculum_targets(self) -> tuple[float, float, float, float, float, float, float]:
        progress = self.curriculum_progress
        weight_shift_scale = float(
            np.clip(
                (progress - WEIGHT_SHIFT_STAGE_START) / max(MARCH_STAGE_START - WEIGHT_SHIFT_STAGE_START, 1e-6),
                0.0,
                1.0,
            )
        )
        march_phase = float(
            np.clip((progress - MARCH_STAGE_START) / max(WALK_STAGE_START - MARCH_STAGE_START, 1e-6), 0.0, 1.0)
        )
        walk_scale = float(np.clip((progress - WALK_STAGE_START) / max(1.0 - WALK_STAGE_START, 1e-6), 0.0, 1.0))
        march_scale = march_phase * (1.0 - walk_scale)
        weight_shift_scale *= 1.0 - march_phase
        stand_scale = float(np.clip(1.0 - weight_shift_scale - march_scale - walk_scale, 0.10, 1.0))
        locomotion_scale = float(np.clip(0.35 * march_scale + walk_scale, 0.0, 1.0))
        target_speed = float(
            np.interp(locomotion_scale, [0.0, 0.25, 0.50, 0.75, 1.0], [0.0, 0.02, 0.05, 0.08, self.target_forward_speed])
        )
        gait_scale = max(weight_shift_scale, march_scale, walk_scale)
        return weight_shift_scale, march_scale, walk_scale, locomotion_scale, stand_scale, target_speed, gait_scale

    @torch.no_grad()
    def reset(self, mask: torch.Tensor | None = None) -> torch.Tensor:
        if mask is None:
            mask = torch.ones(self.n_envs, dtype=torch.bool, device=self.device)
        ids = torch.nonzero(mask, as_tuple=False).squeeze(-1)
        if ids.numel() == 0:
            return self.obs()

        qpos = self.qpos0.repeat(ids.numel(), 1)
        qvel = torch.zeros((ids.numel(), self.mj_model.nv), dtype=torch.float32, device=self.device)

        qpos[:, 0] += torch.empty(ids.numel(), device=self.device).uniform_(-0.015, 0.015)
        qpos[:, 1] += torch.empty(ids.numel(), device=self.device).uniform_(-0.015, 0.015)
        qpos[:, self.joint_qpos_adr] += self.reset_noise_scale * torch.randn(
            (ids.numel(), self.action_dim), device=self.device
        )
        qpos[:, self.joint_qpos_adr] = torch.max(
            torch.min(qpos[:, self.joint_qpos_adr], self.joint_high),
            self.joint_low,
        )

        self.qpos[ids] = qpos
        self.qvel[ids] = qvel
        self.ctrl[ids] = 0.0
        self.prev_action[ids] = 0.0
        self.action_history[:, ids, :] = 0.0
        self.prev_target[ids] = self.nominal_joint_pose
        self.episode_step[ids] = 0
        self.mjw.forward(self.model, self.data)
        self.prev_x[ids] = self.qpos[ids, 0]
        self.prev_y[ids] = self.qpos[ids, 1]
        self.prev_lateral_velocity[ids] = 0.0
        self.prev_forward_velocity[ids] = 0.0
        self.spawn_xy[ids] = self.qpos[ids, :2]
        left_foot = self._foot_center(self.left_toe_body_ids)
        right_foot = self._foot_center(self.right_toe_body_ids)
        left_supporting = self._toe_floor_clearance(self.left_toe_geom_ids) <= SUPPORT_CLEARANCE_THRESHOLD
        right_supporting = self._toe_floor_clearance(self.right_toe_geom_ids) <= SUPPORT_CLEARANCE_THRESHOLD
        self.prev_left_foot[ids] = left_foot[ids]
        self.prev_right_foot[ids] = right_foot[ids]
        self.prev_left_supporting[ids] = left_supporting[ids]
        self.prev_right_supporting[ids] = right_supporting[ids]
        self.left_swing_start[ids] = left_foot[ids]
        self.right_swing_start[ids] = right_foot[ids]
        self.left_swing_airtime[ids] = 0.0
        self.right_swing_airtime[ids] = 0.0
        self.left_swing_max_clearance[ids] = 0.0
        self.right_swing_max_clearance[ids] = 0.0
        self.left_abort_penalized[ids] = False
        self.right_abort_penalized[ids] = False
        self.steps_since_valid_touchdown[ids] = 0
        self.last_touchdown_side[ids] = 0
        self.last_left_step_forward[ids] = 0.0
        self.last_right_step_forward[ids] = 0.0
        self.last_left_step_airtime[ids] = 0.0
        self.last_right_step_airtime[ids] = 0.0
        self.last_left_touchdown_valid[ids] = False
        self.last_right_touchdown_valid[ids] = False
        self.episode_return[ids] = 0.0
        self.episode_length_buf[ids] = 0.0
        return self.obs()

    def _upright(self) -> torch.Tensor:
        quat = self.qpos[:, 3:7]
        norm = torch.linalg.norm(quat, dim=-1).clamp_min(1e-6)
        quat = quat / norm[:, None]
        x = quat[:, 1]
        y = quat[:, 2]
        return 1.0 - 2.0 * (x.square() + y.square())

    def obs(self) -> torch.Tensor:
        joint_pos = self.qpos[:, self.joint_qpos_adr]
        joint_vel = self.qvel[:, self.joint_dof_adr]
        _, march_scale, walk_scale, _, _, target_speed, _ = self._curriculum_targets()
        phase = 2.0 * math.pi * (self.episode_step.float() * self.dt / GAIT_CYCLE_SECONDS)
        extras = torch.stack(
            [
                torch.sin(phase),
                torch.cos(phase),
                torch.full_like(phase, target_speed),
                torch.full_like(phase, 0.35 * march_scale + walk_scale),
            ],
            dim=-1,
        )
        return torch.cat(
            [
                self.qpos[:, :7],
                torch.clamp(self.qvel, -25.0, 25.0),
                joint_pos,
                torch.clamp(joint_vel, -25.0, 25.0),
                self.prev_action,
                extras,
            ],
            dim=-1,
        )

    @torch.no_grad()
    def step(self, raw_action: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, float]]:
        action = torch.clamp(raw_action, -1.0, 1.0)
        filtered_action = 0.20 * self.prev_action + 0.80 * action
        delayed_action = self.action_history[self.action_history_index].clone()

        control_cycle = torch.remainder((self.episode_step.float() + 1.0) * self.dt / GAIT_CYCLE_SECONDS, 1.0)
        control_left_expected_swing = (control_cycle < 0.5).float()
        control_right_expected_swing = 1.0 - control_left_expected_swing
        history_ready = (self.episode_step >= self.mirror_delay_steps).float()
        mirror_blend = (PHASE_MIRROR_ACTION_BLEND * history_ready)[:, None]
        left_now_control = filtered_action[:, [0, 2, 4, 6, 7]]
        right_now_control = filtered_action[:, [1, 3, 5, 8, 9]]
        delayed_left_control = delayed_action[:, [0, 2, 4, 6, 7]]
        delayed_right_control = delayed_action[:, [1, 3, 5, 8, 9]]
        left_from_delayed_right = torch.stack(
            [
                -delayed_right_control[:, 0],
                delayed_right_control[:, 1],
                delayed_right_control[:, 2],
                delayed_right_control[:, 3],
                delayed_right_control[:, 4],
            ],
            dim=-1,
        )
        right_from_delayed_left = torch.stack(
            [
                -delayed_left_control[:, 0],
                delayed_left_control[:, 1],
                delayed_left_control[:, 2],
                delayed_left_control[:, 3],
                delayed_left_control[:, 4],
            ],
            dim=-1,
        )
        projected_left = torch.where(
            control_left_expected_swing[:, None] > 0.5,
            left_from_delayed_right,
            left_now_control,
        )
        projected_right = torch.where(
            control_right_expected_swing[:, None] > 0.5,
            right_from_delayed_left,
            right_now_control,
        )
        projected_action = filtered_action.clone()
        projected_action[:, [0, 2, 4, 6, 7]] = (1.0 - mirror_blend) * left_now_control + mirror_blend * projected_left
        projected_action[:, [1, 3, 5, 8, 9]] = (1.0 - mirror_blend) * right_now_control + mirror_blend * projected_right
        filtered_action = projected_action

        joint_pos = self.qpos[:, self.joint_qpos_adr]
        joint_vel = self.qvel[:, self.joint_dof_adr]
        raw_target = self.nominal_joint_pose + filtered_action * self.position_action_scale
        raw_target = torch.max(torch.min(raw_target, self.joint_high), self.joint_low)
        target = 0.40 * self.prev_target + 0.60 * raw_target
        target = torch.max(torch.min(target, self.joint_high), self.joint_low)

        pd_ctrl = self.pd_kp * (target - joint_pos) - self.pd_kd * joint_vel
        self.ctrl[:] = torch.max(torch.min(pd_ctrl, self.ctrl_high), self.ctrl_low)

        if self.step_graph is not None:
            self.wp.capture_launch(self.step_graph)
        else:
            for _ in range(self.frame_skip):
                self.mjw.step(self.model, self.data)

        self.episode_step += 1
        self.episode_length_buf += 1.0

        current_x = self.qpos[:, 0].clone()
        current_y = self.qpos[:, 1].clone()
        lateral_velocity = (current_x - self.prev_x) / self.dt
        lateral_acceleration = (lateral_velocity - self.prev_lateral_velocity) / self.dt
        forward_velocity = (self.prev_y - current_y) / self.dt
        forward_acceleration = (forward_velocity - self.prev_forward_velocity) / self.dt
        self.prev_x = current_x
        self.prev_y = current_y

        height = self.qpos[:, 2]
        upright = self._upright()
        joint_pos_after = self.qpos[:, self.joint_qpos_adr]
        torso_xy = self.qpos[:, :2]
        torso_forward_progress = -(torso_xy[:, 1] - self.spawn_xy[:, 1])
        left_foot = self._foot_center(self.left_toe_body_ids)
        right_foot = self._foot_center(self.right_toe_body_ids)
        left_floor_clearance = self._toe_floor_clearance(self.left_toe_geom_ids)
        right_floor_clearance = self._toe_floor_clearance(self.right_toe_geom_ids)
        left_toe_clearances = self._toe_floor_clearances(self.left_toe_geom_ids)
        right_toe_clearances = self._toe_floor_clearances(self.right_toe_geom_ids)
        weight_shift_scale, march_scale, walk_scale, locomotion_scale, stand_scale, target_forward_speed, gait_scale = (
            self._curriculum_targets()
        )

        cycle = torch.remainder(self.episode_step.float() * self.dt / GAIT_CYCLE_SECONDS, 1.0)
        left_toe_phase = cycle
        right_toe_phase = torch.remainder(cycle + 0.5, 1.0)
        left_expected_swing, left_swing_progress, left_forward_profile, left_height_profile = self._toe_wave_reference(left_toe_phase)
        right_expected_swing, right_swing_progress, right_forward_profile, right_height_profile = self._toe_wave_reference(right_toe_phase)
        active_swing_progress = torch.where(left_expected_swing > 0.5, left_swing_progress, right_swing_progress)
        target_speed_ratio = target_forward_speed / max(self.target_forward_speed, 1e-6)
        target_step_length = (
            march_scale * MARCH_STEP_TARGET
            + walk_scale
            * (
                WALK_STEP_TARGET_BASE
                + WALK_STEP_TARGET_CURRICULUM_GAIN * (0.30 + 0.70 * walk_scale)
                + WALK_STEP_TARGET_SPEED_GAIN * (0.20 + 0.80 * target_speed_ratio)
            )
        )
        target_clearance = march_scale * MARCH_SWING_HEIGHT_TARGET + walk_scale * WALK_SWING_HEIGHT_TARGET
        target_airtime = (
            march_scale * (0.65 * STEP_AIRTIME_TARGET)
            + walk_scale
            * (
                STEP_AIRTIME_TARGET
                + STEP_AIRTIME_TARGET_GAIN * float(np.clip(target_step_length / max(WALK_STEP_TARGET_BASE, 1e-6) - 1.0, 0.0, 1.5))
            )
        )
        target_forward_offset = LANDING_TARGET_FORWARD_BASE + LANDING_TARGET_FORWARD_GAIN * target_step_length
        left_desired_forward_offset = target_forward_offset * left_forward_profile
        right_desired_forward_offset = target_forward_offset * right_forward_profile
        left_desired_swing_height = target_clearance * left_height_profile
        right_desired_swing_height = target_clearance * right_height_profile

        left_forward_offset = -(left_foot[:, 1] - torso_xy[:, 1])
        right_forward_offset = -(right_foot[:, 1] - torso_xy[:, 1])
        left_lateral_offset = -(left_foot[:, 0] - torso_xy[:, 0])
        right_lateral_offset = -(right_foot[:, 0] - torso_xy[:, 0])
        left_height_error = left_foot[:, 2] - (self.nominal_left_foot_height + left_desired_swing_height)
        right_height_error = right_foot[:, 2] - (self.nominal_right_foot_height + right_desired_swing_height)

        left_path_error_sq = (
            ((left_forward_offset - left_desired_forward_offset) / TOE_PATH_FORWARD_SIGMA).square()
            + ((left_lateral_offset - LANDING_TARGET_LATERAL_TARGET) / TOE_PATH_LATERAL_SIGMA).square()
            + (left_height_error / TOE_PATH_HEIGHT_SIGMA).square()
        )
        right_path_error_sq = (
            ((right_forward_offset - right_desired_forward_offset) / TOE_PATH_FORWARD_SIGMA).square()
            + ((right_lateral_offset + LANDING_TARGET_LATERAL_TARGET) / TOE_PATH_LATERAL_SIGMA).square()
            + (right_height_error / TOE_PATH_HEIGHT_SIGMA).square()
        )
        toe_path_reward = (
            left_expected_swing * torch.exp(-0.5 * left_path_error_sq)
            + right_expected_swing * torch.exp(-0.5 * right_path_error_sq)
        )
        toe_path_cost = (
            left_expected_swing * torch.clamp(left_path_error_sq, 0.0, 6.0)
            + right_expected_swing * torch.clamp(right_path_error_sq, 0.0, 6.0)
        )

        com_wave_profile = left_expected_swing * left_forward_profile + right_expected_swing * right_forward_profile
        support_side_sign = torch.where(left_expected_swing > 0.5, -1.0, 1.0)
        desired_com_lateral = support_side_sign * COM_WAVE_AMPLITUDE * com_wave_profile
        com_lateral_offset = -(torso_xy[:, 0] - self.spawn_xy[:, 0])
        com_square_wave_cost = torch.clamp(
            ((com_lateral_offset - desired_com_lateral) / COM_SQUARE_WAVE_LATERAL_SIGMA).square(),
            0.0,
            6.0,
        )

        left_supporting = left_floor_clearance <= SUPPORT_CLEARANCE_THRESHOLD
        right_supporting = right_floor_clearance <= SUPPORT_CLEARANCE_THRESHOLD
        left_full_contact = (left_toe_clearances <= FULL_TOE_CONTACT_THRESHOLD).all(dim=1)
        right_full_contact = (right_toe_clearances <= FULL_TOE_CONTACT_THRESHOLD).all(dim=1)
        left_full_contact_score = torch.exp(
            -0.5 * (left_toe_clearances / max(float(FULL_TOE_CONTACT_THRESHOLD), 1e-6)).square().mean(dim=1)
        )
        right_full_contact_score = torch.exp(
            -0.5 * (right_toe_clearances / max(float(FULL_TOE_CONTACT_THRESHOLD), 1e-6)).square().mean(dim=1)
        )
        left_liftoff = self.prev_left_supporting & (~left_supporting) & right_supporting
        right_liftoff = self.prev_right_supporting & (~right_supporting) & left_supporting
        left_touchdown = (~self.prev_left_supporting) & left_supporting & right_supporting
        right_touchdown = (~self.prev_right_supporting) & right_supporting & left_supporting

        left_swing_start = torch.where(left_liftoff[:, None], self.prev_left_foot, self.left_swing_start)
        right_swing_start = torch.where(right_liftoff[:, None], self.prev_right_foot, self.right_swing_start)
        left_swing_airtime = torch.where(left_liftoff, torch.zeros_like(self.left_swing_airtime), self.left_swing_airtime)
        right_swing_airtime = torch.where(right_liftoff, torch.zeros_like(self.right_swing_airtime), self.right_swing_airtime)
        left_swing_max_clearance = torch.where(
            left_liftoff,
            torch.zeros_like(self.left_swing_max_clearance),
            self.left_swing_max_clearance,
        )
        right_swing_max_clearance = torch.where(
            right_liftoff,
            torch.zeros_like(self.right_swing_max_clearance),
            self.right_swing_max_clearance,
        )

        left_clearance = torch.clamp(left_foot[:, 2] - right_foot[:, 2], min=0.0)
        right_clearance = torch.clamp(right_foot[:, 2] - left_foot[:, 2], min=0.0)
        left_in_air = ~left_supporting
        right_in_air = ~right_supporting
        left_swing_airtime = torch.where(left_in_air, left_swing_airtime + self.dt, left_swing_airtime)
        right_swing_airtime = torch.where(right_in_air, right_swing_airtime + self.dt, right_swing_airtime)
        left_swing_max_clearance = torch.where(
            left_in_air,
            torch.maximum(left_swing_max_clearance, left_clearance),
            left_swing_max_clearance,
        )
        right_swing_max_clearance = torch.where(
            right_in_air,
            torch.maximum(right_swing_max_clearance, right_clearance),
            right_swing_max_clearance,
        )

        left_swing_forward = torch.clamp(-(left_foot[:, 1] - left_swing_start[:, 1]), min=0.0)
        right_swing_forward = torch.clamp(-(right_foot[:, 1] - right_swing_start[:, 1]), min=0.0)
        left_touchdown_forward = torch.where(left_touchdown, left_swing_forward, torch.zeros_like(left_swing_forward))
        right_touchdown_forward = torch.where(right_touchdown, right_swing_forward, torch.zeros_like(right_swing_forward))
        left_touchdown_airtime = torch.where(left_touchdown, left_swing_airtime, torch.zeros_like(left_swing_airtime))
        right_touchdown_airtime = torch.where(right_touchdown, right_swing_airtime, torch.zeros_like(right_swing_airtime))
        left_touchdown_clearance = torch.where(left_touchdown, left_swing_max_clearance, torch.zeros_like(left_swing_max_clearance))
        right_touchdown_clearance = torch.where(right_touchdown, right_swing_max_clearance, torch.zeros_like(right_swing_max_clearance))
        touchdown_balance = torch.clamp((upright - math.cos(math.radians(20.0))) / (1.0 - math.cos(math.radians(20.0))), 0.0, 1.0)
        left_touchdown_quality = (
            torch.clamp(left_touchdown_clearance / STEP_CLEARANCE_TARGET, 0.0, 1.0)
            * left_full_contact.float()
            * touchdown_balance
        )
        right_touchdown_quality = (
            torch.clamp(right_touchdown_clearance / STEP_CLEARANCE_TARGET, 0.0, 1.0)
            * right_full_contact.float()
            * touchdown_balance
        )
        left_step_ratio = torch.clamp(left_touchdown_forward / max(float(target_forward_offset), 1e-6), 0.0, 1.8)
        right_step_ratio = torch.clamp(right_touchdown_forward / max(float(target_forward_offset), 1e-6), 0.0, 1.8)
        left_valid_touchdown = (
            left_touchdown
            & (left_expected_swing > 0.5)
            & (left_touchdown_forward >= MIN_PROGRESS_TOUCHDOWN_FORWARD)
            & (left_touchdown_clearance >= STEP_CLEARANCE_TARGET)
            & left_full_contact
        )
        right_valid_touchdown = (
            right_touchdown
            & (right_expected_swing > 0.5)
            & (right_touchdown_forward >= MIN_PROGRESS_TOUCHDOWN_FORWARD)
            & (right_touchdown_clearance >= STEP_CLEARANCE_TARGET)
            & right_full_contact
        )
        any_valid_touchdown = left_valid_touchdown | right_valid_touchdown
        no_touchdown_time = self.steps_since_valid_touchdown.float() * self.dt

        left_stance = 1.0 - left_expected_swing
        right_stance = 1.0 - right_expected_swing
        stance_clearance_cost = (
            left_stance * torch.relu(left_floor_clearance - STANCE_FOOT_CLEARANCE_LIMIT).square()
            + right_stance * torch.relu(right_floor_clearance - STANCE_FOOT_CLEARANCE_LIMIT).square()
        ) / (STANCE_FOOT_CLEARANCE_LIMIT * STANCE_FOOT_CLEARANCE_LIMIT)
        full_toe_stance_contact_cost = (
            left_stance * (1.0 - left_full_contact_score)
            + right_stance * (1.0 - right_full_contact_score)
        )
        both_airborne_cost = (
            (left_floor_clearance > STANCE_FOOT_CLEARANCE_LIMIT).float()
            * (right_floor_clearance > STANCE_FOOT_CLEARANCE_LIMIT).float()
        )
        left_foot_velocity = (left_foot - self.prev_left_foot) / self.dt
        right_foot_velocity = (right_foot - self.prev_right_foot) / self.dt
        foot_slip_cost = (
            left_stance * torch.linalg.norm(left_foot_velocity[:, :2], dim=-1)
            + right_stance * torch.linalg.norm(right_foot_velocity[:, :2], dim=-1)
        )
        vertical_velocity_cost = torch.clamp(self.qvel[:, 2].square(), 0.0, 4.0)
        roll_pitch_rate_cost = torch.clamp(self.qvel[:, 3].square() + self.qvel[:, 4].square(), 0.0, 6.0)

        left_airtime_phase_score = (
            torch.exp(-0.5 * ((left_swing_airtime - target_airtime) / STEP_AIRTIME_PHASE_SIGMA).square())
            * left_in_air.float()
            * right_supporting.float()
            * left_expected_swing
            * touchdown_balance
        )
        right_airtime_phase_score = (
            torch.exp(-0.5 * ((right_swing_airtime - target_airtime) / STEP_AIRTIME_PHASE_SIGMA).square())
            * right_in_air.float()
            * left_supporting.float()
            * right_expected_swing
            * touchdown_balance
        )
        left_airtime_window_score = (
            torch.exp(-0.5 * ((left_touchdown_airtime - target_airtime) / STEP_AIRTIME_WINDOW_SIGMA).square())
            * left_touchdown.float()
            * left_expected_swing
            * touchdown_balance
        )
        right_airtime_window_score = (
            torch.exp(-0.5 * ((right_touchdown_airtime - target_airtime) / STEP_AIRTIME_WINDOW_SIGMA).square())
            * right_touchdown.float()
            * right_expected_swing
            * touchdown_balance
        )
        left_landing_target_score = torch.exp(
            -0.5
            * (
                ((left_forward_offset - target_forward_offset) / TOE_PATH_FORWARD_SIGMA).square()
                + ((left_lateral_offset - LANDING_TARGET_LATERAL_TARGET) / TOE_PATH_LATERAL_SIGMA).square()
            )
        )
        right_landing_target_score = torch.exp(
            -0.5
            * (
                ((right_forward_offset - target_forward_offset) / TOE_PATH_FORWARD_SIGMA).square()
                + ((right_lateral_offset + LANDING_TARGET_LATERAL_TARGET) / TOE_PATH_LATERAL_SIGMA).square()
            )
        )
        base_velocity_gate = torch.clamp(forward_velocity / max(float(target_forward_speed), 0.05), 0.0, 1.25)
        cumulative_progress_gate = torch.clamp(
            torso_forward_progress / torch.clamp(0.35 * self.episode_length_buf.float() * self.dt * max(float(target_forward_speed), 0.05), min=0.03),
            0.0,
            1.25,
        )
        base_progress_gate = torch.clamp(0.25 * base_velocity_gate + 0.75 * cumulative_progress_gate, 0.0, 1.25)
        progress_tracking_reward = walk_scale * 0.22 * torch.exp(
            -0.5 * ((forward_velocity - target_forward_speed) / FORWARD_PROGRESS_SIGMA).square()
        ) * touchdown_balance
        positive_progress_reward = walk_scale * 0.10 * torch.clamp(forward_velocity, 0.0, target_forward_speed + 0.04)
        stepping_in_place_penalty = walk_scale * 0.75 * (
            left_valid_touchdown.float() * torch.relu(0.75 - base_progress_gate)
            + right_valid_touchdown.float() * torch.relu(0.75 - base_progress_gate)
        )
        landing_target_reward = walk_scale * 1.55 * (
            left_landing_target_score * left_valid_touchdown.float() * left_touchdown_quality
            + right_landing_target_score * right_valid_touchdown.float() * right_touchdown_quality
        )
        walk_touchdown_reward = walk_scale * 1.20 * (
            left_step_ratio * left_valid_touchdown.float() * left_touchdown_quality
            + right_step_ratio * right_valid_touchdown.float() * right_touchdown_quality
        )
        big_step_bonus = walk_scale * 0.10 * (
            torch.relu((left_step_ratio - BIG_STEP_BONUS_RATIO) / max(1.0 - BIG_STEP_BONUS_RATIO, 1e-6)).square()
            * left_valid_touchdown.float()
            * left_touchdown_quality
            + torch.relu((right_step_ratio - BIG_STEP_BONUS_RATIO) / max(1.0 - BIG_STEP_BONUS_RATIO, 1e-6)).square()
            * right_valid_touchdown.float()
            * right_touchdown_quality
        ) * torch.clamp(base_progress_gate - 0.20, 0.0, 1.0)
        small_step_penalty = walk_scale * 0.45 * (
            torch.relu(MIN_WALK_STEP_REWARD_RATIO - left_step_ratio) * left_touchdown.float() * left_expected_swing
            + torch.relu(MIN_WALK_STEP_REWARD_RATIO - right_step_ratio) * right_touchdown.float() * right_expected_swing
        )
        no_touchdown_penalty = walk_scale * 0.35 * torch.clamp(
            (no_touchdown_time - NO_TOUCHDOWN_TIMEOUT_SECONDS) / max(NO_TOUCHDOWN_TIMEOUT_SECONDS, 1e-6),
            0.0,
            1.5,
        )
        early_touchdown_penalty = walk_scale * 0.45 * (
            (
                torch.relu(EARLY_TOUCHDOWN_AIRTIME_RATIO * target_airtime - left_touchdown_airtime)
                / max(float(EARLY_TOUCHDOWN_AIRTIME_RATIO * target_airtime), 1e-6)
                + torch.relu(EARLY_TOUCHDOWN_PROGRESS_RATIO - left_step_ratio)
            )
            * left_touchdown.float()
            * left_expected_swing
            + (
                torch.relu(EARLY_TOUCHDOWN_AIRTIME_RATIO * target_airtime - right_touchdown_airtime)
                / max(float(EARLY_TOUCHDOWN_AIRTIME_RATIO * target_airtime), 1e-6)
                + torch.relu(EARLY_TOUCHDOWN_PROGRESS_RATIO - right_step_ratio)
            )
            * right_touchdown.float()
            * right_expected_swing
        )
        left_aborted_liftoff = (
            left_in_air
            & (left_swing_airtime >= ABORTED_LIFTOFF_TIMEOUT_SECONDS)
            & (left_swing_forward < ABORTED_LIFTOFF_FORWARD_THRESHOLD)
            & (~self.left_abort_penalized)
        )
        right_aborted_liftoff = (
            right_in_air
            & (right_swing_airtime >= ABORTED_LIFTOFF_TIMEOUT_SECONDS)
            & (right_swing_forward < ABORTED_LIFTOFF_FORWARD_THRESHOLD)
            & (~self.right_abort_penalized)
        )
        aborted_liftoff_penalty = gait_scale * 0.60 * (left_aborted_liftoff.float() + right_aborted_liftoff.float())
        alternation_reward = walk_scale * 0.25 * (
            left_valid_touchdown.float() * (self.last_touchdown_side == 2).float()
            + right_valid_touchdown.float() * (self.last_touchdown_side == 1).float()
        )
        left_pair_known = left_valid_touchdown & self.last_right_touchdown_valid
        right_pair_known = right_valid_touchdown & self.last_left_touchdown_valid
        left_step_symmetry_cost = torch.where(
            left_pair_known,
            ((left_touchdown_forward - self.last_right_step_forward) / STEP_SYMMETRY_FORWARD_SIGMA).square()
            + ((left_touchdown_airtime - self.last_right_step_airtime) / STEP_SYMMETRY_AIRTIME_SIGMA).square(),
            torch.zeros_like(left_touchdown_forward),
        )
        right_step_symmetry_cost = torch.where(
            right_pair_known,
            ((right_touchdown_forward - self.last_left_step_forward) / STEP_SYMMETRY_FORWARD_SIGMA).square()
            + ((right_touchdown_airtime - self.last_left_step_airtime) / STEP_SYMMETRY_AIRTIME_SIGMA).square(),
            torch.zeros_like(right_touchdown_forward),
        )
        balanced_step_reward = walk_scale * 0.50 * (
            left_pair_known.float() * torch.exp(-0.5 * left_step_symmetry_cost)
            + right_pair_known.float() * torch.exp(-0.5 * right_step_symmetry_cost)
        )
        step_symmetry_penalty = walk_scale * 0.38 * (
            left_pair_known.float() * torch.clamp(left_step_symmetry_cost, 0.0, 5.0)
            + right_pair_known.float() * torch.clamp(right_step_symmetry_cost, 0.0, 5.0)
        )
        oversize_step_penalty = walk_scale * 0.55 * (
            torch.relu(left_step_ratio - MAX_STABLE_STEP_RATIO).square() * left_valid_touchdown.float()
            + torch.relu(right_step_ratio - MAX_STABLE_STEP_RATIO).square() * right_valid_touchdown.float()
        )

        left_now = filtered_action[:, [0, 2, 4, 6, 7]]
        right_now = filtered_action[:, [1, 3, 5, 8, 9]]
        right_now_mirrored = torch.stack(
            [-right_now[:, 0], right_now[:, 1], right_now[:, 2], right_now[:, 3], right_now[:, 4]],
            dim=-1,
        )
        delayed_left = delayed_action[:, [0, 2, 4, 6, 7]]
        delayed_right = delayed_action[:, [1, 3, 5, 8, 9]]
        delayed_left_mirrored = torch.stack(
            [-delayed_left[:, 0], delayed_left[:, 1], delayed_left[:, 2], delayed_left[:, 3], delayed_left[:, 4]],
            dim=-1,
        )
        delayed_right_mirrored = torch.stack(
            [-delayed_right[:, 0], delayed_right[:, 1], delayed_right[:, 2], delayed_right[:, 3], delayed_right[:, 4]],
            dim=-1,
        )
        current_mirror_cost = (((left_now - right_now_mirrored) * self.leg_mirror_weights) / MIRROR_ACTION_SIGMA).square().mean(dim=-1)
        delayed_mirror_cost = (
            left_expected_swing
            * (((left_now - delayed_right_mirrored) * self.leg_mirror_weights) / DELAYED_MIRROR_ACTION_SIGMA).square().mean(dim=-1)
            + right_expected_swing
            * (((right_now - delayed_left_mirrored) * self.leg_mirror_weights) / DELAYED_MIRROR_ACTION_SIGMA).square().mean(dim=-1)
        )
        phase_mirror_reward = walk_scale * 0.22 * torch.exp(-0.5 * delayed_mirror_cost) * touchdown_balance
        mirror_action_penalty = walk_scale * (
            0.035 * torch.clamp(current_mirror_cost, 0.0, 6.0)
            + 0.055 * torch.clamp(delayed_mirror_cost, 0.0, 6.0)
        )
        early_step_contact_gate = torch.clamp((active_swing_progress - FULL_CONTACT_PHASE_MARGIN) / 0.18, 0.0, 1.0)
        mid_step_contact_gate = torch.clamp((active_swing_progress - 0.18) / 0.20, 0.0, 1.0) * torch.clamp(
            (1.0 - FULL_CONTACT_PHASE_MARGIN - active_swing_progress) / 0.20,
            0.0,
            1.0,
        )
        transition_contact_gate = (
            (active_swing_progress < FULL_CONTACT_PHASE_MARGIN)
            | (active_swing_progress > 1.0 - FULL_CONTACT_PHASE_MARGIN)
        ).float()
        liftoff_reward = gait_scale * 0.35 * (
            left_liftoff.float() * left_expected_swing + right_liftoff.float() * right_expected_swing
        ) * touchdown_balance
        single_support_reward = gait_scale * 0.42 * (
            left_in_air.float() * right_supporting.float() * left_expected_swing
            + right_in_air.float() * left_supporting.float() * right_expected_swing
        ) * touchdown_balance
        swing_clearance_reward = gait_scale * 0.55 * (
            torch.exp(-0.5 * (left_height_error / TOE_PATH_HEIGHT_SIGMA).square())
            * left_in_air.float()
            * left_expected_swing
            + torch.exp(-0.5 * (right_height_error / TOE_PATH_HEIGHT_SIGMA).square())
            * right_in_air.float()
            * right_expected_swing
        ) * touchdown_balance
        stance_contact_reward = gait_scale * 0.45 * (
            left_stance * left_full_contact_score + right_stance * right_full_contact_score
        ) * touchdown_balance
        transition_full_contact_reward = gait_scale * 0.55 * transition_contact_gate * (
            left_full_contact.float() * right_full_contact.float()
        ) * touchdown_balance
        dragging_swing_penalty = gait_scale * 1.45 * early_step_contact_gate * (
            left_expected_swing * left_supporting.float() + right_expected_swing * right_supporting.float()
        )
        swing_ground_contact_penalty = gait_scale * 1.10 * mid_step_contact_gate * (
            left_expected_swing * left_supporting.float() + right_expected_swing * right_supporting.float()
        )
        stance_lost_contact_penalty = gait_scale * 1.20 * (
            left_stance * (1.0 - left_full_contact.float()) + right_stance * (1.0 - right_full_contact.float())
        )
        transition_contact_penalty = gait_scale * 1.35 * transition_contact_gate * (
            1.0 - left_full_contact.float() * right_full_contact.float()
        )

        balance_gate = (
            torch.clamp((upright - math.cos(math.radians(SAFE_TILT_FOR_FORWARD_DEG))) / (1.0 - math.cos(math.radians(SAFE_TILT_FOR_FORWARD_DEG))), 0.0, 1.0)
            * torch.exp(-35.0 * (height - 0.7987).square())
            * torch.clamp((left_supporting.float() + right_supporting.float()) / 1.0, 0.0, 1.0)
        )
        locomotion_hint_gate = torch.maximum(
            torch.maximum(left_in_air.float() * right_supporting.float() * left_expected_swing, right_in_air.float() * left_supporting.float() * right_expected_swing),
            any_valid_touchdown.float(),
        )
        speed_reward = (
            torch.clamp(torch.exp(-((forward_velocity - target_forward_speed).square()) / (2.0 * COMMAND_SPEED_SIGMA * COMMAND_SPEED_SIGMA)) - 0.25, min=0.0)
            * balance_gate
            * (0.05 + 0.55 * locomotion_hint_gate + 0.40 * torch.clamp(base_progress_gate, 0.0, 1.0))
        )
        forward_reward = torch.clamp(forward_velocity, -0.08, target_forward_speed + 0.05) * torch.clamp(base_progress_gate, 0.0, 1.0)
        upright_reward = torch.clamp((upright - 0.70) / 0.30, 0.0, 1.0)
        height_reward = torch.exp(-((height - 0.78) / 0.12).square())
        posture_reward = torch.exp(-1.5 * (joint_pos_after - self.nominal_joint_pose).square().mean(dim=-1))
        forward_accel_cost = torch.clamp((forward_acceleration / FORWARD_ACCEL_SIGMA).square(), 0.0, 8.0)
        backward_velocity_cost = torch.clamp((torch.relu(-forward_velocity) / BACKWARD_VELOCITY_SIGMA).square(), 0.0, 8.0)
        lateral_velocity_cost = torch.clamp((lateral_velocity / LATERAL_VELOCITY_SIGMA).square(), 0.0, 8.0)
        lateral_accel_cost = torch.clamp((lateral_acceleration / LATERAL_ACCEL_SIGMA).square(), 0.0, 8.0)
        pitch_sway_cost = torch.clamp(self.qvel[:, 4].square(), 0.0, 5.0)

        action_cost = 0.0020 * (self.control_cost_weights * filtered_action.square()).sum(dim=-1)
        torque_cost = 0.000015 * self.qfrc_actuator[:, self.joint_dof_adr].square().mean(dim=-1)
        smooth_cost = 0.0016 * (filtered_action - self.prev_action).square().mean(dim=-1)

        reward = (
            0.20 * (0.15 + 0.85 * stand_scale)
            + 0.12 * speed_reward
            + 0.08 * forward_reward
            + progress_tracking_reward
            + positive_progress_reward
            + 0.35 * upright_reward * (0.20 + 0.80 * stand_scale)
            + 0.16 * height_reward
            + 0.08 * posture_reward * (0.98 * stand_scale + 0.10 * march_scale)
            + gait_scale * (1.05 + 0.25 * self.curriculum_progress) * toe_path_reward
            + gait_scale * 0.72 * (left_airtime_phase_score + right_airtime_phase_score)
            + gait_scale * 1.10 * (left_airtime_window_score + right_airtime_window_score)
            + liftoff_reward
            + single_support_reward
            + swing_clearance_reward
            + stance_contact_reward
            + transition_full_contact_reward
            + landing_target_reward
            + walk_touchdown_reward
            + big_step_bonus
            + alternation_reward
            + balanced_step_reward
            + phase_mirror_reward
            - gait_scale * 0.24 * toe_path_cost
            - gait_scale * 0.36 * com_square_wave_cost
            - gait_scale * 0.55 * torch.clamp(stance_clearance_cost, 0.0, 4.0)
            - gait_scale * 0.75 * full_toe_stance_contact_cost
            - gait_scale * 1.20 * both_airborne_cost
            - small_step_penalty
            - oversize_step_penalty
            - no_touchdown_penalty
            - early_touchdown_penalty
            - stepping_in_place_penalty
            - aborted_liftoff_penalty
            - dragging_swing_penalty
            - swing_ground_contact_penalty
            - stance_lost_contact_penalty
            - transition_contact_penalty
            - step_symmetry_penalty
            - mirror_action_penalty
            - gait_scale * 0.13 * foot_slip_cost
            - gait_scale * 0.09 * vertical_velocity_cost
            - gait_scale * 0.09 * roll_pitch_rate_cost
            - gait_scale * 0.08 * pitch_sway_cost
            - gait_scale * 0.16 * forward_accel_cost
            - gait_scale * 0.34 * backward_velocity_cost
            - gait_scale * 0.28 * lateral_velocity_cost
            - gait_scale * 0.20 * lateral_accel_cost
            - action_cost
            - torque_cost
            - smooth_cost
        )

        bad_state = ~torch.isfinite(self.qpos).all(dim=-1) | ~torch.isfinite(self.qvel).all(dim=-1)
        fallen = (height < 0.42) | (height > 1.05) | (upright < math.cos(math.radians(45.0)))
        out_of_bounds = (self.qpos[:, 0].abs() > 5.0) | (self.qpos[:, 1].abs() > 5.0)
        timeout = self.episode_step >= self.episode_length
        done = bad_state | fallen | out_of_bounds | timeout

        self.action_history[self.action_history_index] = filtered_action
        self.action_history_index = (self.action_history_index + 1) % self.action_history_len
        self.prev_action = filtered_action
        self.prev_lateral_velocity = lateral_velocity
        self.prev_forward_velocity = forward_velocity
        self.prev_target = target
        self.prev_left_foot = left_foot
        self.prev_right_foot = right_foot
        self.prev_left_supporting = left_supporting
        self.prev_right_supporting = right_supporting
        self.left_swing_start = torch.where(left_supporting[:, None], left_foot, left_swing_start)
        self.right_swing_start = torch.where(right_supporting[:, None], right_foot, right_swing_start)
        self.left_swing_airtime = torch.where(left_supporting, torch.zeros_like(left_swing_airtime), left_swing_airtime)
        self.right_swing_airtime = torch.where(right_supporting, torch.zeros_like(right_swing_airtime), right_swing_airtime)
        self.left_swing_max_clearance = torch.where(
            left_supporting,
            torch.zeros_like(left_swing_max_clearance),
            left_swing_max_clearance,
        )
        self.right_swing_max_clearance = torch.where(
            right_supporting,
            torch.zeros_like(right_swing_max_clearance),
            right_swing_max_clearance,
        )
        self.left_abort_penalized = torch.where(left_supporting, torch.zeros_like(self.left_abort_penalized), self.left_abort_penalized | left_aborted_liftoff)
        self.right_abort_penalized = torch.where(right_supporting, torch.zeros_like(self.right_abort_penalized), self.right_abort_penalized | right_aborted_liftoff)
        self.steps_since_valid_touchdown = torch.where(
            any_valid_touchdown,
            torch.zeros_like(self.steps_since_valid_touchdown),
            self.steps_since_valid_touchdown + 1,
        )
        self.last_touchdown_side = torch.where(
            left_valid_touchdown,
            torch.ones_like(self.last_touchdown_side),
            torch.where(right_valid_touchdown, torch.full_like(self.last_touchdown_side, 2), self.last_touchdown_side),
        )
        self.last_left_step_forward = torch.where(left_valid_touchdown, left_touchdown_forward, self.last_left_step_forward)
        self.last_right_step_forward = torch.where(right_valid_touchdown, right_touchdown_forward, self.last_right_step_forward)
        self.last_left_step_airtime = torch.where(left_valid_touchdown, left_touchdown_airtime, self.last_left_step_airtime)
        self.last_right_step_airtime = torch.where(right_valid_touchdown, right_touchdown_airtime, self.last_right_step_airtime)
        self.last_left_touchdown_valid = self.last_left_touchdown_valid | left_valid_touchdown
        self.last_right_touchdown_valid = self.last_right_touchdown_valid | right_valid_touchdown
        self.episode_return += reward

        info = {
            "mean_return": float(self.episode_return[done].mean().detach().cpu()) if done.any() else 0.0,
            "mean_length": float(self.episode_length_buf[done].mean().detach().cpu()) if done.any() else 0.0,
            "mean_forward_velocity": float(forward_velocity.mean().detach().cpu()),
            "mean_forward_progress": float(torso_forward_progress.mean().detach().cpu()),
            "mean_progress_gate": float(base_progress_gate.mean().detach().cpu()),
            "mean_valid_touchdown": float(any_valid_touchdown.float().mean().detach().cpu()),
            "mean_both_airborne": float(both_airborne_cost.mean().detach().cpu()),
            "mean_toe_path_cost": float(toe_path_cost.mean().detach().cpu()),
            "mean_com_wave_cost": float(com_square_wave_cost.mean().detach().cpu()),
            "mean_mirror_cost": float(delayed_mirror_cost.mean().detach().cpu()),
            "mean_step_symmetry": float((left_step_symmetry_cost + right_step_symmetry_cost).mean().detach().cpu()),
            "mean_dragging_swing": float(dragging_swing_penalty.mean().detach().cpu()),
            "mean_stance_lost": float(stance_lost_contact_penalty.mean().detach().cpu()),
            "mean_full_toe_contact": float(
                (0.5 * (left_full_contact.float() + right_full_contact.float())).mean().detach().cpu()
            ),
            "mean_forward_accel_cost": float(forward_accel_cost.mean().detach().cpu()),
            "mean_backward_velocity_cost": float(backward_velocity_cost.mean().detach().cpu()),
            "mean_lateral_velocity_cost": float(lateral_velocity_cost.mean().detach().cpu()),
            "mean_lateral_accel_cost": float(lateral_accel_cost.mean().detach().cpu()),
            "done_fraction": float(done.float().mean().detach().cpu()),
        }
        if done.any():
            self.reset(done)

        return self.obs(), reward, done.float(), info


@dataclass
class Rollout:
    obs: torch.Tensor
    actions: torch.Tensor
    log_probs: torch.Tensor
    rewards: torch.Tensor
    dones: torch.Tensor
    values: torch.Tensor


def collect_rollout(
    env: MjWarpSegmentRunnerEnv,
    model: ActorCritic,
    obs_rms: RunningMeanStd,
    obs: torch.Tensor,
    n_steps: int,
) -> tuple[Rollout, torch.Tensor, dict[str, float]]:
    obs_buf = []
    action_buf = []
    logprob_buf = []
    reward_buf = []
    done_buf = []
    value_buf = []
    last_info: dict[str, float] = {}

    for _ in range(n_steps):
        obs_rms.update(obs)
        norm_obs = obs_rms.normalize(obs)
        with torch.no_grad():
            action, log_prob, value = model.act(norm_obs)
        next_obs, reward, done, info = env.step(action)

        obs_buf.append(norm_obs)
        action_buf.append(action)
        logprob_buf.append(log_prob)
        reward_buf.append(reward)
        done_buf.append(done)
        value_buf.append(value)

        obs = next_obs
        last_info = info

    return (
        Rollout(
            obs=torch.stack(obs_buf),
            actions=torch.stack(action_buf),
            log_probs=torch.stack(logprob_buf),
            rewards=torch.stack(reward_buf),
            dones=torch.stack(done_buf),
            values=torch.stack(value_buf),
        ),
        obs,
        last_info,
    )


def compute_gae(
    rollout: Rollout,
    last_value: torch.Tensor,
    gamma: float,
    gae_lambda: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    advantages = torch.zeros_like(rollout.rewards)
    last_gae = torch.zeros_like(last_value)
    for step in reversed(range(rollout.rewards.shape[0])):
        next_nonterminal = 1.0 - rollout.dones[step]
        next_value = last_value if step == rollout.rewards.shape[0] - 1 else rollout.values[step + 1]
        delta = rollout.rewards[step] + gamma * next_value * next_nonterminal - rollout.values[step]
        last_gae = delta + gamma * gae_lambda * next_nonterminal * last_gae
        advantages[step] = last_gae
    returns = advantages + rollout.values
    return advantages, returns


def train(args: argparse.Namespace) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA PyTorch is required for this MJWarp trainer. Current torch build has no CUDA.")

    if args.smoke_test:
        args.n_envs = min(args.n_envs, 8)
        args.n_steps = min(args.n_steps, 8)
        args.timesteps = min(args.timesteps, args.n_envs * args.n_steps)
        args.episode_length = min(args.episode_length, 50)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.set_float32_matmul_precision("high")

    device = torch.device(args.device)
    xml_path = resolve_xml(args.xml_file)
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    env = MjWarpSegmentRunnerEnv(
        xml_path=xml_path,
        n_envs=args.n_envs,
        device=device,
        frame_skip=args.frame_skip,
        episode_length=args.episode_length,
        target_forward_speed=args.target_forward_speed,
        reset_noise_scale=args.reset_noise_scale,
        nconmax=args.nconmax,
        njmax=args.njmax,
        use_cuda_graph=not args.disable_cuda_graph,
    )
    model = ActorCritic(env.obs_dim, env.action_dim).to(device)
    obs_rms = RunningMeanStd((env.obs_dim,), device=device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, eps=1e-5)

    start_update = 0
    checkpoint_path = output_dir / f"{MODEL_BASENAME}.pt"
    if args.resume and checkpoint_path.exists():
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        obs_rms.load_state_dict(checkpoint["obs_rms"])
        start_update = int(checkpoint["update"]) + 1
    elif args.init_checkpoint:
        init_checkpoint_path = Path(args.init_checkpoint).expanduser().resolve()
        if not init_checkpoint_path.exists():
            raise FileNotFoundError(f"Initial standing checkpoint not found: {init_checkpoint_path}")
        checkpoint = torch.load(init_checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint["model"])
        if "obs_rms" in checkpoint:
            obs_rms.load_state_dict(checkpoint["obs_rms"])
        print(f"Initialized policy from stage-6 checkpoint: {init_checkpoint_path}")
        print("Optimizer state was reset for full-toe-contact stable walk stage training.")

    obs = env.obs()
    steps_per_update = args.n_envs * args.n_steps
    args.minibatch_size = min(args.minibatch_size, steps_per_update)
    total_updates = max(args.timesteps // steps_per_update, 1)
    global_step = start_update * steps_per_update
    start_time = time.perf_counter()

    print(f"XML: {xml_path}")
    print(f"Training {args.n_envs} MJWarp worlds on {device}, obs_dim={env.obs_dim}, action_dim={env.action_dim}")
    print(f"Steps/update={steps_per_update}, updates={total_updates}, output={output_dir}")

    for update in range(start_update, total_updates):
        curriculum_blend = min(float(update + 1) / max(float(total_updates), 1.0), 1.0)
        curriculum_progress = args.curriculum_start + (args.curriculum_end - args.curriculum_start) * curriculum_blend
        env.set_curriculum(curriculum_progress)
        rollout, obs, info = collect_rollout(env, model, obs_rms, obs, args.n_steps)
        with torch.no_grad():
            last_value = model.value(obs_rms.normalize(obs)).squeeze(-1)
        advantages, returns = compute_gae(rollout, last_value, args.gamma, args.gae_lambda)

        batch_obs = rollout.obs.reshape((-1, env.obs_dim))
        batch_actions = rollout.actions.reshape((-1, env.action_dim))
        batch_log_probs = rollout.log_probs.reshape(-1)
        batch_advantages = advantages.reshape(-1)
        batch_returns = returns.reshape(-1)
        batch_values = rollout.values.reshape(-1)

        batch_advantages = (batch_advantages - batch_advantages.mean()) / (batch_advantages.std() + 1e-8)
        batch_size = batch_obs.shape[0]

        policy_losses = []
        value_losses = []
        entropy_losses = []
        approx_kls = []
        clip_fracs = []

        for _ in range(args.update_epochs):
            permutation = torch.randperm(batch_size, device=device)
            for start in range(0, batch_size, args.minibatch_size):
                mb_idx = permutation[start : start + args.minibatch_size]
                new_log_prob, entropy, new_value = model.evaluate(batch_obs[mb_idx], batch_actions[mb_idx])
                log_ratio = new_log_prob - batch_log_probs[mb_idx]
                ratio = log_ratio.exp()

                with torch.no_grad():
                    approx_kl = ((ratio - 1.0) - log_ratio).mean()
                    clip_frac = ((ratio - 1.0).abs() > args.clip_coef).float().mean()

                pg_loss_1 = -batch_advantages[mb_idx] * ratio
                pg_loss_2 = -batch_advantages[mb_idx] * torch.clamp(
                    ratio,
                    1.0 - args.clip_coef,
                    1.0 + args.clip_coef,
                )
                policy_loss = torch.max(pg_loss_1, pg_loss_2).mean()

                value_unclipped = (new_value - batch_returns[mb_idx]).square()
                value_clipped = batch_values[mb_idx] + torch.clamp(
                    new_value - batch_values[mb_idx],
                    -args.clip_coef,
                    args.clip_coef,
                )
                value_loss = 0.5 * torch.max(
                    value_unclipped,
                    (value_clipped - batch_returns[mb_idx]).square(),
                ).mean()

                entropy_loss = entropy.mean()
                loss = policy_loss - args.ent_coef * entropy_loss + args.vf_coef * value_loss

                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step()

                policy_losses.append(float(policy_loss.detach().cpu()))
                value_losses.append(float(value_loss.detach().cpu()))
                entropy_losses.append(float(entropy_loss.detach().cpu()))
                approx_kls.append(float(approx_kl.detach().cpu()))
                clip_fracs.append(float(clip_frac.detach().cpu()))

            if args.target_kl is not None and approx_kls and approx_kls[-1] > args.target_kl:
                break

        global_step += steps_per_update
        torch.cuda.synchronize(device)
        elapsed = max(time.perf_counter() - start_time, 1e-6)
        sps = int(global_step / elapsed)
        ev = explained_variance(batch_values, batch_returns)

        if update % args.log_interval == 0:
            print(
                f"update={update + 1}/{total_updates} "
                f"step={global_step} sps={sps} "
                f"reward={rollout.rewards.mean().item():.3f} "
                f"curr={curriculum_progress:.3f} "
                f"done={info.get('done_fraction', 0.0):.3f} "
                f"td={info.get('mean_valid_touchdown', 0.0):.3f} "
                f"air={info.get('mean_both_airborne', 0.0):.3f} "
                f"prog={info.get('mean_forward_progress', 0.0):.3f} "
                f"pgate={info.get('mean_progress_gate', 0.0):.3f} "
                f"toe={info.get('mean_toe_path_cost', 0.0):.3f} "
                f"com={info.get('mean_com_wave_cost', 0.0):.3f} "
                f"mir={info.get('mean_mirror_cost', 0.0):.3f} "
                f"sym={info.get('mean_step_symmetry', 0.0):.3f} "
                f"drag={info.get('mean_dragging_swing', 0.0):.3f} "
                f"stlost={info.get('mean_stance_lost', 0.0):.3f} "
                f"fulltoe={info.get('mean_full_toe_contact', 0.0):.3f} "
                f"acc={info.get('mean_forward_accel_cost', 0.0):.3f} "
                f"back={info.get('mean_backward_velocity_cost', 0.0):.3f} "
                f"latv={info.get('mean_lateral_velocity_cost', 0.0):.3f} "
                f"lata={info.get('mean_lateral_accel_cost', 0.0):.3f} "
                f"ep_ret={info.get('mean_return', 0.0):.2f} "
                f"ep_len={info.get('mean_length', 0.0):.1f} "
                f"vel={info.get('mean_forward_velocity', 0.0):.3f} "
                f"pg={np.mean(policy_losses):.4f} "
                f"v={np.mean(value_losses):.4f} "
                f"ent={np.mean(entropy_losses):.3f} "
                f"kl={np.mean(approx_kls):.5f} "
                f"clip={np.mean(clip_fracs):.3f} "
                f"ev={ev:.3f}"
            )

        if (update + 1) % args.save_interval == 0 or update == total_updates - 1:
            torch.save(
                {
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "obs_rms": obs_rms.state_dict(),
                    "update": update,
                    "args": vars(args),
                },
                checkpoint_path,
            )
            print(f"saved checkpoint: {checkpoint_path}")


def resolve_checkpoint(args: argparse.Namespace) -> Path:
    if args.model_path:
        checkpoint_path = Path(args.model_path).expanduser().resolve()
    else:
        checkpoint_path = Path(args.output_dir).expanduser().resolve() / f"{MODEL_BASENAME}.pt"
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    return checkpoint_path


def make_cpu_policy_obs(
    data,
    joint_qpos_adr: np.ndarray,
    joint_dof_adr: np.ndarray,
    prev_action: np.ndarray,
    episode_step: int,
    dt: float,
    episode_length: int,
    target_forward_speed: float,
) -> np.ndarray:
    phase = 2.0 * math.pi * (float(episode_step) * dt / 0.72)
    extras = np.array(
        [
            math.sin(phase),
            math.cos(phase),
            float(target_forward_speed),
            float(episode_step) / max(float(episode_length), 1.0),
        ],
        dtype=np.float32,
    )
    return np.concatenate(
        [
            data.qpos[:7].astype(np.float32, copy=True),
            np.clip(data.qvel, -25.0, 25.0).astype(np.float32, copy=True),
            data.qpos[joint_qpos_adr].astype(np.float32, copy=True),
            np.clip(data.qvel[joint_dof_adr], -25.0, 25.0).astype(np.float32, copy=True),
            prev_action.astype(np.float32, copy=True),
            extras,
        ],
        dtype=np.float32,
    )


def cpu_upright(data) -> float:
    quat = np.asarray(data.qpos[3:7], dtype=np.float64)
    quat = quat / max(float(np.linalg.norm(quat)), 1e-6)
    return float(1.0 - 2.0 * (quat[1] ** 2 + quat[2] ** 2))


def play(args: argparse.Namespace) -> None:
    import mujoco
    import mujoco.viewer

    checkpoint_path = resolve_checkpoint(args)
    checkpoint = torch.load(checkpoint_path, map_location=args.device, weights_only=False)
    checkpoint_args = checkpoint.get("args", {})

    xml_path = resolve_xml(args.xml_file)
    mj_model = mujoco.MjModel.from_xml_path(str(xml_path))
    mj_data = mujoco.MjData(mj_model)

    action_names = [
        mujoco.mj_id2name(mj_model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
        for i in range(mj_model.nu)
    ]
    if action_names != EXPECTED_ACTUATORS:
        raise ValueError(f"Unexpected actuator order: {action_names}. Expected {EXPECTED_ACTUATORS}.")

    joint_ids = np.array(
        [mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_JOINT, name) for name in JOINT_NAMES],
        dtype=np.int32,
    )
    joint_qpos_adr = mj_model.jnt_qposadr[joint_ids].astype(np.int32)
    joint_dof_adr = mj_model.jnt_dofadr[joint_ids].astype(np.int32)
    joint_bounds = mj_model.jnt_range[joint_ids].astype(np.float32)
    joint_low = joint_bounds[:, 0]
    joint_high = joint_bounds[:, 1]
    ctrl_low = mj_model.actuator_ctrlrange[:, 0].astype(np.float32)
    ctrl_high = mj_model.actuator_ctrlrange[:, 1].astype(np.float32)

    obs_dim = 7 + mj_model.nv + 10 + 10 + 10 + 4
    action_dim = int(mj_model.nu)
    device = torch.device(args.device if torch.cuda.is_available() or not str(args.device).startswith("cuda") else "cpu")
    policy = ActorCritic(obs_dim, action_dim).to(device)
    policy.load_state_dict(checkpoint["model"])
    policy.eval()

    obs_rms = RunningMeanStd((obs_dim,), device=device)
    obs_rms.load_state_dict(checkpoint["obs_rms"])

    nominal_joint_pose = NOMINAL_JOINT_POSE.astype(np.float32)
    position_action_scale = POSITION_ACTION_SCALE.astype(np.float32)
    pd_kp = PD_KP.astype(np.float32)
    pd_kd = PD_KD.astype(np.float32)

    frame_skip = int(checkpoint_args.get("frame_skip", args.frame_skip))
    episode_length = int(checkpoint_args.get("episode_length", args.episode_length))
    target_forward_speed = float(checkpoint_args.get("target_forward_speed", args.target_forward_speed))
    control_dt = float(mj_model.opt.timestep * frame_skip)

    qpos0 = mj_model.qpos0.copy()
    qpos0[joint_qpos_adr] = nominal_joint_pose

    def reset_episode() -> tuple[np.ndarray, np.ndarray, int]:
        mj_data.qpos[:] = qpos0
        mj_data.qvel[:] = 0.0
        mj_data.ctrl[:] = 0.0
        mujoco.mj_forward(mj_model, mj_data)
        return np.zeros(action_dim, dtype=np.float32), nominal_joint_pose.copy(), 0

    prev_action, prev_target, episode_step = reset_episode()
    print(f"Viewing checkpoint: {checkpoint_path}")
    print("Close the MuJoCo viewer window to stop playback.")

    with mujoco.viewer.launch_passive(mj_model, mj_data) as viewer:
        while viewer.is_running() and episode_step < args.play_steps:
            step_start = time.perf_counter()
            obs = make_cpu_policy_obs(
                mj_data,
                joint_qpos_adr,
                joint_dof_adr,
                prev_action,
                episode_step,
                control_dt,
                episode_length,
                target_forward_speed,
            )
            obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
            with torch.no_grad():
                norm_obs = obs_rms.normalize(obs_tensor)
                if args.stochastic_play:
                    raw_action, _, _ = policy.act(norm_obs)
                else:
                    raw_action = policy.policy(norm_obs)
            action = np.clip(raw_action.squeeze(0).detach().cpu().numpy(), -1.0, 1.0).astype(np.float32)
            filtered_action = 0.20 * prev_action + 0.80 * action

            joint_pos = mj_data.qpos[joint_qpos_adr].astype(np.float32)
            joint_vel = mj_data.qvel[joint_dof_adr].astype(np.float32)
            raw_target = np.clip(
                nominal_joint_pose + filtered_action * position_action_scale,
                joint_low,
                joint_high,
            )
            target = np.clip(0.40 * prev_target + 0.60 * raw_target, joint_low, joint_high)
            pd_ctrl = pd_kp * (target - joint_pos) - pd_kd * joint_vel
            mj_data.ctrl[:] = np.clip(pd_ctrl, ctrl_low, ctrl_high)

            for _ in range(frame_skip):
                mujoco.mj_step(mj_model, mj_data)

            prev_action = filtered_action
            prev_target = target
            episode_step += 1
            viewer.sync()

            height = float(mj_data.qpos[2])
            fallen = height < 0.42 or height > 1.05 or cpu_upright(mj_data) < math.cos(math.radians(45.0))
            out_of_bounds = abs(float(mj_data.qpos[0])) > 5.0 or abs(float(mj_data.qpos[1])) > 5.0
            if fallen or out_of_bounds or episode_step >= episode_length:
                prev_action, prev_target, episode_step = reset_episode()

            sleep_time = control_dt / max(args.real_time_factor, 1e-6) - (time.perf_counter() - step_start)
            if sleep_time > 0.0:
                time.sleep(sleep_time)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="GPU-vectorized MJWarp + PyTorch PPO full-toe-contact stable walk stage-7 trainer.")
    parser.add_argument("--mode", choices=["train", "play"], default="train")
    parser.add_argument("--xml-file", default=DEFAULT_XML)
    parser.add_argument("--output-dir", default="artifacts/mjwarp_walk_stage_7_full_toe_stable")
    parser.add_argument("--model-path", default=None)
    parser.add_argument("--init-checkpoint", default=DEFAULT_STAGE_6_CHECKPOINT)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-envs", type=int, default=256)
    parser.add_argument("--timesteps", type=int, default=4_000_000)
    parser.add_argument("--episode-length", type=int, default=800)
    parser.add_argument("--frame-skip", type=int, default=5)
    parser.add_argument("--target-forward-speed", type=float, default=0.14)
    parser.add_argument("--curriculum-start", type=float, default=0.68)
    parser.add_argument("--curriculum-end", type=float, default=0.90)
    parser.add_argument("--reset-noise-scale", type=float, default=1e-3)
    parser.add_argument("--nconmax", type=int, default=128)
    parser.add_argument("--njmax", type=int, default=512)
    parser.add_argument("--n-steps", type=int, default=256)
    parser.add_argument("--minibatch-size", type=int, default=4096)
    parser.add_argument("--update-epochs", type=int, default=5)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--clip-coef", type=float, default=0.2)
    parser.add_argument("--ent-coef", type=float, default=0.0)
    parser.add_argument("--vf-coef", type=float, default=0.5)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--target-kl", type=float, default=0.02)
    parser.add_argument("--log-interval", type=int, default=1)
    parser.add_argument("--save-interval", type=int, default=10)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--disable-cuda-graph", action="store_true")
    parser.add_argument("--smoke-test", action="store_true", help="Run one tiny update to verify the environment.")
    parser.add_argument("--play-steps", type=int, default=100_000)
    parser.add_argument("--real-time-factor", type=float, default=1.0)
    parser.add_argument("--stochastic-play", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.mode == "train":
        train(args)
    else:
        play(args)


if __name__ == "__main__":
    main()
