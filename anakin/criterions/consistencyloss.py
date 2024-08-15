import json
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn.functional as torch_f
from anakin.datasets.hoquery import Queries
from anakin.utils.bop_toolkit.bop_misc import get_symmetry_transformations
from anakin.utils.builder import LOSS
from ..utils.transform import batch_ref_bone_len, compute_rotation_matrix_from_ortho6d
from anakin.utils.logger import logger
from anakin.utils.misc import CONST
from scipy.spatial.transform import Rotation as R
from roma.mappings import rotmat_to_rotvec

import os

from .criterion import TensorLoss

import torch
import numpy as np

def compute_velocity_and_omega(corners3d1, corners3d2, ortho6d1, ortho6d2, fps):
    # Compute the center of the 3D bounding box for each frame
    center1 = corners3d1.mean(1)  # corners3d: [B, 8, 3] -> [B, 3]
    center2 = corners3d2.mean(1)

    # Compute the translation and rotation from frame1 to frame2
    rot1 = compute_rotation_matrix_from_ortho6d(ortho6d1)
    rot2 = compute_rotation_matrix_from_ortho6d(ortho6d2)
    rot = rot2 @ rot1.transpose(1, 2)

    # Compute the velocity
    velocity = (center2 - center1) * fps

    r = rotmat_to_rotvec(rot)
    omega = r * fps

    return velocity, omega

def get_vel_and_omega_from_preds(preds):
    corner_3d_abs = preds["corners_3d_abs"]
    prev_corner_3d_abs = preds["prev_corners_3d_abs"]
    next_corner_3d_abs = preds["next_corners_3d_abs"]
    box_rot_6d = preds["box_rot_6d"]
    prev_box_rot_6d = preds["prev_box_rot_6d"]
    next_box_rot_6d = preds["next_box_rot_6d"]
    fps = 30

    vel1, omega1 = compute_velocity_and_omega(prev_corner_3d_abs, corner_3d_abs, prev_box_rot_6d, box_rot_6d, fps)
    vel2, omega2 = compute_velocity_and_omega(corner_3d_abs, next_corner_3d_abs, box_rot_6d, next_box_rot_6d, fps)
    return vel1, omega1, vel2, omega2

@LOSS.register_module
class VelConsistencyLoss(TensorLoss):
    def __init__(self, **cfg):
        super(VelConsistencyLoss, self).__init__()

        logger.info(f"Construct {type(self).__name__} with lambda: ")

    def __call__(self, preds: Dict, targs: Dict, **kwargs) -> Tuple[torch.Tensor, Dict]:
        final_loss, losses = super().__call__(preds, targs, **kwargs)  # TENSOR(0.), {}
        # ============== Consistency LOSS >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
        vel1, omega1, vel2, omega2 = get_vel_and_omega_from_preds(preds)
        vel_predict = preds["box_vel_12d"][:, 0:3]
        vel = (vel1 + vel2) / 2

        vel_loss = torch_f.mse_loss(vel, vel_predict).float()
        final_loss += vel_loss

        # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

        losses[self.output_key] = final_loss
        return final_loss, losses

@LOSS.register_module
class OmegaConsistencyLoss(TensorLoss):
    def __init__(self, **cfg):
        super(OmegaConsistencyLoss, self).__init__()

        logger.info(f"Construct {type(self).__name__} with lambda: ")

    def __call__(self, preds: Dict, targs: Dict, **kwargs) -> Tuple[torch.Tensor, Dict]:
        final_loss, losses = super().__call__(preds, targs, **kwargs)  # TENSOR(0.), {}
        # ============== Consistency LOSS >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
        vel1, omega1, vel2, omega2 = get_vel_and_omega_from_preds(preds)
        omega_predict = preds["box_vel_12d"][:, 3:6]
        omega = (omega1 + omega2) / 2

        omega_loss = torch_f.mse_loss(omega, omega_predict).float()
        final_loss += omega_loss

        # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

        losses[self.output_key] = final_loss
        return final_loss, losses

@LOSS.register_module
class AccConsistencyLoss(TensorLoss):
    def __init__(self, **cfg):
        super(AccConsistencyLoss, self).__init__()

        logger.info(f"Construct {type(self).__name__} with lambda: ")

    def __call__(self, preds: Dict, targs: Dict, **kwargs) -> Tuple[torch.Tensor, Dict]:
        final_loss, losses = super().__call__(preds, targs, **kwargs)  # TENSOR(0.), {}
        # ============== Consistency LOSS >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
        fps = 30
        vel1, omega1, vel2, omega2 = get_vel_and_omega_from_preds(preds)
        acc_predict = preds["box_vel_12d"][:, 6:9]
        acc = (vel2 - vel1) * fps

        acc_loss = torch_f.mse_loss(acc, acc_predict).float()
        final_loss += acc_loss

        # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

        losses[self.output_key] = final_loss
        return final_loss, losses

@LOSS.register_module
class BetaConsistencyLoss(TensorLoss):
    def __init__(self, **cfg):
        super(BetaConsistencyLoss, self).__init__()

        logger.info(f"Construct {type(self).__name__} with lambda: ")

    def __call__(self, preds: Dict, targs: Dict, **kwargs) -> Tuple[torch.Tensor, Dict]:
        final_loss, losses = super().__call__(preds, targs, **kwargs)  # TENSOR(0.), {}
        # ============== Consistency LOSS >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
        fps = 30
        vel1, omega1, vel2, omega2 = get_vel_and_omega_from_preds(preds)
        beta_predict = preds["box_vel_12d"][:, 9:12]
        beta = (omega2 - omega1) * fps

        beta_loss = torch_f.mse_loss(beta, beta_predict).float()
        final_loss += beta_loss

        # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

        losses[self.output_key] = final_loss
        return final_loss, losses