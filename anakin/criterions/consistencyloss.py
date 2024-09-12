import json
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn.functional as torch_f
from anakin.datasets.hoquery import Queries
from anakin.utils.bop_toolkit.bop_misc import get_symmetry_transformations
from anakin.utils.builder import LOSS
from ..utils.transform import batch_ref_bone_len, compute_rotation_matrix_from_ortho6d, rotmat_to_quat
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

def get_kin_mid_from_preds(preds):
    corner_3d_abs_list = preds["corners_3d_abs_list"]
    box_rot_6d_list = preds["box_rot_6d_list"]
    corner_3d_abs_list = torch.stack(corner_3d_abs_list, dim=1)
    box_rot_6d_list = torch.stack(box_rot_6d_list, dim=1)
    
    obj_center = corner_3d_abs_list.mean(2)
    box_rot_flat = box_rot_6d_list.view(-1, 6)
    rot_list_flat = compute_rotation_matrix_from_ortho6d(box_rot_flat)
    rotvec_list_flat = rotmat_to_rotvec(rot_list_flat)
    rotvec_list_flat = rotvec_list_flat.view(-1, 5, 3)
    
    fps = 30

    vel_list = torch.diff(obj_center, dim=1) * fps
    acc_list = torch.diff(vel_list, dim=1) * fps
    omega_list = torch.diff(rotvec_list_flat, dim=1) * fps
    beta_list = torch.diff(omega_list, dim=1) * fps

    vel_mid = (vel_list[:, 1] + vel_list[:, 2]) / 2
    acc_mid = acc_list[:, 1]

    omega_mid = (omega_list[:, 1] + omega_list[:, 2]) / 2
    beta_mid = beta_list[:, 1]

    return vel_mid, omega_mid, acc_mid, beta_mid


@LOSS.register_module
class VelConsistencyLoss(TensorLoss):
    def __init__(self, **cfg):
        super(VelConsistencyLoss, self).__init__()

        logger.info(f"Construct {type(self).__name__} with lambda: ")

    def __call__(self, preds: Dict, targs: Dict, **kwargs) -> Tuple[torch.Tensor, Dict]:
        final_loss, losses = super().__call__(preds, targs, **kwargs)  # TENSOR(0.), {}
        # ============== Consistency LOSS >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
        vel_mid, omega_mid, acc_mid, beta_mid = get_kin_mid_from_preds(preds)
        vel_predict = preds["box_kin_12d"][:, 0:3]

        vel_loss = torch_f.mse_loss(vel_mid, vel_predict).float()
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
        vel_mid, omega_mid, acc_mid, beta_mid = get_kin_mid_from_preds(preds)
        omega_predict = preds["box_kin_12d"][:, 3:6]

        omega_loss = torch_f.mse_loss(omega_mid, omega_predict).float()
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
        vel_mid, omega_mid, acc_mid, beta_mid = get_kin_mid_from_preds(preds)
        acc_predict = preds["box_kin_12d"][:, 6:9]

        acc_loss = torch_f.mse_loss(acc_mid, acc_predict).float()
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
        vel_mid, omega_mid, acc_mid, beta_mid = get_kin_mid_from_preds(preds)
        beta_predict = preds["box_kin_12d"][:, 9:12]

        beta_loss = torch_f.mse_loss(beta_mid, beta_predict).float()
        final_loss += beta_loss

        # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

        losses[self.output_key] = final_loss
        return final_loss, losses