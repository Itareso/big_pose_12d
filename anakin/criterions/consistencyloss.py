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
    acc_mid = acc_list[:, 1] * 0.7 + acc_list[:, 2] * 0.15 + acc_list[:, 0] * 0.15

    omega_mid = (omega_list[:, 1] + omega_list[:, 2]) / 2
    beta_mid = beta_list[:, 1] * 0.7 + beta_list[:, 2] * 0.15 + beta_list[:, 0] * 0.15

    return vel_mid, omega_mid, acc_mid, beta_mid

def get_vel_and_omega_from_preds(preds):
    corner_3d_abs = preds["corners_3d_abs"]
    prev_corner_3d_abs = preds["corners_3d_abs_list"][0]
    next_corner_3d_abs = preds["corners_3d_abs_list"][2]
    box_rot_6d = preds["box_rot_6d"]
    prev_box_rot_6d = preds["box_rot_6d_list"][0]
    next_box_rot_6d = preds["box_rot_6d_list"][2]
    fps = 30

    vel1, omega1 = compute_velocity_and_omega(prev_corner_3d_abs, corner_3d_abs, prev_box_rot_6d, box_rot_6d, fps)
    vel2, omega2 = compute_velocity_and_omega(corner_3d_abs, next_corner_3d_abs, box_rot_6d, next_box_rot_6d, fps)
    return vel1, omega1, vel2, omega2


@LOSS.register_module
class VelConsistencyLoss(TensorLoss):
    def __init__(self, **cfg):
        super(VelConsistencyLoss, self).__init__()
        self.use_norm = cfg.get("USE_NORM", False)

        logger.info(f"Construct {type(self).__name__} with lambda: ")

    def __call__(self, preds: Dict, targs: Dict, **kwargs) -> Tuple[torch.Tensor, Dict]:
        final_loss, losses = super().__call__(preds, targs, **kwargs)  # TENSOR(0.), {}
        # ============== Consistency LOSS >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
        frame_num = targs[Queries.CORNERS_3D_LIST][0].shape[0]
        if frame_num == 5:
            vel_mid, omega_mid, acc_mid, beta_mid = get_kin_mid_from_preds(preds)
        elif frame_num == 3:
            vel1, omega1, vel2, omega2 = get_vel_and_omega_from_preds(preds)
            vel_mid = (vel1 + vel2) / 2
        vel_predict = preds["box_kin_12d"][:, 0:3]

        vel_mean, vel_std = targs[Queries.KIN_DATA_MEAN][:,:3], targs[Queries.KIN_DATA_STD][:,:3]
        vel_mean, vel_std = vel_mean.to(final_loss.device), vel_std.to(final_loss.device)
        if self.use_norm:
            vel_mid = (vel_mid - vel_mean) / vel_std
        
        vel_real = targs[Queries.TARGET_VEL].to(final_loss.device)

        vel_loss = torch_f.mse_loss(vel_mid, vel_predict).float()
        # vel_loss = torch_f.mse_loss(vel_mid, vel_real).float()
        final_loss += vel_loss

        # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

        losses[self.output_key] = final_loss
        return final_loss, losses

@LOSS.register_module
class OmegaConsistencyLoss(TensorLoss):
    def __init__(self, **cfg):
        super(OmegaConsistencyLoss, self).__init__()
        self.exist_model_info = True
        self.use_norm = cfg.get("USE_NORM", False)
        try:
            model_info_path = cfg["MODEL_INFO_PATH"]
        except:
            self.exist_model_info = False
        if self.exist_model_info:
            self.model_info = json.load(open(model_info_path, "r"))
            self.model_sym = {}
            for obj_idx in range(1, len(self.model_info) + 1):
                self.model_sym[obj_idx] = get_symmetry_transformations(self.model_info[str(obj_idx)], 0.01)

        logger.info(f"Construct {type(self).__name__} with lambda: ")

    def __call__(self, preds: Dict, targs: Dict, **kwargs) -> Tuple[torch.Tensor, Dict]:
        final_loss, losses = super().__call__(preds, targs, **kwargs)  # TENSOR(0.), {}
        # ============== Consistency LOSS >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
        frame_num = targs[Queries.CORNERS_3D_LIST][0].shape[0]
        if frame_num == 5:
            vel_mid, omega_mid, acc_mid, beta_mid = get_kin_mid_from_preds(preds)
        elif frame_num == 3:
            vel1, omega1, vel2, omega2 = get_vel_and_omega_from_preds(preds)
            omega_mid = (omega1 + omega2) / 2
        omega_predict = preds["box_kin_12d"][:, 3:6]

        if self.exist_model_info:
            self.obj_idx = targs[Queries.OBJ_IDX]
            self.no_sym = torch.tensor([len(self.model_sym[i.item()]) == 1 for i in self.obj_idx], dtype = torch.int32).to(final_loss.device)
            omega_predict_mask = omega_predict[self.no_sym == 1]
            omega_mid_mask = omega_mid[self.no_sym == 1]
        else:
            omega_predict_mask = omega_predict
            omega_mid_mask = omega_mid
        
        omega_mean, omega_std = targs[Queries.KIN_DATA_MEAN][:,3:6], targs[Queries.KIN_DATA_STD][:,3:6]
        omega_mean, omega_std = omega_mean.to(final_loss.device), omega_std.to(final_loss.device)
        if self.use_norm:
            omega_mid_mask = (omega_mid_mask - omega_mean) / omega_std

        omega_loss = torch_f.mse_loss(omega_mid_mask, omega_predict_mask).float()
        final_loss += omega_loss

        # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

        losses[self.output_key] = final_loss
        return final_loss, losses

@LOSS.register_module
class AccConsistencyLoss(TensorLoss):
    def __init__(self, **cfg):
        super(AccConsistencyLoss, self).__init__()
        self.use_norm = cfg.get("USE_NORM", False)

        logger.info(f"Construct {type(self).__name__} with lambda: ")

    def __call__(self, preds: Dict, targs: Dict, **kwargs) -> Tuple[torch.Tensor, Dict]:
        final_loss, losses = super().__call__(preds, targs, **kwargs)  # TENSOR(0.), {}
        # ============== Consistency LOSS >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
        frame_num = targs[Queries.CORNERS_3D_LIST][0].shape[0]
        if frame_num == 5:
            vel_mid, omega_mid, acc_mid, beta_mid = get_kin_mid_from_preds(preds)
        elif frame_num == 3:
            vel1, omega1, vel2, omega2 = get_vel_and_omega_from_preds(preds)
            acc_mid = (vel2 - vel1) * 30
        acc_predict = preds["box_kin_12d"][:, 6:9]

        acc_mean, acc_std = targs[Queries.KIN_DATA_MEAN][:,6:9], targs[Queries.KIN_DATA_STD][:,6:9]
        acc_mean, acc_std = acc_mean.to(final_loss.device), acc_std.to(final_loss.device)
        if self.use_norm:
            acc_mid = (acc_mid - acc_mean) / acc_std

        acc_loss = torch_f.mse_loss(acc_mid, acc_predict).float()
        final_loss += acc_loss

        # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

        losses[self.output_key] = final_loss
        return final_loss, losses

@LOSS.register_module
class BetaConsistencyLoss(TensorLoss):
    def __init__(self, **cfg):
        super(BetaConsistencyLoss, self).__init__()
        self.use_norm = cfg.get("USE_NORM", False)

        self.exist_model_info = True
        try:
            model_info_path = cfg["MODEL_INFO_PATH"]
        except:
            self.exist_model_info = False
        if self.exist_model_info:
            self.model_info = json.load(open(model_info_path, "r"))
            self.model_sym = {}
            for obj_idx in range(1, len(self.model_info) + 1):
                self.model_sym[obj_idx] = get_symmetry_transformations(self.model_info[str(obj_idx)], 0.01)

        logger.info(f"Construct {type(self).__name__} with lambda: ")

    def __call__(self, preds: Dict, targs: Dict, **kwargs) -> Tuple[torch.Tensor, Dict]:
        final_loss, losses = super().__call__(preds, targs, **kwargs)  # TENSOR(0.), {}
        # ============== Consistency LOSS >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
        frame_num = targs[Queries.CORNERS_3D_LIST][0].shape[0]
        if frame_num == 5:
            vel_mid, omega_mid, acc_mid, beta_mid = get_kin_mid_from_preds(preds)
        elif frame_num == 3:
            vel1, omega1, vel2, omega2 = get_vel_and_omega_from_preds(preds)
            beta_mid = (omega2 - omega1) * 30
        beta_predict = preds["box_kin_12d"][:, 9:12]

        if self.exist_model_info:
            self.obj_idx = targs[Queries.OBJ_IDX]
            self.no_sym = torch.tensor([len(self.model_sym[i.item()]) == 1 for i in self.obj_idx], dtype = torch.int32).to(final_loss.device)
            beta_predict_mask = beta_predict[self.no_sym == 1]
            beta_mid_mask = beta_mid[self.no_sym == 1]
        else:
            beta_predict_mask = beta_predict
            beta_mid_mask = beta_mid
        
        beta_mean, beta_std = targs[Queries.KIN_DATA_MEAN][:,9:12], targs[Queries.KIN_DATA_STD][:,9:12]
        beta_mean, beta_std = beta_mean.to(final_loss.device), beta_std.to(final_loss.device)
        if self.use_norm:
            beta_mid_mask = (beta_mid_mask - beta_mean) / beta_std

        beta_loss = torch_f.mse_loss(beta_mid_mask, beta_predict_mask).float()
        final_loss += beta_loss

        # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

        losses[self.output_key] = final_loss
        return final_loss, losses