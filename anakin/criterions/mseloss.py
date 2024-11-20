import json
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn.functional as torch_f
from anakin.datasets.hoquery import Queries
from anakin.utils.bop_toolkit.bop_misc import get_symmetry_transformations
from anakin.utils.builder import LOSS
from ..utils.transform import batch_ref_bone_len
from anakin.utils.logger import logger
from anakin.utils.misc import CONST

import os

from .criterion import TensorLoss

@LOSS.register_module
class MSEVelLoss(TensorLoss):
    def __init__(self, **cfg):
        super(MSEVelLoss, self).__init__()

        self.use_last = cfg.get("USE_LAST", False)
        self.use_norm = cfg.get("USE_NORM", False)
        logger.info(f"Construct {type(self).__name__} with lambda: ")

    def __call__(self, preds: Dict, targs: Dict, **kwargs) -> Tuple[torch.Tensor, Dict]:
        final_loss, losses = super().__call__(preds, targs, **kwargs)  # TENSOR(0.), {}
        # ============== OBJ VEL&OMEGA MSE LOSS >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
        box_vel_12d = preds["box_kin_12d"]
        vel_predict = box_vel_12d[:, :3]
        if not self.use_last:
            vel_real = targs[Queries.TARGET_VEL].to(final_loss.device)
        else:
            frame_num = targs[Queries.CORNERS_3D_LIST][0].shape[0]
            if frame_num == 3:  
                vel_real = targs[Queries.TARGET_NEXT_VEL].to(final_loss.device)
            elif frame_num == 5:
                vel_real = targs[Queries.TARGET_NNEXT_VEL].to(final_loss.device)
        vel_mean, vel_std = targs[Queries.KIN_DATA_MEAN][:,:3], targs[Queries.KIN_DATA_STD][:,:3]
        vel_mean, vel_std = vel_mean.to(final_loss.device), vel_std.to(final_loss.device)
        if self.use_norm:
            vel_real = (vel_real - vel_mean) / vel_std
        loss_vel = torch_f.mse_loss(vel_predict, vel_real).float()
        final_loss += loss_vel

        # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

        losses[self.output_key] = final_loss
        return final_loss, losses

@LOSS.register_module
class MSEOmegaLoss(TensorLoss):
    def __init__(self, **cfg):
        super(MSEOmegaLoss, self).__init__()

        self.use_last = cfg.get("USE_LAST", False)
        self.use_norm = cfg.get("USE_NORM", False)
        logger.info(f"Construct {type(self).__name__} with lambda: ")

    def __call__(self, preds: Dict, targs: Dict, **kwargs) -> Tuple[torch.Tensor, Dict]:
        final_loss, losses = super().__call__(preds, targs, **kwargs)  # TENSOR(0.), {}
        # ============== OBJ VEL&OMEGA MSE LOSS >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
        box_vel_12d = preds["box_kin_12d"]
        omega_predict = box_vel_12d[:, 3:6]
        if not self.use_last:
            omega_real = targs[Queries.TARGET_OMEGA].to(final_loss.device)
        else:
            frame_num = targs[Queries.CORNERS_3D_LIST][0].shape[0]
            if frame_num == 3:  
                omega_real = targs[Queries.TARGET_NEXT_OMEGA].to(final_loss.device)
            elif frame_num == 5:
                omega_real = targs[Queries.TARGET_NNEXT_OMEGA].to(final_loss.device)
        omega_mean, omega_std = targs[Queries.KIN_DATA_MEAN][:,3:6], targs[Queries.KIN_DATA_STD][:,3:6]
        omega_mean, omega_std = omega_mean.to(final_loss.device), omega_std.to(final_loss.device)
        if self.use_norm:
            omega_real = (omega_real - omega_mean) / omega_std
        loss_omega = torch_f.mse_loss(omega_predict, omega_real).float()
        final_loss += loss_omega

        # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

        losses[self.output_key] = final_loss
        return final_loss, losses

@LOSS.register_module
class MSEAccLoss(TensorLoss):
    def __init__(self, **cfg):
        super(MSEAccLoss, self).__init__()

        self.use_last = cfg.get("USE_LAST", False)
        self.use_norm = cfg.get("USE_NORM", False)
        logger.info(f"Construct {type(self).__name__} with lambda: ")

    def __call__(self, preds: Dict, targs: Dict, **kwargs) -> Tuple[torch.Tensor, Dict]:
        final_loss, losses = super().__call__(preds, targs, **kwargs)
        # ============== OBJ ACC&BETA MSE LOSS >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
        box_vel_12d = preds["box_kin_12d"]
        acc_predict = box_vel_12d[:, 6:9]
        if not self.use_last:
            acc_real = targs[Queries.TARGET_ACC].to(final_loss.device)
        else:
            frame_num = targs[Queries.CORNERS_3D_LIST][0].shape[0]
            if frame_num == 3:  
                acc_real = targs[Queries.TARGET_NEXT_ACC].to(final_loss.device)
            elif frame_num == 5:
                acc_real = targs[Queries.TARGET_NNEXT_ACC].to(final_loss.device)
        acc_mean, acc_std = targs[Queries.KIN_DATA_MEAN][:,6:9], targs[Queries.KIN_DATA_STD][:,6:9]
        acc_mean, acc_std = acc_mean.to(final_loss.device), acc_std.to(final_loss.device)
        if self.use_norm:
            acc_real = (acc_real - acc_mean) / acc_std
        loss_acc = torch_f.mse_loss(acc_predict, acc_real).float()
        final_loss += loss_acc

        # <<

        losses[self.output_key] = final_loss
        return final_loss, losses

@LOSS.register_module
class MSEBetaLoss(TensorLoss):
    def __init__(self, **cfg):
        super(MSEBetaLoss, self).__init__()

        self.use_last = cfg.get("USE_LAST", False)
        self.use_norm = cfg.get("USE_NORM", False)
        logger.info(f"Construct {type(self).__name__} with lambda: ")

    def __call__(self, preds: Dict, targs: Dict, **kwargs) -> Tuple[torch.Tensor, Dict]:
        final_loss, losses = super().__call__(preds, targs, **kwargs)
        # ============== OBJ ACC&BETA MSE LOSS >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
        box_vel_12d = preds["box_kin_12d"]
        beta_predict = box_vel_12d[:, 9:12]
        if not self.use_last:
            beta_real = targs[Queries.TARGET_BETA].to(final_loss.device)
        else:
            frame_num = targs[Queries.CORNERS_3D_LIST][0].shape[0]
            if frame_num == 3:  
                beta_real = targs[Queries.TARGET_NEXT_BETA].to(final_loss.device)
            elif frame_num == 5:
                beta_real = targs[Queries.TARGET_NNEXT_BETA].to(final_loss.device)
        beta_mean, beta_std = targs[Queries.KIN_DATA_MEAN][:,9:12], targs[Queries.KIN_DATA_STD][:,9:12]
        beta_mean, beta_std = beta_mean.to(final_loss.device), beta_std.to(final_loss.device)
        if self.use_norm:
            beta_real = (beta_real - beta_mean) / beta_std
        loss_beta = torch_f.mse_loss(beta_predict, beta_real).float()
        final_loss += loss_beta

        # <<
        losses[self.output_key] = final_loss
        return final_loss, losses