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
        logger.info(f"Construct {type(self).__name__} with lambda: ")

    def __call__(self, preds: Dict, targs: Dict, **kwargs) -> Tuple[torch.Tensor, Dict]:
        final_loss, losses = super().__call__(preds, targs, **kwargs)  # TENSOR(0.), {}
        # ============== OBJ VEL&OMEGA MSE LOSS >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
        box_vel_12d = preds["box_kin_12d"]
        vel_predict = box_vel_12d[:, :3]
        if not self.use_last:
            vel_real = targs[Queries.TARGET_VEL].to(final_loss.device)
        else:
            vel_real = targs[Queries.TARGET_NEXT_VEL].to(final_loss.device)
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
        logger.info(f"Construct {type(self).__name__} with lambda: ")

    def __call__(self, preds: Dict, targs: Dict, **kwargs) -> Tuple[torch.Tensor, Dict]:
        final_loss, losses = super().__call__(preds, targs, **kwargs)  # TENSOR(0.), {}
        # ============== OBJ VEL&OMEGA MSE LOSS >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
        box_vel_12d = preds["box_kin_12d"]
        omega_predict = box_vel_12d[:, 3:6]
        if not self.use_last:
            omega_real = targs[Queries.TARGET_OMEGA].to(final_loss.device)
        else:
            omega_real = targs[Queries.TARGET_NEXT_OMEGA].to(final_loss.device)
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
        logger.info(f"Construct {type(self).__name__} with lambda: ")

    def __call__(self, preds: Dict, targs: Dict, **kwargs) -> Tuple[torch.Tensor, Dict]:
        final_loss, losses = super().__call__(preds, targs, **kwargs)
        # ============== OBJ ACC&BETA MSE LOSS >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
        box_vel_12d = preds["box_kin_12d"]
        acc_predict = box_vel_12d[:, 6:9]
        if not self.use_last:
            acc_real = targs[Queries.TARGET_ACC].to(final_loss.device)
        else:
            acc_real = targs[Queries.TARGET_NEXT_ACC].to(final_loss.device)
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
        logger.info(f"Construct {type(self).__name__} with lambda: ")

    def __call__(self, preds: Dict, targs: Dict, **kwargs) -> Tuple[torch.Tensor, Dict]:
        final_loss, losses = super().__call__(preds, targs, **kwargs)
        # ============== OBJ ACC&BETA MSE LOSS >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
        box_vel_12d = preds["box_kin_12d"]
        beta_predict = box_vel_12d[:, 9:12]
        if not self.use_last:
            beta_real = targs[Queries.TARGET_BETA].to(final_loss.device)
        else:
            beta_real = targs[Queries.TARGET_NEXT_BETA].to(final_loss.device)
        loss_beta = torch_f.mse_loss(beta_predict, beta_real).float()
        final_loss += loss_beta

        # <<
        losses[self.output_key] = final_loss
        return final_loss, losses