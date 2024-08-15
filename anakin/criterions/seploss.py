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
class VelCosLoss(TensorLoss):
    def __init__(self, **cfg):
        super(VelCosLoss, self).__init__()

        logger.info(f"Construct {type(self).__name__} with lambda: ")

    def __call__(self, preds: Dict, targs: Dict, **kwargs) -> Tuple[torch.Tensor, Dict]:
        final_loss, losses = super().__call__(preds, targs, **kwargs)  # TENSOR(0.), {}
        # ============== OBJ VEL COS LOSS >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
        box_vel_12d = preds["box_vel_12d"]
        vel_predict = box_vel_12d[:, :3]
        vel_real = targs[Queries.TARGET_VEL].to(final_loss.device)
        loss = 1 - torch_f.cosine_similarity(vel_predict, vel_real, dim=1)
        loss = loss.mean()
        final_loss += loss

        # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

        losses[self.output_key] = final_loss
        return final_loss, losses

@LOSS.register_module
class AccCosLoss(TensorLoss):
    def __init__(self, **cfg):
        super(AccCosLoss, self).__init__()

        logger.info(f"Construct {type(self).__name__} with lambda: ")

    def __call__(self, preds: Dict, targs: Dict, **kwargs) -> Tuple[torch.Tensor, Dict]:
        final_loss, losses = super().__call__(preds, targs, **kwargs)  # TENSOR(0.), {}
        # ============== OBJ ACC COS LOSS >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
        box_vel_12d = preds["box_vel_12d"]
        acc_predict = box_vel_12d[:, 6:9]
        acc_real = targs[Queries.TARGET_ACC].to(final_loss.device)
        loss = 1 - torch_f.cosine_similarity(acc_predict, acc_real, dim=1)
        loss = loss.mean()
        final_loss += loss

        # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

        losses[self.output_key] = final_loss
        return final_loss, losses

@LOSS.register_module
class VelMagLoss(TensorLoss):
    def __init__(self, **cfg):
        super(VelMagLoss, self).__init__()

        logger.info(f"Construct {type(self).__name__} with lambda: ")

    def __call__(self, preds: Dict, targs: Dict, **kwargs) -> Tuple[torch.Tensor, Dict]:
        final_loss, losses = super().__call__(preds, targs, **kwargs)  # TENSOR(0.), {}
        # ============== OBJ VEL MAG LOSS >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
        box_vel_12d = preds["box_vel_12d"]
        vel_predict = box_vel_12d[:, :3]
        vel_predict_norm = torch.norm(vel_predict, dim=1)
        vel_real = targs[Queries.TARGET_VEL].to(final_loss.device)
        vel_real_norm = torch.norm(vel_real, dim=1)
        loss = torch_f.mse_loss(vel_predict_norm, vel_real_norm).float()
        final_loss += loss

        # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

        losses[self.output_key] = final_loss
        return final_loss, losses

@LOSS.register_module
class AccMagLoss(TensorLoss):
    def __init__(self, **cfg):
        super(AccMagLoss, self).__init__()

        logger.info(f"Construct {type(self).__name__} with lambda: ")

    def __call__(self, preds: Dict, targs: Dict, **kwargs) -> Tuple[torch.Tensor, Dict]:
        final_loss, losses = super().__call__(preds, targs, **kwargs)
        # ============== OBJ ACC MAG LOSS >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
        box_vel_12d = preds["box_vel_12d"]
        acc_predict = box_vel_12d[:, 6:9]
        acc_predict_norm = torch.norm(acc_predict, dim=1)
        acc_real = targs[Queries.TARGET_ACC].to(final_loss.device)
        acc_real_norm = torch.norm(acc_real, dim=1)
        loss = torch_f.mse_loss(acc_predict_norm, acc_real_norm).float()
        final_loss += loss

        # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

        losses[self.output_key] = final_loss
        return final_loss, losses