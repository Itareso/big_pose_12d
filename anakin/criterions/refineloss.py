from typing import Dict, Tuple

import torch
import torch.nn.functional as torch_f
from anakin.datasets.hoquery import Queries
from anakin.utils.builder import LOSS
from anakin.utils.logger import logger
from anakin.utils.misc import CONST

from .criterion import TensorLoss


@LOSS.register_module
class RefineLoss(TensorLoss):

    def __init__(self, **cfg):
        super(RefineLoss, self).__init__()

    def __call__(self, preds: Dict, targs: Dict, **kwargs) -> Tuple[torch.Tensor, Dict]:
        final_loss, losses = super().__call__(preds, targs, **kwargs)  # TENSOR(0.), {}
        # ============== OBJ CORNERS 3D MSE LOSS >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
        corners_3d_loss = torch.tensor(0.0, device=final_loss.device)
        pred_corners_3d_abs_refine = preds["corners_3d_abs_pred"]
        corners_3d = targs[Queries.CORNERS_3D]
        root_joint = targs[Queries.ROOT_JOINT]
        corners_vis_mask = targs[Queries.CORNERS_VIS]
        pred_corners_3d_abs = torch.einsum("bij,bi->bij", pred_corners_3d_abs_refine,
                                        corners_vis_mask.to(final_loss.device))
        corners_3d_abs = corners_3d + root_joint.unsqueeze(1)
        corners_3d_abs = torch.einsum("bij,bi->bij", corners_3d_abs, corners_vis_mask)

        corners_3d_loss += torch_f.mse_loss(pred_corners_3d_abs, corners_3d_abs.to(final_loss.device))
        losses["refine_loss"] = corners_3d_loss
        final_loss += corners_3d_loss
        # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

        losses[self.output_key] = final_loss
        return final_loss, losses