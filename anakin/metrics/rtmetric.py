import functools
from abc import ABC, abstractmethod
from typing import Dict, List

import numpy as np
import torch
from anakin.datasets.hoquery import Queries
from anakin.metrics.metric import AverageMeter, Metric
from anakin.utils.builder import METRIC
from anakin.utils.logger import logger
from ..utils.transform import batch_ref_bone_len, compute_rotation_matrix_from_ortho6d, rotmat_to_quat
from scipy.spatial.transform import Rotation as R

@METRIC.register_module
class RTmetric(Metric):
    def __init__(self, **cfg) -> None:
        super(RTmetric, self).__init__()
        self.rdiff = []
        self.tdiff = []

        self.reset()

    def reset(self):
        self.rdiff = []
        self.tdiff = []

    def feed(self, preds: Dict, targs: Dict, **kwargs):
        try:
            corner_3d_abs = preds["corners_3d_abs_new"]
            box_rot_6d = preds["box_rot_6d_new"]
        except:
            corner_3d_abs = preds["corners_3d_abs"]
            box_rot_6d = preds["box_rot_6d"]
        
        T_pred = corner_3d_abs.mean(1)
        R_pred = compute_rotation_matrix_from_ortho6d(box_rot_6d)
        corners_3d_gt = targs[Queries.CORNERS_3D]
        root_joint_gt = targs[Queries.ROOT_JOINT]
        corner_3d_abs_gt = corners_3d_gt + root_joint_gt.unsqueeze(1)
        T_gt = corner_3d_abs_gt.mean(1)
        R_gt = targs[Queries.OBJ_TRANSF][:, :3, :3]
        T_pred = T_pred.detach().cpu().numpy()
        R_pred = R_pred.detach().cpu().numpy()
        T_gt = T_gt.detach().cpu().numpy()
        R_gt = R_gt.detach().cpu().numpy()
        R21 = R_gt @ R_pred.transpose(0, 2, 1)
        rotangles = np.linalg.norm(R.from_matrix(R21).as_rotvec(), axis = 1)
        T_diff = np.linalg.norm(T_pred - T_gt, axis = 1)
        self.rdiff.extend(list(rotangles))
        self.tdiff.extend(list(T_diff))



    def get_measures(self, **kwargs) -> Dict[str, float]:
        """
        Args:
            **kwargs:

        Returns:
            eg: {joints_3d_abs_mepe : 22.0, }

        """
        measures = {}
        measures["tdiff"] = sum(self.tdiff) / len(self.tdiff) if len(self.tdiff) > 0 else 0
        measures["rdiff"] = sum(self.rdiff) / len(self.rdiff) if len(self.rdiff) > 0 else 0

        return measures

    def __str__(self):
        return "tdiff | rdiff"