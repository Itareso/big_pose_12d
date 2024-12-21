import os
from collections import OrderedDict
from typing import Dict, Optional

import torch
import torch.nn as nn

from anakin.utils.builder import HEAD, MODEL, build_backbone, build_head, build_model
from anakin.utils.logger import logger
from anakin.utils.misc import enable_lower_param, param_size
from ..utils.transform import batch_uvd2xyz, compute_rotation_matrix_from_ortho6d, aa_to_rotmat, rotmat_to_rot6d
from anakin.datasets.hoquery import Queries
from anakin.utils.misc import CONST
from .simplebaseline import IntegralDeconvHead, norm_heatmap, integral_heatmap3d
from numpy import isnan
from roma.mappings import rotmat_to_rotvec

@MODEL.register_module
class HybridBaseline(nn.Module):
    @enable_lower_param
    def __init__(self, **cfg):
        super(HybridBaseline, self).__init__()

        if cfg["BACKBONE"]["PRETRAINED"] and cfg["PRETRAINED"]:
            logger.warning(
                f"{type(self).__name__}'s backbone {cfg['BACKBONE']['TYPE']} weights will be rewritten by {cfg['PRETRAINED']}"
            )
        self.num_tasks = cfg["NTASKS"]
        self.direct = cfg["DIRECT"]
        self.weights = torch.nn.Parameter(torch.ones(self.num_tasks).float())
        self.center_idx = cfg["DATA_PRESET"].get("CENTER_IDX", 9)
        self.inp_res = cfg["DATA_PRESET"]["IMAGE_SIZE"]
        self.frame_num = cfg.get("FRAME_NUM", 3)
        self.frame_mid = self.frame_num // 2

        self.backbone = build_backbone(cfg["BACKBONE"], default_args=cfg["DATA_PRESET"])

        self.hybrid_head = build_head(cfg["HYBRID_HEAD"], default_args=cfg["DATA_PRESET"])  # IntegralDeconvHead

        self.box_head_kin1 = build_model(cfg["BOX_HEAD_KIN1"], default_args=cfg["DATA_PRESET"])  # box_head, mlp

        self.box_head_kin2 = build_model(cfg["BOX_HEAD_KIN2"], default_args=cfg["DATA_PRESET"])  # box_head, mlp

        #self.box_head_kin3 = build_model(cfg["BOX_HEAD_KIN3"], default_args=cfg["DATA_PRESET"])
        self.refine_net = build_model(cfg["REFINE_NET"], default_args=cfg["DATA_PRESET"])

        self.box_head_pose = build_model(cfg["BOX_HEAD_POSE"], default_args=cfg["DATA_PRESET"])

        self.init_weights(pretrained=cfg["PRETRAINED"])

        logger.info(f"{type(self).__name__} uses center_idx {self.center_idx}")
        logger.info(f"{type(self).__name__} has {param_size(self)}M parameters")
    
    def get_last_shared_layer(self):
        return self.backbone.get_last_shared_layer()
    
    def get_task_weights(self):
        return self.weights
    
    def get_num_tasks(self):
        return self.num_tasks 

    def forward(self, inputs: Dict):
        batch_size, n_channel, height, width = inputs["image"].shape
        #128, 3, 224, 224
        image_list = inputs["image_list"]
        #fnum, 128, 3, 224, 224

        #features[res_layer4].shape=128, 512, 7, 7
        features_list = []
        for i in range(self.frame_num):
            features_list.append(self.backbone(image = image_list[i]))

        pose_results_list = []
        for i in range(self.frame_num):
            pose_results_list.append(self.hybrid_head(feature = features_list[i]["res_layer4"]))

        # box_rot_6d_list = []
        # for i in range(self.frame_num):
        #     box_rot_6d_list.append(self.box_head_pose(torch.cat((features_list[i]["res_layer4_mean"], box_kin_12d), dim=1)))
        box_rot_6d_list = []
        for i in range(self.frame_num):
            box_rot_6d_list.append(self.box_head_pose(features_list[i]["res_layer4_mean"]))
        
        kin_mlp_input = features_list[0]["res_layer4_mean"]
        for i in range(1, self.frame_num):
            kin_mlp_input = torch.cat((kin_mlp_input, features_list[i]["res_layer4_mean"]), dim=1)
        # kin_mlp_input = torch.cat(box_rot_6d_list, dim=1)
        box_kin_12d_intem = self.box_head_kin1(kin_mlp_input)
        
        # box_rot_quat = normalize_quaternion(box_rot_quat)
        # # patch [0,0,0,0] case
        # invalid_mask = torch.all(torch.isclose(box_rot_quat, torch.zeros_like(box_rot_quat)), dim=1)
        # n_invalid = torch.sum(invalid_mask.long())
        # invalid_patch = torch.zeros((n_invalid, 4), dtype=box_rot_quat.dtype, device=box_rot_quat.device)
        # invalid_patch[..., 0] = 1.0  # no grad back
        # box_rot_quat[invalid_mask] = invalid_patch

        pose_3d_abs_list = []
        for i in range(self.frame_num):
            _pose_3d_abs = batch_uvd2xyz(
                uvd=pose_results_list[i]["kp3d"],
                root_joint=inputs[Queries.ROOT_JOINT_LIST][:, i],
                intr=inputs[Queries.CAM_INTR],
                inp_res=self.inp_res,
            ) 
            pose_3d_abs_list.append(_pose_3d_abs)

        joints_3d_abs = pose_3d_abs_list[self.frame_mid][:, 0:21, :]  # TENSOR[B, 21, 3]
        boxroot_3d_abs_list = []
        for i in range(self.frame_num):
            boxroot_3d_abs_list.append(pose_3d_abs_list[i][:, 21:22, :])
        corners_can_3d = inputs[Queries.CORNERS_CAN].to(boxroot_3d_abs_list[0].device)  # TENSOR[B, 8, 3]
        box_rot_rotmat_list = []
        for i in range(self.frame_num):
            box_rot_rotmat_list.append(compute_rotation_matrix_from_ortho6d(box_rot_6d_list[i]))

        boxroot_3d_abs_all = torch.cat(boxroot_3d_abs_list, dim = 2)
        boxroot_3d_abs_all = torch.squeeze(boxroot_3d_abs_all)
        box_rot_6d_all = torch.cat(box_rot_6d_list, dim = 1)
        pose_all = torch.cat((boxroot_3d_abs_all, box_rot_6d_all), dim=1)
        kin_mlp_input_intem = torch.cat((pose_all, box_kin_12d_intem), dim=1)
        #kin_mlp_input_intem = torch.cat((box_rot_6d_all, box_kin_12d_intem), dim=1)
        box_pose_21d = self.box_head_kin2(kin_mlp_input_intem)
        box_kin_12d = box_pose_21d[:, 0:12]
        boxroot_3d_abs_new = box_pose_21d[:, 12:15].view(-1, 1, 3)
        box_rot_6d_new = box_pose_21d[:, 15:21]
        box_rot_rotmat_new = compute_rotation_matrix_from_ortho6d(box_rot_6d_new)

        # vel = box_kin_12d[:, 0:3]
        # omega = box_kin_12d[:, 3:6]
        # acc = box_kin_12d[:, 6:9]
        # beta = box_kin_12d[:, 9:12]
        # ref_boxroot_from_prev = boxroot_3d_abs_all[:, 0:3] + vel / 30 - 0.5 * acc / 30 ** 2
        # ref_boxroot_from_next = boxroot_3d_abs_all[:, 6:9] - vel / 30 - 0.5 * acc / 30 ** 2
        # delta_23 = omega / 30 + 0.5 * beta / 30 ** 2
        # delta_12 = omega / 30 - 0.5 * beta / 30 ** 2
        # rotmat_23 = aa_to_rotmat(delta_23)
        # rotmat_12 = aa_to_rotmat(delta_12)
        # box_rot_rotmat_prev = compute_rotation_matrix_from_ortho6d(box_rot_6d_list[0])
        # box_rot_rotmat_next = compute_rotation_matrix_from_ortho6d(box_rot_6d_list[2])
        # ref_box_rot_rotmat_from_next = torch.matmul(box_rot_rotmat_next, rotmat_23.permute(0, 2, 1))
        # ref_box_rot_rotmat_from_prev = torch.matmul(box_rot_rotmat_prev, rotmat_12)
        # ref_box_rot_6d_from_prev = rotmat_to_rot6d(ref_box_rot_rotmat_from_prev)
        # ref_box_rot_6d_from_next = rotmat_to_rot6d(ref_box_rot_rotmat_from_next)
        # #refine_input = torch.cat((pose_all, box_kin_12d, ref_boxroot_from_prev, ref_boxroot_from_next, ref_box_rot_6d_from_prev, ref_box_rot_6d_from_next), dim=1)
        # refine_input = torch.cat((pose_all, box_kin_12d[:, 0:3], box_kin_12d[:, 6:9], ref_boxroot_from_prev, ref_boxroot_from_next), dim=1)
        # refine_output = self.refine_net(refine_input)
        # boxroot_3d_abs_new = refine_output[:, :3].view(-1, 1, 3)
        # #box_rot_6d_new = refine_output[:, 3:9]
        # box_rot_6d_new = box_rot_6d_list[1]
        # box_rot_rotmat_new = compute_rotation_matrix_from_ortho6d(box_rot_6d_new)


        corners_3d_abs_list = []
        for i in range(self.frame_num):
            corners_3d_abs_list.append(torch.matmul(box_rot_rotmat_list[i], corners_can_3d.permute(0, 2, 1)).permute(0, 2, 1) + boxroot_3d_abs_list[i])

        corners_3d_abs_new = torch.matmul(box_rot_rotmat_new, corners_can_3d.permute(0, 2, 1)).permute(0, 2, 1) + boxroot_3d_abs_new
        # TENSOR[B, 8, 3]

        # dispatch
        root_joint = joints_3d_abs[:, self.center_idx, :]  # (B, 3)
        joints_confd = pose_results_list[self.frame_mid]["kp3d_confd"][:, :21]  # (B, 21)
        # corners_confid = box_rot_quat["kp3d_confd"][:, 1:]  # (B, 8) # TODO: might need do something to this

        cam_intr = inputs[Queries.CAM_INTR].to(corners_3d_abs_list[self.frame_mid].device)  # [B, 3, 3]
        corners_2d = torch.matmul(cam_intr, corners_3d_abs_list[self.frame_mid].permute(0, 2, 1)).permute(0, 2, 1)  # [B, 8, 3], homogeneous
        corners_2d = corners_2d[:, :, 0:2] / corners_2d[:, :, 2:3]  # [B, 8, 2], 2d
        corners_2d[:, :, 0] /= width
        corners_2d[:, :, 1] /= height
        corners_2d_uvd = torch.cat(
            (corners_2d, torch.zeros_like(corners_2d[:, :, 0:1])), dim=2
        )  # [B, 8, 3], where[:, :, 2] all zeros
        final_2d_uvd = torch.cat(
            (pose_results_list[self.frame_mid]["kp3d"][:, 0:21, :], corners_2d_uvd, pose_results_list[self.frame_mid]["kp3d"][:, 21:22, :]), dim=1
        )
        # diff = torch.norm(inputs[Queries.ROOT_JOINT] - root_joint, dim=1)
        # logger.debug(diff)

        return {
            # ↓ absolute value feed to criterion
            "joints_3d_abs": joints_3d_abs,
            "corners_3d_abs": corners_3d_abs_new,
            "corners_3d_abs_ori": corners_3d_abs_list[self.frame_mid],
            "corners_3d_abs_list": corners_3d_abs_list,
            "box_rot_6d": box_rot_6d_new,
            "box_rot_6d_ori": box_rot_6d_list[self.frame_mid],
            "box_rot_6d_list": box_rot_6d_list,
            "box_kin_12d": box_kin_12d,
            "corners_3d_abs_new": corners_3d_abs_new,
            "box_rot_6d_new": box_rot_6d_new,
            #"corners_3d_abs_refine": corners_3d_abs_refine,
            # ↓ root relative valus feed to evaluator
            "joints_3d": joints_3d_abs - root_joint.unsqueeze(1),
            "corners_3d": corners_3d_abs_new - root_joint.unsqueeze(1),
            #"corners_3d": corners_3d_abs_refine - root_joint.unsqueeze(1),
            "2d_uvd": final_2d_uvd,
            "boxroot_3d_abs": boxroot_3d_abs_list[self.frame_mid],
            "box_rot_rotmat": box_rot_rotmat_list[self.frame_mid],
        }

    def init_weights(self, pretrained=""):
        if pretrained == "":
            logger.warning(f"=> Init {type(self).__name__} weights in backbone and head")
            """
            Add init for other modules
            ...
            """
        elif os.path.isfile(pretrained):
            # pretrained_state_dict = torch.load(pretrained)
            logger.info(f"=> Loading {type(self).__name__} pretrained model from: {pretrained}")
            # self.load_state_dict(pretrained_state_dict, strict=False)
            checkpoint = torch.load(pretrained)
            if isinstance(checkpoint, OrderedDict):
                state_dict = checkpoint
            elif isinstance(checkpoint, dict) and "state_dict" in checkpoint:
                state_dict_old = checkpoint["state_dict"]
                state_dict = OrderedDict()
                # delete 'module.' because it is saved from DataParallel module
                for key in state_dict_old.keys():
                    if key.startswith("module."):
                        # state_dict[key[7:]] = state_dict[key]
                        # state_dict.pop(key)
                        state_dict[key[7:]] = state_dict_old[key]  # delete "module." (in nn.parallel)
                    else:
                        state_dict[key] = state_dict_old[key]
            else:
                logger.error(f"=> No state_dict found in checkpoint file {pretrained}")
                raise RuntimeError()
            self.load_state_dict(state_dict, strict=False)
        else:
            logger.error(f"=> No {type(self).__name__} checkpoints file found in {pretrained}")
            raise FileNotFoundError()

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