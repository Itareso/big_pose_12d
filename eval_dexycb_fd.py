import torch
from anakin.models.arch import Arch
from anakin.opt import arg, cfg
from anakin.utils import builder
from anakin.datasets.hodata import ho_collate
import random
import numpy as np
from matplotlib import pyplot as plt
import os
import sys
from anakin.utils.transform import batch_ref_bone_len, compute_rotation_matrix_from_ortho6d
from roma.mappings import rotmat_to_rotvec
from scipy.spatial.transform import Rotation as R
import json, yaml
from collections import Counter
from anakin.utils.kinematics import compute_velocity_and_omega, compute_pos_and_rot, get_acc_beta_from_pose
from anakin.criterions.criterion import Criterion
from anakin.datasets.hodata import ho_collate
from anakin.metrics.evaluator import Evaluator

import cv2

from sim import eval_object_pos

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
cv2.setNumThreads(0)

def set_all_seeds(seed):
    torch.cuda.manual_seed_all(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

set_all_seeds(cfg["TRAIN"]["MANUAL_SEED"])

# train_data = builder.build_dataset(cfg["DATASET"]["TRAIN"], preset_cfg=cfg["DATA_PRESET"])
test_data = builder.build_dataset(cfg["DATASET"]["TEST"], preset_cfg=cfg["DATA_PRESET"])
test_loader = torch.utils.data.DataLoader(test_data,
                                        batch_size=arg.batch_size,
                                        shuffle=True,
                                        num_workers=int(arg.workers),
                                        drop_last=False,
                                        collate_fn=ho_collate)

loss_list = builder.build_criterion_loss_list(cfg["CRITERION"],
                                                  preset_cfg=cfg["DATA_PRESET"],
                                                  LAMBDAS=cfg["LAMBDAS"])
criterion = Criterion(cfg, loss_list=loss_list)

metrics_list = builder.build_evaluator_metric_list(cfg["EVALUATOR"], preset_cfg=cfg["DATA_PRESET"])
evaluator = Evaluator(cfg, metrics_list=metrics_list)

counter = 0

crit_dict = {}

evaluator.reset_all()

with open("dexycb_video2path.json", "r") as f:
    video2path = json.load(f)

with open("../FoundationPose/debug/ycbv_res.yml", "r") as f:
    predict_data = yaml.safe_load(f)

print(f"start evaluating")

with torch.no_grad():
    for batch_idx, batch in enumerate(test_loader):

        label_paths = batch["label_path"]
        batch_size = len(label_paths)
        corners_can = batch["corners_can"]
        predict_arch_dict = {}
        corners_3d_abs = torch.empty(batch_size, 8, 3)
        box_rotmat = torch.empty(batch_size, 3, 3)
        box_root = torch.empty(batch_size, 1, 3)
        for idx, label_path in enumerate(label_paths):
            abs_path = os.path.join("/mnt/public/datasets/DexYCB", label_path)
            frame_id = abs_path[-10:-4]
            abs_path = os.path.dirname(abs_path)
            video_id = video2path[abs_path]
            predict_trans = predict_data[int(video_id)][frame_id]
            obj_id = list(predict_trans.keys())[0]
            predict_trans = torch.tensor(predict_trans[obj_id], dtype=torch.float32)
            R, t = predict_trans[:3, :3], predict_trans[:3, 3]
            t = t.reshape(1, 3)
            corners = torch.matmul(R, corners_can[idx].permute(1, 0)).permute(1, 0) + t
            corners_3d_abs[idx] = corners
            box_rotmat[idx] = R
            box_root[idx] = t
        corners_3d_abs = torch.tensor(corners_3d_abs, dtype=torch.float32)
        box_rotmat = torch.tensor(box_rotmat, dtype=torch.float32)
        box_root = torch.tensor(box_root, dtype=torch.float32)
        predict_arch_dict["corners_3d_abs"] = corners_3d_abs
        predict_arch_dict["corners_3d_abs_list"] = [corners_3d_abs, corners_3d_abs, corners_3d_abs]
        predict_arch_dict["box_rot_rotmat"] = box_rotmat
        predict_arch_dict["boxroot_3d_abs"] = box_root

        predicts = {}
        predicts.update(predict_arch_dict)
        final_loss, losses, nan_loss_list, task_loss = criterion.compute_losses(predicts, batch)
        # for key in losses.keys():
        #     if losses[key] is None:
        #         continue
        #     if key not in crit_dict:
        #         crit_dict[key] = 0
        #     crit_dict[key] += losses[key].item()
        evaluator.feed_all(predicts, batch, losses)
        counter += 1
        print("finished", counter)

# for key in crit_dict.keys():
#     crit_dict[key] /= counter

eval_dict = evaluator.get_measures_all()

eval_save_name = "eval_dexycb_fd_model_free.txt"

with open(eval_save_name, "w") as f:
    f.write(str(eval_dict))