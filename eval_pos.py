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
import json
from collections import Counter
from anakin.utils.kinematics import compute_velocity_and_omega, compute_pos_and_rot, get_acc_beta_from_pose

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

model_list = builder.build_arch_model_list(cfg["ARCH"], preset_cfg=cfg["DATA_PRESET"])
model = Arch(cfg, model_list=model_list)
model = torch.nn.DataParallel(model).to(arg.device)

# train_data = builder.build_dataset(cfg["DATASET"]["TRAIN"], preset_cfg=cfg["DATA_PRESET"])
test_data = builder.build_dataset(cfg["DATASET"]["TEST"], preset_cfg=cfg["DATA_PRESET"])
test_loader = torch.utils.data.DataLoader(test_data,
                                        batch_size=1,
                                        shuffle=False,
                                        num_workers=int(arg.workers),
                                        drop_last=False,
                                        collate_fn=ho_collate)

save_path = "/mnt/homes/zhushengjia/OakInkDiffNew"
dataset_path = "/mnt/public/datasets/OakInk"

obj_dict = {}

obj_idx = None
obj_name = None
last_seq_id = None
last_timestamp = None
last_cam_name = None

cur_trans = None
cur_rot = None
target_trans = None
target_rot = None
original_trans = None
original_rot = None
acc_list = []
beta_list = []

gt_trans = []
gt_rot = []

predict_trans = []
predict_rot = []

obj_id2name_path = "/mnt/public/datasets/OakInk/shape/metaV2"

obj_id_path = os.path.join(obj_id2name_path, "object_id.json")
vir_obj_id_path = os.path.join(obj_id2name_path, "virtual_object_id.json")

obj_id_dict = json.load(open(obj_id_path))
vir_obj_id_dict = json.load(open(vir_obj_id_path))

total_obj_dict = {**obj_id_dict, **vir_obj_id_dict}

counter = 0

mode = "frompose"

save_dict = {}

for batch_idx, batch in enumerate(test_loader):
    info_str = batch["info_str"][0]
    seq_id = info_str.split("__")[0]
    timestamp = info_str.split("__")[1]
    frame_id = info_str.split("__")[3]
    cam_name = info_str.split("__")[4]

    cur_obj_transf = batch['obj_transf']
    cur_trans = cur_obj_transf[0, :3, 3].detach().cpu().numpy()
    cur_rot = cur_obj_transf[0, :3, 0:3].detach().cpu().numpy()

    predict = model(batch)

    data_path = os.path.join(save_path, seq_id, timestamp, f"{cam_name}_{frame_id}_predict.npz")
    data = np.load(data_path)

    acc = data["predict_acc"]
    beta = data["predict_beta"]
    if mode == "gt":
        acc = batch["target_acc"][0]
        beta = batch["target_beta"][0]

    corner_3d_abs = predict['HybridBaseline']["corners_3d_abs"]
    box_rot_6d = predict['HybridBaseline']["box_rot_6d"]
    _trans, _rot = compute_pos_and_rot(corner_3d_abs, box_rot_6d)

    if last_seq_id is None:
        original_obj_transf = batch['prev_obj_transf']
        original_trans = original_obj_transf[0, :3, 3].detach().cpu().numpy()
        original_rot = original_obj_transf[0, :3, 0:3].detach().cpu().numpy()
        gt_trans = [original_trans]
        gt_rot = [original_rot]

        prev_corner_3d_abs = predict['HybridBaseline']["prev_corners_3d_abs"]
        prev_box_rot_6d = predict['HybridBaseline']["prev_box_rot_6d"]
        prev_trans, prev_rot = compute_pos_and_rot(prev_corner_3d_abs, prev_box_rot_6d)
        predict_trans = [prev_trans]
        predict_rot = [prev_rot]

    if last_seq_id is not None and (last_seq_id != seq_id or last_timestamp != timestamp or last_cam_name != cam_name):
        print(last_seq_id, seq_id)
        print(len(acc_list))
        gt_trans.append(target_trans)
        gt_rot.append(target_rot)
        gt_trans = np.array(gt_trans)
        gt_rot = np.array(gt_rot)

        predict_trans.append(next_trans)
        predict_rot.append(next_rot)
        predict_trans = np.array(predict_trans)
        predict_rot = np.array(predict_rot)
        predict_trans = predict_trans.squeeze(axis=1)
        predict_rot = predict_rot.squeeze(axis=1)

        if mode == "frompose":
            acc_list, beta_list = get_acc_beta_from_pose(predict_trans, predict_rot)
        
        info_save = f"{last_seq_id}__{last_timestamp}__{last_cam_name}"

        try:

            trans_loss, rot_loss = eval_object_pos(obj_name, acc_list, beta_list, gt_trans, 
                            gt_rot, last_seq_id, last_timestamp, last_cam_name, mode)
            
            save_dict[info_save] = {"trans_loss":trans_loss, "rot_loss":rot_loss}
        except:
            print(f"Error:{info_save}_{obj_name}")

        original_obj_transf = batch['prev_obj_transf']
        original_trans = original_obj_transf[0, :3, 3].detach().cpu().numpy()
        original_rot = original_obj_transf[0, :3, 0:3].detach().cpu().numpy()
        gt_trans = [original_trans, cur_trans]
        gt_rot = [original_rot, cur_rot]

        prev_corner_3d_abs = predict['HybridBaseline']["prev_corners_3d_abs"]
        prev_box_rot_6d = predict['HybridBaseline']["prev_box_rot_6d"]
        prev_trans, prev_rot = compute_pos_and_rot(prev_corner_3d_abs, prev_box_rot_6d)
        predict_trans = [prev_trans, _trans]
        predict_rot = [prev_rot, _rot]

        acc_list = [acc]
        beta_list = [beta]
        last_seq_id = seq_id
        last_timestamp = timestamp
        last_cam_name = cam_name
        counter += 1
        continue

    # if counter == 5:
    #     assert False

    last_seq_id = seq_id
    last_timestamp = timestamp
    last_cam_name = cam_name


    acc_list.append(acc)
    beta_list.append(beta)

    predict_trans.append(_trans)
    predict_rot.append(_rot)

    target_obj_transf = batch['next_obj_transf']
    target_trans = target_obj_transf[0, :3, 3].detach().cpu().numpy()
    target_rot = target_obj_transf[0, :3, 0:3].detach().cpu().numpy()

    next_corner_3d_abs = predict['HybridBaseline']["next_corners_3d_abs"]
    next_box_rot_6d = predict['HybridBaseline']["next_box_rot_6d"]
    next_trans, next_rot = compute_pos_and_rot(next_corner_3d_abs, next_box_rot_6d)

    gt_trans.append(cur_trans)
    gt_rot.append(cur_rot)

    obj_idx = batch["obj_idx"][0]
    obj_name = total_obj_dict[obj_idx]['name']

try:
    trans_loss, rot_loss = eval_object_pos(obj_name, acc_list, beta_list, gt_trans, 
                                gt_rot, last_seq_id, last_timestamp, last_cam_name, mode)
    info_save = f"{last_seq_id}__{last_timestamp}__{last_cam_name}"
    save_dict[info_save] = {"trans_loss":trans_loss, "rot_loss":rot_loss}
except:
    print(f"Error:{info_save}_{obj_name}")

save_path = "./eval_pos_frompose_1654.json"

with open(save_path, "w") as f:
    json.dump(save_dict, f, indent=4)