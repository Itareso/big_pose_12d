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
from anakin.utils.kinematics import compute_velocity_and_omega, compute_pos_and_rot, get_acc_beta_from_pose, get_acc_beta_from_vel

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
model.eval()

frame_num = cfg["ARCH"]["FRAME_NUM"]

#train_data = builder.build_dataset(cfg["DATASET"]["TRAIN"], preset_cfg=cfg["DATA_PRESET"])
test_data = builder.build_dataset(cfg["DATASET"]["TEST"], preset_cfg=cfg["DATA_PRESET"])
test_loader = torch.utils.data.DataLoader(test_data,
                                        batch_size=1,
                                        shuffle=False,
                                        num_workers=int(arg.workers),
                                        drop_last=False,
                                        collate_fn=ho_collate)



obj_dict = {}

obj_idx = None
obj_name = None
last_info_str = None
id_str = None
last_id_str = None

cur_trans = None
cur_rot = None
target_trans = None
target_rot = None
original_trans = None
original_rot = None
vel_list = []
omega_list = []
acc_list = []
beta_list = []

gt_trans = []
gt_rot = []

predict_trans = []
predict_rot = []

mode = sys.argv[1]

date = sys.argv[2]

use_last = sys.argv[3] == "True"

dataset = sys.argv[4]

if dataset == "oakink":

    obj_id2name_path = "/mnt/public/datasets/OakInk/shape/metaV2"
    save_path = "/mnt/homes/zhushengjia/OakInkDiffNew"
    dataset_path = "/mnt/public/datasets/OakInk"

    obj_id_path = os.path.join(obj_id2name_path, "object_id.json")
    vir_obj_id_path = os.path.join(obj_id2name_path, "virtual_object_id.json")

    obj_id_dict = json.load(open(obj_id_path))
    vir_obj_id_dict = json.load(open(vir_obj_id_path))

    total_obj_dict = {**obj_id_dict, **vir_obj_id_dict}

elif dataset == "dexycb":
    save_path = "/mnt/homes/zhushengjia/DexYCBVel"
    dataset_path = "/mnt/public/datasets/DexYCB"
    total_obj_dict = {}
    objects = os.listdir("./models")
    objects.sort(key = lambda x: int(x[ : 3]))
    for idx, obj_name in enumerate(objects):
        total_obj_dict[idx + 1] = {"name": obj_name}

counter = 0

print(f"start evaluating pose loss in {date} using data from {mode}")

record_acc = True

with open(f"eval_pos/acc_{date}_{mode}_save.txt", "w") as f:
    f.write("")

save_dict = {}

with torch.no_grad():

    for batch_idx, batch in enumerate(test_loader):

        if dataset == "oakink":
            info_str = batch["info_str"][0]
            seq_id = info_str.split("__")[0]
            timestamp = info_str.split("__")[1]
            frame_id = info_str.split("__")[3]
            cam_name = info_str.split("__")[4]
            data_path = os.path.join(save_path, seq_id, timestamp, f"{cam_name}_{frame_id}_predict.npz")
            data = np.load(data_path)
            id_str = seq_id + "__" + timestamp + "__" + cam_name
        elif dataset == "dexycb":
            label_path = batch["label_path"][0]
            data_path = label_path.split(".")[0]+"_predict.npz"
            data_path = os.path.join(save_path, data_path)
            data = np.load(data_path)
            label_list = label_path.split("/")[:-1]
            id_str = "_".join(label_list)

        cur_obj_transf = batch['obj_transf']
        cur_trans = cur_obj_transf[0, :3, 3].detach().cpu().numpy()
        cur_rot = cur_obj_transf[0, :3, 0:3].detach().cpu().numpy()

        predict = model(batch)

        
        vel = data["predict_vel"]
        omega = data["predict_omega"]
        acc = data["predict_acc"]
        beta = data["predict_beta"]

        if mode == "gt" or mode == "gtfromvel":
            vel = batch["target_vel"][0].detach().cpu().numpy()
            omega = batch["target_omega"][0].detach().cpu().numpy()
            acc = batch["target_acc"][0]
            beta = batch["target_beta"][0]

        corner_3d_abs = predict['HybridBaseline']["corners_3d_abs"]
        box_rot_6d = predict['HybridBaseline']["box_rot_6d"]
        _trans, _rot = compute_pos_and_rot(corner_3d_abs, box_rot_6d)

        if last_id_str is None:
            if frame_num != 1:
                original_obj_transf = batch['obj_transf_list'][:,frame_num//2-1]
                original_trans = original_obj_transf[0, :3, 3].detach().cpu().numpy()
                original_rot = original_obj_transf[0, :3, 0:3].detach().cpu().numpy()
                gt_trans = [original_trans]
                gt_rot = [original_rot]
                

                prev_corner_3d_abs = predict['HybridBaseline']["corners_3d_abs_list"][frame_num//2-1]
                prev_box_rot_6d = predict['HybridBaseline']["box_rot_6d_list"][frame_num//2-1]
                prev_trans, prev_rot = compute_pos_and_rot(prev_corner_3d_abs, prev_box_rot_6d)
                predict_trans = [prev_trans]
                predict_rot = [prev_rot]
            else:
                gt_trans = []
                gt_rot = []
                predict_trans = []
                predict_rot = []

        if last_id_str is not None and last_id_str != id_str:
            # print(last_seq_id, seq_id)
            print(last_id_str)
            print(len(acc_list))
            if frame_num != 1:
                gt_trans.append(target_trans)
                gt_rot.append(target_rot)
            gt_trans = np.array(gt_trans)
            gt_rot = np.array(gt_rot)

            if frame_num != 1:
                predict_trans.append(next_trans)
                predict_rot.append(next_rot)
            predict_trans = np.array(predict_trans)
            predict_rot = np.array(predict_rot)
            predict_trans = predict_trans.squeeze(axis=1)
            predict_rot = predict_rot.squeeze(axis=1)

            if mode == "frompose":
                acc_list, beta_list = get_acc_beta_from_pose(predict_trans, predict_rot)
            elif mode == "fromvel" or mode == "gtfromvel":
                #print(vel_list, omega_list)
                acc_list, beta_list = get_acc_beta_from_vel(vel_list, omega_list)
            elif mode == "zeros":
                acc_list = np.zeros((len(acc_list), 3))
                beta_list = np.zeros((len(beta_list), 3))

            if frame_num == 1 and mode == "fromvel":
                acc_list = acc_list[1:-1]
                beta_list = beta_list[1:-1]
            
            if use_last:
                if frame_num == 3:
                    acc_list = acc_list[:-1]
                    beta_list = beta_list[:-1]
                    gt_trans = gt_trans[1:]
                    gt_rot = gt_rot[1:]
                elif frame_num == 5:
                    acc_list = acc_list[:-2]
                    beta_list = beta_list[:-2]
                    gt_trans = gt_trans[2:]
                    gt_rot = gt_rot[2:]
            
            acc_list_tmp = []
            for _acc in acc_list:
                acc_list_tmp.append(_acc.tolist())

            if record_acc:

                with open(f"eval_pos/acc_{date}_{mode}_save.txt", "a") as f:
                    f.write(f"{acc_list_tmp}\n")
            
            info_save = last_id_str

            try:

                trans_loss, rot_loss = eval_object_pos(obj_name, acc_list, beta_list, gt_trans, 
                                gt_rot, last_id_str, mode, dataset)
                
                save_dict[info_save] = {"trans_loss":trans_loss, "rot_loss":rot_loss}
            except:
                print(f"Error:{info_save}_{obj_name}")
            # trans_loss, rot_loss = eval_object_pos(obj_name, acc_list, beta_list, gt_trans, 
            #                     gt_rot, last_seq_id, last_timestamp, last_cam_name, mode)
                
            # save_dict[info_save] = {"trans_loss":trans_loss, "rot_loss":rot_loss}

            if frame_num != 1:
                original_obj_transf = batch['obj_transf_list'][:,frame_num//2-1]
                original_trans = original_obj_transf[0, :3, 3].detach().cpu().numpy()
                original_rot = original_obj_transf[0, :3, 0:3].detach().cpu().numpy()
                gt_trans = [original_trans, cur_trans]
                gt_rot = [original_rot, cur_rot]

                prev_corner_3d_abs = predict['HybridBaseline']["corners_3d_abs_list"][frame_num//2-1]
                prev_box_rot_6d = predict['HybridBaseline']["box_rot_6d_list"][frame_num//2-1]
                prev_trans, prev_rot = compute_pos_and_rot(prev_corner_3d_abs, prev_box_rot_6d)
                predict_trans = [prev_trans, _trans]
                predict_rot = [prev_rot, _rot]
            else:
                gt_trans = [cur_trans]
                gt_rot = [cur_rot]
                predict_trans = [_trans]
                predict_rot = [_rot]

            vel_list = [vel]
            omega_list = [omega]
            acc_list = [acc]
            beta_list = [beta]
            last_id_str = id_str
            counter += 1
            continue

        # if counter == 5:
        #     assert False

        last_id_str = id_str


        vel_list.append(vel)
        omega_list.append(omega)
        acc_list.append(acc)
        beta_list.append(beta)

        predict_trans.append(_trans)
        predict_rot.append(_rot)

        if frame_num != 1:
            target_obj_transf = batch['obj_transf_list'][:,frame_num//2+1]
            target_trans = target_obj_transf[0, :3, 3].detach().cpu().numpy()
            target_rot = target_obj_transf[0, :3, 0:3].detach().cpu().numpy()

            next_corner_3d_abs = predict['HybridBaseline']["corners_3d_abs_list"][frame_num//2+1]
            next_box_rot_6d = predict['HybridBaseline']["box_rot_6d_list"][frame_num//2+1]
            next_trans, next_rot = compute_pos_and_rot(next_corner_3d_abs, next_box_rot_6d)

        gt_trans.append(cur_trans)
        gt_rot.append(cur_rot)

        obj_idx = batch["obj_idx"][0]
        if dataset == "dexycb":
            obj_idx = int(obj_idx)
        obj_name = total_obj_dict[obj_idx]['name']

info_save = last_id_str

try:
    trans_loss, rot_loss = eval_object_pos(obj_name, acc_list, beta_list, gt_trans, 
                                gt_rot, last_id_str, mode, dataset)
    info_save = last_id_str
    save_dict[info_save] = {"trans_loss":trans_loss, "rot_loss":rot_loss}
except:
    print(f"Error:{info_save}_{obj_name}")

save_path = f"eval_pos/eval_pos_{mode}_{date}.json"

with open(save_path, "w") as f:
    json.dump(save_dict, f, indent=4)