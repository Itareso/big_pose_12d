import torch
import torch.nn as nn
import torch.optim as optim
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

# Define the policy network
class PolicyNetwork(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(PolicyNetwork, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, output_dim),
        )
        self.log_std = nn.Parameter(torch.zeros(1, output_dim))  # Learnable log standard deviation

    def forward(self, x):
        mu = self.fc(x)
        std = torch.exp(self.log_std)  # Standard deviation is positive
        return mu, std

def select_action(policy_net, state):
    state = torch.FloatTensor(state).unsqueeze(0)  # Add batch dimension
    mu, std = policy_net(state)
    dist = torch.distributions.Normal(mu, std)  # Create a normal distribution
    action = dist.sample()  # Sample action
    return action.detach().numpy()[0], dist.log_prob(action).sum()  # Return action and log probability

frame_num = cfg["ARCH"]["FRAME_NUM"]

model = PolicyNetwork(frame_num * 9, 3)

optimizer = optim.Adam(model.parameters(), lr=0.01)

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

dataset = "oakink"

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


record_acc = True

save_dict = {}



for epoch_idx in range(50):

    rot_loss_record = []

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

        if last_id_str is None:
            if frame_num != 1:
                original_obj_transf = batch['obj_transf_list'][:,frame_num//2-1]
                original_trans = original_obj_transf[0, :3, 3].detach().cpu().numpy()
                original_rot = original_obj_transf[0, :3, 0:3].detach().cpu().numpy()
                gt_trans = [original_trans]
                gt_rot = [original_rot]
                
            else:
                gt_trans = []
                gt_rot = []

        if last_id_str is not None and last_id_str != id_str:
            # print(last_seq_id, seq_id)
            #print(last_id_str)
            if frame_num != 1:
                gt_trans.append(target_trans)
                gt_rot.append(target_rot)
            gt_trans = np.array(gt_trans)
            gt_rot = np.array(gt_rot)

            gt_rot_tensor = torch.FloatTensor(gt_rot)
            log_probs = []
            for idx in range(1, len(gt_rot)-1):
                model_input = torch.cat((gt_rot_tensor[idx-1], gt_rot_tensor[idx], gt_rot_tensor[idx+1])).reshape(-1)
                omega_tmp, log_prob = select_action(model, model_input)
                omega_list.append(omega_tmp)
                log_probs.append(log_prob)
            vel_list = omega_list
            acc_list, beta_list = get_acc_beta_from_vel(vel_list, omega_list)
            info_save = last_id_str

            try:

                trans_loss, rot_loss = eval_object_pos(obj_name, acc_list, beta_list, gt_trans, 
                                gt_rot, last_id_str, "train_ang", dataset)
                
                rot_loss_record.append(rot_loss)
            except:
                pass
                #print(f"Error:{info_save}_{obj_name}")
            # trans_loss, rot_loss = eval_object_pos(obj_name, acc_list, beta_list, gt_trans, 
            #                     gt_rot, last_seq_id, last_timestamp, last_cam_name, mode)
                
            # save_dict[info_save] = {"trans_loss":trans_loss, "rot_loss":rot_loss}
            #print(rot_loss)

            optimizer.zero_grad()
            rot_loss = torch.tensor(rot_loss)
            # print(log_probs[0].requires_grad)
            # print(rot_loss.requires_grad)
            loss = rot_loss * torch.sum(torch.stack(log_probs))
            loss.backward()
            optimizer.step()

            if frame_num != 1:
                original_obj_transf = batch['obj_transf_list'][:,frame_num//2-1]
                original_trans = original_obj_transf[0, :3, 3].detach().cpu().numpy()
                original_rot = original_obj_transf[0, :3, 0:3].detach().cpu().numpy()
                gt_trans = [original_trans, cur_trans]
                gt_rot = [original_rot, cur_rot]

            else:
                gt_trans = [cur_trans]
                gt_rot = [cur_rot]


            omega_list = []
            last_id_str = id_str
            counter += 1
            continue

        # if counter == 5:
        #     assert False

        last_id_str = id_str

        if frame_num != 1:
            target_obj_transf = batch['obj_transf_list'][:,frame_num//2+1]
            target_trans = target_obj_transf[0, :3, 3].detach().cpu().numpy()
            target_rot = target_obj_transf[0, :3, 0:3].detach().cpu().numpy()

        gt_trans.append(cur_trans)
        gt_rot.append(cur_rot)

        obj_idx = batch["obj_idx"][0]
        if dataset == "dexycb":
            obj_idx = int(obj_idx)
        obj_name = total_obj_dict[obj_idx]['name']

    print(f"epoch idx={epoch_idx}, rot loss={sum(rot_loss_record)/len(rot_loss_record)}")

torch.save(model.state_dict(), 'checkpoints/train_ang_parameters.pth')
