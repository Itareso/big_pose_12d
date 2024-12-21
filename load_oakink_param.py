import torch
from anakin.models.arch import Arch
from anakin.opt import arg, cfg
from anakin.utils import builder
from anakin.datasets.hodata import ho_collate
import random
import numpy as np
from matplotlib import pyplot as plt
import os
from anakin.utils.transform import batch_ref_bone_len, compute_rotation_matrix_from_ortho6d
from roma.mappings import rotmat_to_rotvec
from scipy.spatial.transform import Rotation as R
import re

def set_all_seeds(seed):
    torch.cuda.manual_seed_all(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

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

# compute position and quaternion from corner points and orthogonal 6d pose
def compute_pos_and_rot(corners3d, ortho6d):
    center = corners3d.mean(1)
    center = center.detach().cpu().numpy()
    rot = compute_rotation_matrix_from_ortho6d(ortho6d)
    rot = rot.detach().cpu().numpy()
    quat = R.from_matrix(rot).as_quat()
    return center, quat

set_all_seeds(cfg["TRAIN"]["MANUAL_SEED"])

model_list = builder.build_arch_model_list(cfg["ARCH"], preset_cfg=cfg["DATA_PRESET"])
model = Arch(cfg, model_list=model_list)
model = torch.nn.DataParallel(model).to(arg.device)
model.eval()

train_data = builder.build_dataset(cfg["DATASET"]["TRAIN"], preset_cfg=cfg["DATA_PRESET"])
test_data = builder.build_dataset(cfg["DATASET"]["TEST"], preset_cfg=cfg["DATA_PRESET"])
test_loader = torch.utils.data.DataLoader(train_data,
                                        batch_size=arg.batch_size,
                                        shuffle=False,
                                        num_workers=int(arg.workers),
                                        drop_last=False,
                                        collate_fn=ho_collate)

target_data = None
predict_data = None
predict_pos2vel = None
predict_pos2omega = None
predict_pos2acc = None
predict_pos2beta = None
gt_pos = np.empty((0, 3))
gt_quat = np.empty((0, 4))
predict_pos = np.empty((0, 3))
predict_quat = np.empty((0, 4))

#save_path = "/mnt/homes/zhushengjia/OakInkDiffNew"
save_path = "/mnt/homes/zhushengjia/OakInkPred"
dataset_path = "/mnt/public/datasets/OakInk"

obj_dict = {}

min_batch_idx = 0
max_batch_idx = 1

print("start loading oakink params")

with torch.no_grad():

    for batch_idx, batch in enumerate(test_loader):
        # if batch_idx < min_batch_idx:
        #     continue
        # if batch_idx > max_batch_idx:
        #     break
        target_vel = batch['target_vel']
        target_omega = batch['target_omega']
        target_acc = batch['target_acc']
        target_beta = batch['target_beta']
        target_obj_transf = batch['obj_transf']
        #data_mean, data_std = batch["kin_data_mean"], batch["kin_data_std"]
        target_6d = torch.cat((target_vel, target_omega, target_acc, target_beta), dim=1)
        predict = model(batch)
        predict_6d = predict['HybridBaseline']['box_kin_12d']
        #data_mean, data_std = data_mean.to(predict_6d.device), data_std.to(predict_6d.device)
        # predict_6d = predict_6d * data_std + data_mean
        predict_vel = predict_6d[:, 0:3]
        predict_omega = predict_6d[:, 3:6]
        predict_acc = predict_6d[:, 6:9]
        predict_beta = predict_6d[:, 9:12]

        corner_3d_abs = predict['HybridBaseline']["corners_3d_abs"]
        box_rot_6d = predict['HybridBaseline']["box_rot_6d"]
        center = corner_3d_abs.mean(1).detach().cpu().numpy()
        rot = compute_rotation_matrix_from_ortho6d(box_rot_6d).detach().cpu().numpy()
        corner_3d_abs = corner_3d_abs.detach().cpu().numpy()
        box_rot_6d = box_rot_6d.detach().cpu().numpy()

        # if batch_idx >= min_batch_idx and batch_idx <= max_batch_idx:
        #     print(target_acc, predict_acc)

        # if predict_pos2vel is None:
        #     predict_pos2vel = vel1
        #     predict_pos2omega = omega1
        #     predict_pos2acc = acc
        #     predict_pos2beta = beta
        # else:
        #     predict_pos2vel = torch.cat((predict_pos2vel, vel1), dim=0)
        #     predict_pos2omega = torch.cat((predict_pos2omega, omega1), dim=0)
        #     predict_pos2acc = torch.cat((predict_pos2acc, acc), dim=0)
        #     predict_pos2beta = torch.cat((predict_pos2beta, beta), dim=0)

        #obj_id = batch["obj_idx"].detach().cpu().numpy()
        #grasp_idx = batch['grasp_idx']
        # if target_data is None:
        #     target_data = target_6d
        #     predict_data = predict_6d
        # else:
        #     target_data = torch.cat((target_data, target_6d), dim=0)
        #     predict_data = torch.cat((predict_data, predict_6d), dim=0)
        
        #label_paths = batch["label_path"]
        #print(label_paths)

        # _predict_pos, _predict_quat = compute_pos_and_rot(corner_3d_abs, box_rot_6d)
        # predict_pos = np.append(predict_pos, _predict_pos, axis=0)
        # predict_quat = np.append(predict_quat, _predict_quat, axis=0)
        # for i, obj_transf in enumerate(target_obj_transf):
        #     trans = target_obj_transf[i, :3, 3].detach().cpu().numpy()
        #     rot = target_obj_transf[i, :3, 0:3].detach().cpu().numpy()
        #     quat = R.from_matrix(rot).as_quat()
        #     trans = np.expand_dims(trans, axis=0)
        #     quat = np.expand_dims(quat, axis=0)
        #     #print(trans.shape, quat.shape)
        #     gt_pos = np.append(gt_pos, trans, axis=0)
        #     gt_quat = np.append(gt_quat, quat, axis=0)
        info_strs = batch["info_str"]
        for i, info_str in enumerate(info_strs):
            #print(info_str)
            seq_id = info_str.split("__")[0]
            timestamp = info_str.split("__")[1]
            cam_name = info_str.split("__")[4]
            frame_id = info_str.split("__")[3]
            final_save_dir = os.path.join(save_path, seq_id, timestamp)
            save_file_name = f"{cam_name}_{frame_id}_predict.npz"
            if not os.path.exists(final_save_dir):
                os.makedirs(final_save_dir)
            final_save_path = os.path.join(final_save_dir, save_file_name)
            #np.savez(final_save_path, predict_vel=predict_vel[i].detach().cpu().numpy(), predict_omega=predict_omega[i].detach().cpu().numpy(), predict_acc=predict_acc[i].detach().cpu().numpy(), predict_beta=predict_beta[i].detach().cpu().numpy())
            #print(predict_acc[i], target_acc[i])
            np.savez(final_save_path, pred_trans = center[i], pred_rot = rot[i], pred_corner = corner_3d_abs[i], pred_boxrot = box_rot_6d[i])
        print(f"save success: batch_idx: {batch_idx}")

# predict_frompos = torch.cat((predict_pos2vel, predict_pos2omega, predict_pos2acc, predict_pos2beta), dim=1)

# tags = ["vel_x", "vel_y", "vel_z", "omega_x", "omega_y", "omega_z", "acc_x", "acc_y", "acc_z", "beta_x", "beta_y", "beta_z"]

# for i, tag in enumerate(tags):
#     plt.plot(target_data[:, i].detach().cpu().numpy(), label=f"target_{tag}")
#     plt.plot(predict_data[:, i].detach().cpu().numpy(), label=f"predict_{tag}")
#     #plt.plot(predict_frompos[:, i].detach().cpu().numpy(), label=f"predict_frompos_{tag}")
#     plt.legend()
#     plt.savefig(f"images/output_{tag}.png")
#     plt.cla()

# print(gt_pos.shape, predict_pos.shape)

# # print(gt_pos[:50, 1])
# # print(predict_pos[:50, 1])

# _axis = ["x", "y", "z"]

# for i, axis in enumerate(_axis):
#     plt.plot(predict_pos[:, i], label=f"predict_pos_{axis}")
#     plt.plot(gt_pos[:, i], label=f"gt_pos_{axis}")
#     plt.legend()
#     plt.savefig(f"images/output_pos_{axis}.png")
#     plt.cla()


# predict_pos2vel = np.diff(predict_pos, axis=0) * 30
# gt_pos2vel = np.diff(gt_pos, axis=0) * 30

# # kill outliers between different images
# for i in range(len(gt_pos2vel)):
#     for j in range(3):
#         if abs(gt_pos2vel[i, j]) > 2:
#             gt_pos2vel[i, j] = 0
#         if abs(predict_pos2vel[i, j]) > 2:
#             predict_pos2vel[i, j] = 0

# for i, axis in enumerate(_axis):
#     plt.plot(predict_pos2vel[:, i], label=f"predict_pos2vel_{axis}")
#     plt.plot(gt_pos2vel[:, i], label=f"gt_pos2vel_{axis}")
#     plt.legend()
#     plt.savefig(f"images/output_pos2vel_{axis}.png")
#     plt.cla()

# for i, axis in enumerate(_axis):
#     plt.plot(predict_quat[:, i], label=f"predict_quat_{axis}")
#     plt.plot(gt_quat[:, i], label=f"gt_quat_{axis}")
#     plt.legend()
#     plt.savefig(f"images/output_quat_{axis}.png")
#     plt.cla()