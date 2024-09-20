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

model_list = builder.build_arch_model_list(cfg["ARCH"], preset_cfg=cfg["DATA_PRESET"])
model = Arch(cfg, model_list=model_list)
model = torch.nn.DataParallel(model).to(arg.device)

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

save_path = "/mnt/homes/zhushengjia/OakInkDiffNew"
dataset_path = "/mnt/public/datasets/OakInk"

counter = 0

crit_dict = {}

evaluator.reset_all()

model.eval()

with torch.no_grad():
    for batch_idx, batch in enumerate(test_loader):
        predict_arch_dict = model(batch)
        predicts = {}
        for key in predict_arch_dict.keys():
            predicts.update(predict_arch_dict[key])
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

eval_save_name = "eval_oakink/eval_0859.txt"


with open(eval_save_name, "w") as f:
    f.write(str(eval_dict))