from matplotlib import pyplot as plt
import numpy as np
import ast

eval_path1 = "/mnt/homes/zhushengjia/oakink_big_pose/exp/default_2024_0821_1430_12/evaluations/test_eval.txt"

eval_path2 = "/mnt/homes/zhushengjia/oakink_big_pose/exp/default_2024_0828_2043_52/evaluations/test_eval.txt"


eval_paths = [eval_path1, eval_path2]

metrics = ["Hand3DPCKMetric", "LossesMetric", "Mean3DEPE"]

sub_metrics = {
    "Hand3DPCKMetric": ["auc_all", "epe_mean_all"],
    "LossesMetric": ["acc_cos_loss_output", "acc_mag_loss_output", "final_loss",
                     "hand_ord_loss_output", "joints_loss_output", "part_ord_loss",
                     "sym_corner_loss_output", "vel_cos_loss_output", "vel_mag_loss_output",
                     "m_s_e_omega_loss_output", "m_s_e_beta_loss_output", "acc_consistency_loss_output",
                     "beta_consistency_loss_output"],
    "Mean3DEPE": ["corners_3d_abs_mepe", "joints_3d_abs_mepe"]
}

eval_list = {}

for metric in metrics:
    for sub_metric in sub_metrics[metric]:
        eval_list[f"{sub_metric}"] = []

parsed_data = []

for eval_path in eval_paths:

    with open(eval_path, "r") as f:
        content = f.read()

    epochs = content.split('Epoch')

    for epoch in epochs:
        if epoch.strip():
            lines = epoch.strip().split('evaluator msg:\n', 1)
            epoch_number = lines[0].strip()
            dict_str = lines[1].strip()
            
            data_dict = ast.literal_eval(dict_str)
            
            parsed_data.append((epoch_number, data_dict))

for epoch, data in parsed_data:
    for metric in metrics:
        for sub_metric in sub_metrics[metric]:
            if sub_metric in data[metric]:
                eval_list[f"{sub_metric}"].append(data[metric][sub_metric])
            else:
                eval_list[f"{sub_metric}"].append(0)

for metric in metrics:
    for sub_metric in sub_metrics[metric]:
        plt.plot(eval_list[f"{sub_metric}"], label=sub_metric)
        plt.xlabel("x5 Epoch")
        plt.legend()
        plt.savefig(f"eval_metrics/{sub_metric}.png")
        plt.cla()