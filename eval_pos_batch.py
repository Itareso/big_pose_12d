import sys, os
import subprocess

names = [
    ("1252", "3", "False"),
]

dataset = "oakink"
file = f"./config_eval/eval_{dataset}_clasbased_sym_artiboost.yaml"
gpu_id = "6"
seq = 6
# use_last = "False"

for name in names:
    date, frame_num, use_last = name
    #subprocess.run(["python", "yaml_modifier.py", frame_num, date, str(seq), dataset])
    #subprocess.run(["python", "eval_oakink.py", date, "--cfg", file, "--gpu_id", gpu_id, "--gpu_render_id", gpu_id])
    # if dataset == "oakink":
    #     subprocess.run(["python", "load_oakink_param.py", "--cfg", file, "--gpu_id", gpu_id, "--gpu_render_id", gpu_id])
    # elif dataset == "dexycb":
    #     subprocess.run(["python", "load_param.py", "--cfg", file, "--gpu_id", gpu_id, "--gpu_render_id", gpu_id])
    # if frame_num != "1":
    #subprocess.run(["python", "eval_pos.py", "predict", date, use_last, dataset, "--cfg", file, "--gpu_id", gpu_id, "--gpu_render_id", gpu_id])
    subprocess.run(["python", "eval_pos.py", "fromvel", date, use_last, dataset, "--cfg", file, "--gpu_id", gpu_id, "--gpu_render_id", gpu_id])
    #subprocess.run(["python", "eval_pos.py", "frompose", date, use_last, dataset, "--cfg", file, "--gpu_id", gpu_id, "--gpu_render_id", gpu_id])