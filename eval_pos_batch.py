import sys, os
import subprocess

names = [
    ("1942_58", "3"),
]

file = "./config_eval/eval_oakink_clasbased_sym_artiboost1.yaml"
gpu_id = "1"
seq = 1

for name in names:
    date, frame_num = name
    subprocess.run(["python", "yaml_modifier.py", frame_num, date, str(seq)])
    subprocess.run(["python", "eval_oakink.py", date, "--cfg", file, "--gpu_id", gpu_id, "--gpu_render_id", gpu_id])
    subprocess.run(["python", "load_oakink_param.py", "--cfg", file, "--gpu_id", gpu_id, "--gpu_render_id", gpu_id])
    subprocess.run(["python", "eval_pos.py", "predict", date, "--cfg", file, "--gpu_id", gpu_id, "--gpu_render_id", gpu_id])
    subprocess.run(["python", "eval_pos.py", "fromvel", date, "--cfg", file, "--gpu_id", gpu_id, "--gpu_render_id", gpu_id])
    subprocess.run(["python", "eval_pos.py", "frompose", date, "--cfg", file, "--gpu_id", gpu_id, "--gpu_render_id", gpu_id])