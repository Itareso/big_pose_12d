import sys, os
import subprocess
import yaml

file_name = f"./config_eval/eval_oakink_clasbased_sym_artiboost9.yaml"

for layer in [
    [18, 36],
    [18, 64, 36],
]:
    with open(file_name, 'r') as file:
        config = yaml.safe_load(file)

    config["DATASET"]["TRAIN"]["SHRINK"] = True
    config["DATASET"]["TEST"]["SHRINK"] = True
    config["ARCH"]["BOX_HEAD_KIN"]["LAYERS_N"] = layer

    with open(f"./config_eval/eval_oakink_clasbased_sym_artiboost9.yaml", 'w') as file:
        yaml.safe_dump(config, file)
    
    subprocess.run(["python", "train_artiboost.py", "--cfg", file_name, "--gpu_id", "6", "--gpu_render_id", "6"])
