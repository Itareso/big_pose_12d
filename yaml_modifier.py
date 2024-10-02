import yaml
import sys, os

frame_num = int(sys.argv[1])
date = sys.argv[2]
seq = sys.argv[3]

print(f"modify yaml for frame_num {frame_num} and date {date}")

file_name = "./config_eval/eval_oakink_clasbased_sym_artiboost.yaml"

with open(file_name, 'r') as file:
    config = yaml.safe_load(file)

config["DATASET"]["TRAIN"]["FRAME_NUM"] = frame_num
config["DATASET"]["TEST"]["FRAME_NUM"] = frame_num
config["ARCH"]["FRAME_NUM"] = frame_num
config["ARCH"]["PRETRAINED"] = f"checkpoints/HybridBaseline{date}.pth.tar"
config["ARCH"]["BOX_HEAD_KIN"]["LAYERS_N"] = [frame_num * 512, 256, 128]

if seq == "1":
    with open("./config_eval/eval_oakink_clasbased_sym_artiboost1.yaml", 'w') as file:
        yaml.safe_dump(config, file)
elif seq == "2":
    with open("./config_eval/eval_oakink_clasbased_sym_artiboost2.yaml", 'w') as file:
        yaml.safe_dump(config, file)