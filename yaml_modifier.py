import yaml
import sys, os

frame_num = int(sys.argv[1])
date = sys.argv[2]
seq = sys.argv[3]
dataset = sys.argv[4]

print(f"modify yaml for frame_num {frame_num} and date {date}")

file_name = f"./config_eval/eval_{dataset}_clasbased_sym_artiboost.yaml"

with open(file_name, 'r') as file:
    config = yaml.safe_load(file)

config["DATASET"]["TRAIN"]["FRAME_NUM"] = frame_num
config["DATASET"]["TEST"]["FRAME_NUM"] = frame_num
config["DATASET"]["TRAIN"]["SHRINK"] = False
config["DATASET"]["TEST"]["SHRINK"] = False
config["ARCH"]["FRAME_NUM"] = frame_num
config["ARCH"]["PRETRAINED"] = f"checkpoints/HybridBaseline{date}.pth.tar"
config["ARCH"]["BOX_HEAD_KIN1"]["LAYERS_N"] = [frame_num * 512, 256, 128]

with open(f"./config_eval/eval_{dataset}_clasbased_sym_artiboost{seq}.yaml", 'w') as file:
    yaml.safe_dump(config, file)