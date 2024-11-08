import yaml
import sys, os

lr = eval(sys.argv[1])

file_name = f"./config_eval/eval_oakink_clasbased_sym_artiboost9.yaml"

with open(file_name, 'r') as file:
    config = yaml.safe_load(file)

config["DATASET"]["TRAIN"]["SHRINK"] = True
config["DATASET"]["TEST"]["SHRINK"] = True
config["TRAIN"]["LR"] = lr

with open(f"./config_eval/eval_oakink_clasbased_sym_artiboost9.yaml", 'w') as file:
    yaml.safe_dump(config, file)