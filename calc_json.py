import json
import numpy as np
import sys
import os

file = sys.argv[1]
trans = np.array([])
rot = np.array([])

cwd = os.getcwd()
file = os.path.join(cwd, file)

with open(file) as f:
    data = json.load(f)

for key in data.keys():
    trans = np.append(trans, data[key]["trans_loss"])
    rot = np.append(rot, data[key]["rot_loss"])

print(f"trans means: {np.mean(trans)}")
print(f"rot means: {np.mean(rot)}")