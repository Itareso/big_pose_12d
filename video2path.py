import yaml, os, json
from pathlib import Path

dataset_path = "/mnt/public/datasets/DexYCB/bop/s0/test"

test_files = os.listdir(dataset_path)

save_dict = {}

for test_id in test_files:
    test_path = os.path.join(dataset_path, test_id)
    assert os.path.islink(test_path)
    real_path = os.readlink(test_path)
    real_path = os.path.join(dataset_path, real_path)
    video_id = real_path[-4:]
    real_path = os.path.join(real_path, "depth")
    depth_path = os.path.join(real_path, "000000.png")
    real_depth_path = os.readlink(depth_path)
    real_depth_path = os.path.join(real_path, real_depth_path)
    real_path = os.path.dirname(real_depth_path)
    real_path = Path(real_path).resolve()
    save_dict[str(real_path)] = video_id

with open("dexycb_video2path.json", "w") as f:
    json.dump(save_dict, f, indent = 4)