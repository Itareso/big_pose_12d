import json
import os
import pickle
import hashlib

import imageio
import numpy as np
import trimesh
from anakin.utils.common import quat_to_aa, quat_to_rotmat, rotmat_to_aa
from anakin.utils.common import suppress_trimesh_logging
from PIL import Image

from anakin.utils.oikitutils import load_object, load_object_by_id, persp_project

from anakin.datasets.hodata import HOdata
from anakin.utils import transform
from anakin.utils.builder import DATASET
from anakin.utils.etqdm import etqdm
from anakin.utils.logger import logger
from anakin.utils.misc import enable_lower_param, CONST
from anakin.utils.transform import batch_ref_bone_len

ALL_INTENT = {
    "use": "0001",
    "hold": "0002",
    "liftup": "0003",
    "handover": "0004",
}
ALL_INTENT_REV = {_v: _k for _k, _v in ALL_INTENT.items()}

savedir = "/mnt/homes/zhushengjia/OakInkDiffNew"


def decode_seq_cat(seq_cat):
    field_list = seq_cat.split("_")
    obj_id = field_list[0]
    action_id = field_list[1]
    if action_id == "0004":
        subject_id = tuple(field_list[2:4])
    else:
        subject_id = (field_list[2],)
    return obj_id, action_id, subject_id

@DATASET.register_module
class OakInkImage(HOdata):

    @staticmethod
    def _get_info_list(data_dir, split_key, data_split):
        if data_split == "train+val":
            info_list = json.load(open(os.path.join(data_dir, "image", "anno", "split", split_key, "seq_train.json")))
        elif data_split == "train":
            info_list = json.load(
                open(os.path.join(data_dir, "image", "anno", "split_train_val", split_key, "example_split_train.json")))
        elif data_split == "val":
            info_list = json.load(
                open(os.path.join(data_dir, "image", "anno", "split_train_val", split_key, "example_split_val.json")))
        else:  # data_split == "test":
            info_list = json.load(open(os.path.join(data_dir, "image", "anno", "split", split_key, "seq_test.json")))
        return info_list

    @staticmethod
    def _get_info_str(info_item):
        info_str = "__".join([str(x) for x in info_item])
        info_str = info_str.replace("/", "__")
        return info_str

    @staticmethod
    def _get_handover_info(info_list):
        hand_over_map = {}
        hand_over_index_list = []
        for idx, info_item in enumerate(info_list):
            info = info_item[0]
            seq_cat, _ = info.split("/")
            _, action_id, _ = decode_seq_cat(seq_cat)
            if action_id != "0004":
                continue
            sub_id = info_item[1]
            alt_sub_id = 1 if sub_id == 0 else 0  # flip sub_id to get alt_sub_id
            alt_info_item = (info_item[0], alt_sub_id, info_item[2], info_item[3])
            hand_over_map[tuple(info_item)] = alt_info_item
            hand_over_index_list.append(idx)
        # sanity check: all alt_info_item should be in hand_over_map
        for alt_info_item in hand_over_map.values():
            assert alt_info_item in hand_over_map, str(alt_info_item)
        # TODO: extra filter, like to limit for subject_id 0/1
        return hand_over_map, hand_over_index_list

    @enable_lower_param
    def __init__(self, **cfg) -> None:
        super().__init__(**cfg)

        self._name = "OakInkImage"
        self.name = self._name
        self._data_split = cfg["DATA_SPLIT"]
        self._mode_split = cfg["SPLIT_MODE"]
        self._enable_handover = cfg["ENABLE_HANDOVER"]
        if self._enable_handover:
            assert self._data_split == "all", "handover need to be enabled in all split"

        oakink_name = f"{self._data_split}_{self._mode_split}"
        logger.info(f"OakInk use split: {oakink_name}")

        self.cache_identifier_dict = {
            "data_split": self._data_split,
            "split_mode": self._mode_split,
            "cache_version": 0
        }
        self.cache_identifier_raw = json.dumps(self.cache_identifier_dict, sort_keys=True)
        self.cache_identifier = hashlib.md5(self.cache_identifier_raw.encode("ascii")).hexdigest()
        self.cache_path = os.path.join("common", "cache", self._name, "{}.pkl".format(self.cache_identifier))

        self._data_dir = cfg["DATA_ROOT"]

        if self.use_cache and os.path.exists(self.cache_path):
            with open(self.cache_path, "rb") as p_f:
                self.info_list = pickle.load(p_f)
            logger.info(f"Loaded cache for {self._name}_{self._data_split}_{self._mode_split} from {self.cache_path}")
        else:
            if self._data_split == "all":
                self.info_list_raw = json.load(open(os.path.join(self._data_dir, "image", "anno", "seq_all.json")))
            elif self._mode_split == "default":
                self.info_list_raw = self._get_info_list(self._data_dir, "split0", self._data_split)
            elif self._mode_split == "subject":
                self.info_list_raw = self._get_info_list(self._data_dir, "split1", self._data_split)
            elif self._mode_split == "object":
                self.info_list_raw = self._get_info_list(self._data_dir, "split2", self._data_split)
            else:  # self._mode_split == "handobject":
                self.info_list_raw = self._get_info_list(self._data_dir, "split0_ho", self._data_split)

            logger.info("filtering samples")
            self.info_list = []
            counter = 0
            for info in self.info_list_raw:
                image_dir = os.path.join(self._data_dir, "image", "stream_release_v2", info[0])
                image_files = os.listdir(image_dir)
                north_east_files = [f for f in image_files if "north_east_color" in f]
                south_east_files = [f for f in image_files if "south_east_color" in f]
                north_west_files = [f for f in image_files if "north_west_color" in f]
                south_west_files = [f for f in image_files if "south_west_color" in f]
                north_east_files = sorted(north_east_files, key=lambda x: int(x.split("_")[-1].split(".")[0]))
                south_east_files = sorted(south_east_files, key=lambda x: int(x.split("_")[-1].split(".")[0]))
                north_west_files = sorted(north_west_files, key=lambda x: int(x.split("_")[-1].split(".")[0]))
                south_west_files = sorted(south_west_files, key=lambda x: int(x.split("_")[-1].split(".")[0]))
                north_east_min = int(north_east_files[0].split("_")[-1].split(".")[0])
                north_east_max = int(north_east_files[-1].split("_")[-1].split(".")[0])
                south_east_min = int(south_east_files[0].split("_")[-1].split(".")[0])
                south_east_max = int(south_east_files[-1].split("_")[-1].split(".")[0])
                north_west_min = int(north_west_files[0].split("_")[-1].split(".")[0])
                north_west_max = int(north_west_files[-1].split("_")[-1].split(".")[0])
                south_west_min = int(south_west_files[0].split("_")[-1].split(".")[0])
                south_west_max = int(south_west_files[-1].split("_")[-1].split(".")[0])
                if info[3] == 0 and (info[2] <= north_east_min+1 or info[2] >= north_east_max-1):
                    counter += 1
                elif info[3] == 1 and (info[2] <= south_east_min+1 or info[2] >= south_east_max-1):
                    counter += 1
                elif info[3] == 2 and (info[2] <= north_west_min+1 or info[2] >= north_west_max-1):
                    counter += 1
                elif info[3] == 3 and (info[2] <= south_west_min+1 or info[2] >= south_west_max-1):
                    counter += 1
                else:
                    self.info_list.append(info)
            logger.info(f"Filtered {counter} samples")
            with open(self.cache_path, "wb") as p_f:
                pickle.dump(self.info_list, p_f)
            logger.info(f"Wrote cache for {self._name}_{self._data_split}_{self._mode_split} to {self.cache_path}")

        self.info_str_list = []
        for info in self.info_list:
            info_str = "__".join([str(x) for x in info])
            info_str = info_str.replace("/", "__")
            self.info_str_list.append(info_str)

        # load obj
        suppress_trimesh_logging()

        self.obj_mapping = {}
        obj_root = os.path.join(self._data_dir, "image", "obj")
        all_obj_fn = sorted(os.listdir(obj_root))
        for obj_fn in all_obj_fn:
            obj_id = os.path.splitext(obj_fn)[0]
            obj_model = load_object(obj_root, obj_fn)
            self.obj_mapping[obj_id] = obj_model

        self.framedata_color_name = [
            "north_east_color",
            "south_east_color",
            "north_west_color",
            "south_west_color",
        ]

        self._image_size = (848, 480)  # (W, H)
        self._hand_side = "right"

        # seq status
        with open(os.path.join(self._data_dir, "image", "anno", "seq_status.json"), "r") as f:
            self.seq_status = json.load(f)

        # handover
        if self._enable_handover:
            self.handover_info, self.handover_sample_index_list = self._get_handover_info(self.info_list)
            self.handover_info_list = list(self.handover_info.keys())
        else:
            self.handover_info, self.handover_sample_index_list = None, None
            self.handover_info_list = None
        
        

    def __len__(self):
        return len(self.info_list)
    
    def get_info_str(self, idx):
        return self.info_str_list[idx]

    def get_image_path(self, idx, seq = 0):
        info = self.info_list[idx]
        # compute image path
        offset = os.path.join(info[0], f"{self.framedata_color_name[info[3]]}_{info[2] + seq}.png")
        image_path = os.path.join(self._data_dir, "image", "stream_release_v2", offset)
        return image_path

    def get_image(self, idx, seq = 0):
        info = self.info_list[idx]
        offset = os.path.join(info[0], f"{self.framedata_color_name[info[3]]}_{info[2] + seq}.png")
        image_path = os.path.join(self._data_dir, "image", "stream_release_v2", offset)
        image = Image.open(image_path).convert("RGB")
        return image
    
    def get_center_scale_wrt_bbox(self, idx):
        if self.require_full_image:
            full_width, full_height = self._image_size[0], self._image_size[1]
            center = np.array((full_width / 2, full_height / 2))
            scale = full_width
            return center, scale

        if self.crop_model == "hand_obj":
            joints2d = self.get_joints_2d(idx)  # (21, 2)
            corners_2d = self.get_corners_2d(idx)  # (8, 2)
            all2d = np.concatenate([joints2d, corners_2d], axis=0)  # (29, 2)
            center = HOdata.get_annot_center(all2d)
            scale = HOdata.get_annot_scale(all2d)
            return center, scale
        elif self.crop_model == "hand":
            joints2d = self.get_joints_2d(idx)  # (21, 2)
            center = HOdata.get_annot_center(joints2d)
            scale = HOdata.get_annot_scale(joints2d)
            return center, scale
        else:
            raise NotImplementedError()
    
    def get_hand_faces(self, idx):
        raise NotImplementedError()
    
    def get_hand_verts_2d(self, idx):
        raise NotImplementedError()

    def get_hand_verts_3d(self, idx):
        raise NotImplementedError()

    def get_cam_intr(self, idx):
        cam_path = os.path.join(self._data_dir, "image", "anno", "cam_intr", f"{self.info_str_list[idx]}.pkl")
        with open(cam_path, "rb") as f:
            cam_intr = pickle.load(f)
        return cam_intr

    def get_joints_3d(self, idx, seq = 0):
        info = self.info_list[idx][:]
        info[2] += seq
        info_str = self._get_info_str(info)
        joints_path = os.path.join(self._data_dir, "image", "anno", "hand_j", f"{info_str}.pkl")
        with open(joints_path, "rb") as f:
            joints_3d = pickle.load(f)
        return joints_3d

    def get_verts_3d(self, idx):
        verts_path = os.path.join(self._data_dir, "image", "anno", "hand_v", f"{self.info_str_list[idx]}.pkl")
        with open(verts_path, "rb") as f:
            verts_3d = pickle.load(f)
        return verts_3d

    def get_joints_2d(self, idx):
        cam_intr = self.get_cam_intr(idx)
        joints_3d = self.get_joints_3d(idx)
        return persp_project(joints_3d, cam_intr)

    def get_verts_2d(self, idx):
        cam_intr = self.get_cam_intr(idx)
        verts_3d = self.get_verts_3d(idx)
        return persp_project(verts_3d, cam_intr)

    def get_mano_pose(self, idx):
        general_info_path = os.path.join(self._data_dir, "image", "anno", "general_info",
                                         f"{self.info_str_list[idx]}.pkl")
        with open(general_info_path, "rb") as f:
            general_info = pickle.load(f)
        raw_hand_anno = general_info["hand_anno"]

        raw_hand_pose = (raw_hand_anno["hand_pose"]).reshape((16, 4))  # quat (16, 4)
        _wrist, _remain = raw_hand_pose[0, :], raw_hand_pose[1:, :]
        cam_extr = general_info["cam_extr"]  # SE3 (4, 4))
        extr_R = cam_extr[:3, :3]  # (3, 3)

        wrist_R = extr_R.matmul(quat_to_rotmat(_wrist))  # (3, 3)
        wrist = rotmat_to_aa(wrist_R).unsqueeze(0).numpy()  # (1, 3)
        remain = quat_to_aa(_remain).numpy()  # (15, 3)
        hand_pose = np.concatenate([wrist, remain], axis=0)  # (16, 3)

        return hand_pose.astype(np.float32)

    def get_mano_shape(self, idx):
        general_info_path = os.path.join(self._data_dir, "image", "anno", "general_info",
                                         f"{self.info_str_list[idx]}.pkl")
        with open(general_info_path, "rb") as f:
            general_info = pickle.load(f)
        raw_hand_anno = general_info["hand_anno"]
        hand_shape = raw_hand_anno["hand_shape"].numpy().astype(np.float32)
        return hand_shape

    def get_obj_idx(self, idx):
        info = self.info_list[idx][0]
        seq_cat, _ = info.split("/")
        obj_id, _, _ = decode_seq_cat(seq_cat)
        return obj_id

    def get_obj_faces(self, idx):
        obj_id = self.get_obj_idx(idx)
        return np.asarray(self.obj_mapping[obj_id].faces).astype(np.int32)

    def get_obj_transf(self, idx, seq = 0):
        info = self.info_list[idx][:]
        info[2] += seq
        info_str = self._get_info_str(info)
        obj_transf_path = os.path.join(self._data_dir, "image", "anno", "obj_transf", f"{info_str}.pkl")
        with open(obj_transf_path, "rb") as f:
            obj_transf = pickle.load(f)
        return obj_transf.astype(np.float32)

    def get_obj_verts_3d(self, idx):
        obj_verts = self.get_obj_verts_can(idx)
        obj_transf = self.get_obj_transf(idx)
        obj_rot = obj_transf[:3, :3]
        obj_tsl = obj_transf[:3, 3]
        obj_verts_transf = (obj_rot @ obj_verts.transpose(1, 0)).transpose(1, 0) + obj_tsl
        return obj_verts_transf

    def get_obj_verts_2d(self, idx):
        obj_verts_3d = self.get_obj_verts_3d(idx)
        cam_intr = self.get_cam_intr(idx)
        return persp_project(obj_verts_3d, cam_intr)

    def get_obj_verts_can(self, idx):
        obj_id = self.get_obj_idx(idx)
        obj_verts = np.asarray(self.obj_mapping[obj_id].vertices).astype(np.float32)
        return obj_verts
    
    def get_obj_verts_transf(self, idx):
        # * deprecated
        # transf = self._get_raw_obj_transf(idx)
        # R, t = transf[:3, :3], transf[:3, [3]]
        # verts_can = self._get_raw_obj_verts(idx)
        # raw_verts = (R @ verts_can.T + t).T

        transf = self.get_obj_transf(idx)
        R, t = transf[:3, :3], transf[:3, [3]]
        verts_can, _, _ = self.get_obj_verts_can(idx)
        verts = (R @ verts_can.T + t).T

        return verts

    def get_sample_identifier(self, idx):
        res = f"{self._name}__{self.cache_identifier_raw}__{idx}"
        return res

    def get_sides(self, idx):
        return "right"
    
    def get_sample_idxs(self):
        length = len(self.info_list)
        return [i for i in range(length)]

    def get_corners_3d(self, idx, seq = 0):
        obj_corners = self.get_corners_can(idx)
        obj_transf = self.get_obj_transf(idx, seq)
        obj_rot = obj_transf[:3, :3]
        obj_tsl = obj_transf[:3, 3]
        obj_corners_transf = (obj_rot @ obj_corners.transpose(1, 0)).transpose(1, 0) + obj_tsl
        return obj_corners_transf

    def get_corners_2d(self, idx, seq = 0):
        obj_corners = self.get_corners_3d(idx, seq)
        cam_intr = self.get_cam_intr(idx)
        return persp_project(obj_corners, cam_intr)

    def get_corners_can(self, idx):
        obj_id = self.get_obj_idx(idx)
        obj_mesh = self.obj_mapping[obj_id]
        obj_corners = trimesh.bounds.corners(obj_mesh.bounds)
        return np.asfarray(obj_corners, dtype=np.float32)

    def get_corners_vis(self, idx, seq = 0):
        corners_2d = self.get_corners_2d(idx, seq)
        corners_vis = ((corners_2d[:, 0] >= 0) &
                           (corners_2d[:, 0] < self._image_size[0])) & ((corners_2d[:, 1] >= 0) &
                                                                     (corners_2d[:, 1] < self._image_size[1]))
        return corners_vis.astype(np.float32)

    def get_sample_status(self, idx):
        info = self.info_list[idx][0]
        status = self.seq_status[info]
        return status

    def get_intent_mode(self, idx):
        info_item = self.info_list[idx]
        info = info_item[0]
        seq_cat, _ = info.split("/")
        _, action_id, _ = decode_seq_cat(seq_cat)
        intent_mode = ALL_INTENT_REV[action_id]
        return intent_mode

    def get_hand_over(self, idx):
        if self.handover_info is None:
            return None
        info_item = tuple(self.info_list[idx])
        if info_item not in self.handover_info:
            return None

        info = info_item[0]
        status = self.seq_status[info]
        seq_cat, _ = info.split("/")
        _, action_id, _ = decode_seq_cat(seq_cat)
        intent_mode = ALL_INTENT_REV[action_id]

        alt_info_item = self.handover_info[info_item]
        # load by alt_info
        alt_res = self.load_by_info(alt_info_item)
        # rename to alt
        res = {
            "sample_status": status,
            "intent_mode": intent_mode,
        }
        for _k, _v in alt_res.items():
            res[f"alt_{_k}"] = _v
        return res

    def load_by_info(self, info_item):
        info = info_item[0]
        status = self.seq_status[info]
        seq_cat, _ = info.split("/")
        _, action_id, _ = decode_seq_cat(seq_cat)
        intent_mode = ALL_INTENT_REV[action_id]

        info_str = self._get_info_str(info_item)
        joints_path = os.path.join(self._data_dir, "image", "anno", "hand_j", f"{info_str}.pkl")
        with open(joints_path, "rb") as f:
            joints_3d = pickle.load(f)
        verts_path = os.path.join(self._data_dir, "image", "anno", "hand_v", f"{info_str}.pkl")
        with open(verts_path, "rb") as f:
            verts_3d = pickle.load(f)
        return {
            "sample_status": status,
            "intent_mode": intent_mode,
            "joints": joints_3d,
            "verts": verts_3d,
        }

    def get_real_vel(self, idx):
        info = self.info_list[idx]
        offset = f"{self.framedata_color_name[info[3]]}_{info[2]}.npz"
        save_path = os.path.join(savedir, info[0], offset)
        data = np.load(save_path)
        vel = data["vel"]
        return vel.astype(np.float32)

    def get_real_acc(self, idx):
        info = self.info_list[idx]
        offset = f"{self.framedata_color_name[info[3]]}_{info[2]}.npz"
        save_path = os.path.join(savedir, info[0], offset)
        data = np.load(save_path)
        acc = data["acc"]
        return acc.astype(np.float32)

    def get_real_omega(self, idx):
        info = self.info_list[idx]
        offset = f"{self.framedata_color_name[info[3]]}_{info[2]}.npz"
        save_path = os.path.join(savedir, info[0], offset)
        data = np.load(save_path)
        omega = data["angvel"]
        return omega.astype(np.float32)

    def get_real_beta(self, idx):
        info = self.info_list[idx]
        offset = f"{self.framedata_color_name[info[3]]}_{info[2]}.npz"
        save_path = os.path.join(savedir, info[0], offset)
        data = np.load(save_path)
        beta = data["angacc"]
        return beta.astype(np.float32)
    
    def get_grasp_idx(self, idx):
        return 0

class OakInkImageSequence(OakInkImage):

    def __init__(self, seq_id, view_id, enable_handover=False) -> None:

        self.framedata_color_name = [
            "north_east_color",
            "south_east_color",
            "north_west_color",
            "south_west_color",
        ]
        view_name = self.framedata_color_name[view_id]
        self._name = f"OakInkImage_{seq_id}_{view_name}"

        assert "OAKINK_DIR" in os.environ, "environment variable 'OAKINK_DIR' is not set"
        self._data_dir = os.environ["OAKINK_DIR"]
        info_list_all = json.load(open(os.path.join(self._data_dir, "image", "anno", "seq_all.json")))

        seq_cat, seq_timestamp = seq_id.split("/")
        self.info_list = [info for info in info_list_all if (info[0] == seq_id and info[3] == view_id)]

        # deal with two hand cases.
        self.info_list.sort(key=lambda x: x[1] * 1000 + x[2])

        self.info_str_list = []
        for info in self.info_list:
            info_str = "__".join([str(x) for x in info])
            info_str = info_str.replace("/", "__")
            self.info_str_list.append(info_str)

        self.obj_id, self.intent_id, self.subject_id = decode_seq_cat(seq_cat)
        # load obj
        self.obj_mapping = {}
        suppress_trimesh_logging()
        obj_root = os.path.join(self._data_dir, "image", "obj")
        self.obj_model = load_object_by_id(self.obj_id, obj_root)
        self.obj_mapping[self.obj_id] = self.obj_model

        self._image_size = (848, 480)  # (W, H)
        self._hand_side = "right"

        self._enable_handover = enable_handover
        # seq status
        with open(os.path.join(self._data_dir, "image", "anno", "seq_status.json"), "r") as f:
            self.seq_status = json.load(f)

        # handover
        if self._enable_handover:
            self.handover_info, self.handover_sample_index_list = self._get_handover_info(self.info_list)
            self.handover_info_list = list(self.handover_info.keys())
        else:
            self.handover_info, self.handover_sample_index_list = None, None
            self.handover_info_list = None
