from abc import ABC, abstractmethod
from typing import List

import numpy as np
import torchvision.transforms.functional as tvF
from anakin.datasets.hoquery import (Queries, SynthQueries, match_collate_queries)
from anakin.utils import img_augment, transform
from anakin.utils.logger import logger
from anakin.utils.misc import CONST
from manotorch.manolayer import ManoLayer
from PIL import Image, ImageFilter
from torch.distributions.normal import Normal
from torch.distributions.uniform import Uniform
from torch.utils.data._utils.collate import default_collate


def ho_collate(batch):
    return hodata_collate(batch)


def hodata_collate(batch):
    """
    Collate function, duplicating the items in extend_queries along the
    first dimension so that they all have the same length.
    Typically applies to faces and vertices, which have different sizes
    depending on the object.
    """
    # *  NEW QUERY: CollateQueries.PADDING_MASK

    extend_queries = {
        Queries.OBJ_VERTS_3D,
        Queries.OBJ_VERTS_CAN,
        Queries.OBJ_VERTS_2D,
    }

    pop_queries = []
    for poppable_query in extend_queries:
        if poppable_query in batch[0]:
            pop_queries.append(poppable_query)

    # Remove fields that don't have matching sizes
    for pop_query in pop_queries:
        padding_query_field = match_collate_queries(pop_query)
        max_size = max([sample[pop_query].shape[0] for sample in batch])
        for sample in batch:
            pop_value = sample[pop_query]
            orig_len = pop_value.shape[0]
            # Repeat vertices so all have the same number
            pop_value = np.concatenate([pop_value] * int(max_size / pop_value.shape[0] + 1))[:max_size]
            sample[pop_query] = pop_value
            if padding_query_field not in sample:
                # !===== this is only done for verts / faces >>>>>
                # generate a new field, contains padding mask
                # note that only the beginning pop_value.shape[0] points are in effect
                # so the mask will be a vector of length max_size, with origin_len ones in the beginning
                padding_mask = np.zeros(max_size, dtype=np.int)
                padding_mask[:orig_len] = 1
                sample[padding_query_field] = padding_mask

    # store the mask filtering the points
    batch = default_collate(batch)
    return batch


class HOdata(ABC):

    def __init__(self, **cfg):
        """
        cfg contain keys:
        data_root: str = "data",
        data_split: str = "train",
        use_cache: bool = True,
        filter_no_contact: bool = False,
        filter_thresh: float = 10.0,
        scale_jittering: float = 0.0,
        center_jittering: float = 0.0,
        max_rot: float = 0.0 * np.pi,
        queries: Set[str] = None,
        """
        super().__init__()
        self.name = None
        self.data_root = cfg["DATA_ROOT"]
        self.data_split = cfg["DATA_SPLIT"]
        self.use_cache = cfg["DATA_PRESET"]["USE_CACHE"]
        self.filter_no_contact = cfg["DATA_PRESET"]["FILTER_NO_CONTACT"]
        self.filter_thresh = float(cfg["DATA_PRESET"]["FILTER_THRESH"])
        self.bbox_expand_ratio = float(cfg["DATA_PRESET"]["BBOX_EXPAND_RATIO"])

        self.crop_model = cfg["DATA_PRESET"].get("CROP_MODEL", "hand_obj")
        logger.warning(f"Use {self.crop_model} to crop image in {type(self).__name__}")

        self.require_full_image = cfg["DATA_PRESET"]["FULL_IMAGE"]
        if self.require_full_image:
            logger.warning(f"bbox_expand_ratio is set to be 1.0 in FULL_IMAGE mode")
            self.bbox_expand_ratio = 1.0
        self.aug = cfg["AUG"]

        self.raw_size = (256, 256)  # this will be overried by its subclass, original image size
        self.image_size = cfg["DATA_PRESET"]["IMAGE_SIZE"]  # (W, H)
        self.mini_factor_of_dataset = float(cfg.get("MINI_FACTOR", 1.0))
        self.center_idx = int(cfg["DATA_PRESET"].get("CENTER_IDX", 9))
        logger.info(f"{type(self).__name__} uses center_idx {self.center_idx}")

        if cfg["AUG"]:
            self.hue = 0.075
            self.saturation = 0.1
            self.contrast = 0.1
            self.brightness = 0.1
            self.blur_radius = 0.1
            if cfg["AUG_PARAM"] is not None:
                aug_param = cfg["AUG_PARAM"]
                self.scale_jittering = aug_param["SCALE_JIT"]
                self.center_jittering = aug_param["CENTER_JIT"]
                self.max_rot = aug_param["MAX_ROT"] * np.pi
        else:
            self.hue = 0.0
            self.saturation = 0.0
            self.brightness = 0.0
            self.contrast = 0.0
            self.blur_radius = 0.0
            self.scale_jittering = 0.0
            self.center_jittering = 0.0
            self.max_rot = 0.0

        self.sides = CONST.SIDE
        self.njoints = CONST.NUM_JOINTS
        self.ncorners = CONST.NUM_CORNERS

        self.mano_layer = ManoLayer(
            rot_mode="axisang",
            use_pca=False,
            mano_assets_root="assets/mano_v1_2",
            center_idx=None,
            flat_hand_mean=True,
        )

    @staticmethod
    def _flip_2d(raw_size, annot_2d):
        annot_2d = annot_2d.copy()
        annot_2d[:, 0] = raw_size[0] - annot_2d[:, 0]
        return annot_2d

    @staticmethod
    def _flip_3d(annot_3d):
        annot_3d = annot_3d.copy()
        annot_3d[:, 0] = -annot_3d[:, 0]
        return annot_3d

    @staticmethod
    def persp_project(points3d, cam_intr):
        hom_2d = np.array(cam_intr).dot(points3d.transpose()).transpose()
        points2d = (hom_2d / (hom_2d[:, 2:] + 1e-6))[:, :2]
        return points2d.astype(np.float32)

    @staticmethod
    def ortho_project(points3d, ortho_intr):
        points3d_xy1 = np.concatenate([points3d[:, :2], np.ones_like(points3d[:, 2:])], axis=1)  # (21, 3)
        points2d = ortho_intr.dot(points3d_xy1.T).T  # (21, 2)
        return points2d

    @staticmethod
    def get_annot_scale(annots, visibility=None, scale_factor=1.0):
        """
        Retreives the size of the square we want to crop by taking the
        maximum of vertical and horizontal span of the hand and multiplying
        it by the scale_factor to add some padding around the hand
        """
        if visibility is not None:
            annots = annots[visibility]
        min_x, min_y = annots.min(0)
        max_x, max_y = annots.max(0)
        delta_x = max_x - min_x
        delta_y = max_y - min_y
        max_delta = max(delta_x, delta_y)
        s = max_delta * scale_factor
        return s

    @staticmethod
    def get_annot_center(annots, visibility=None):
        if visibility is not None:
            annots = annots[visibility]
        min_x, min_y = annots.min(0)
        max_x, max_y = annots.max(0)
        c_x = int((max_x + min_x) / 2)
        c_y = int((max_y + min_y) / 2)
        return np.asarray([c_x, c_y])

    @staticmethod
    def fit_ortho_param(joints3d: np.ndarray, joints2d: np.ndarray) -> np.ndarray:
        joints3d_xy = joints3d[:, :2]  # (21, 2)
        joints3d_xy = joints3d_xy.reshape(-1)[:, np.newaxis]
        joints2d = joints2d.reshape(-1)[:, np.newaxis]
        pad2 = np.array(range(joints2d.shape[0]))
        pad2 = (pad2 % 2)[:, np.newaxis]
        pad1 = 1 - pad2
        jM = np.concatenate([joints3d_xy, pad1, pad2], axis=1)  # (42, 3)
        jMT = jM.transpose()  # (3, 42)
        jMTjM = np.matmul(jMT, jM)
        jMTb = np.matmul(jMT, joints2d)
        ortho_param = np.matmul(np.linalg.inv(jMTjM), jMTb)
        ortho_param = ortho_param.reshape(-1)
        return ortho_param  # [f, tx, ty]

    @abstractmethod
    def get_sample_idxs(self) -> List[int]:
        pass

    @abstractmethod
    def get_image(self, idx):
        pass

    @abstractmethod
    def get_image_path(self, idx):
        pass

    @abstractmethod
    def get_hand_verts_3d(self, idx):
        pass

    @abstractmethod
    def get_hand_verts_2d(self, idx):
        pass

    @abstractmethod
    def get_hand_faces(self, idx):
        pass

    @abstractmethod
    def get_joints_3d(self, idx):
        pass

    @abstractmethod
    def get_joints_2d(self, idx):
        pass

    # @abstractmethod
    def obj_load_driver(self):
        pass

    @abstractmethod
    def get_obj_idx(self, idx):
        pass

    @abstractmethod
    def get_obj_faces(self, idx):
        pass

    @abstractmethod
    def get_obj_transf(self, idx):
        pass

    @abstractmethod
    def get_obj_verts_transf(self, idx):
        pass

    @abstractmethod
    def get_obj_verts_2d(self, idx):
        pass

    @abstractmethod
    def get_obj_verts_can(self, idx):
        pass

    @abstractmethod
    def get_corners_3d(self, idx):
        pass

    @abstractmethod
    def get_corners_2d(self, idx):
        pass

    @abstractmethod
    def get_corners_can(self, idx):
        pass

    @abstractmethod
    def get_cam_intr(self, idx):
        pass

    @abstractmethod
    def get_sides(self, idx):
        pass

    @abstractmethod
    def get_center_scale_wrt_bbox(self, idx):
        pass

    @abstractmethod
    def get_sample_identifier(self, idx):
        pass

    def __len__(self):
        return len(self)

    def get_joints_vis(self, idx):
        if self.data_split not in ["train", "trainval"]:
            joints_vis = np.ones(self.njoints)
        else:
            joints_2d = self.get_joints_2d(idx)
            joints_vis = ((joints_2d[:, 0] >= 0) &
                          (joints_2d[:, 0] < self.raw_size[0])) & ((joints_2d[:, 1] >= 0) &
                                                                   (joints_2d[:, 1] < self.raw_size[1]))
        return joints_vis.astype(np.float32)

    def get_corners_vis(self, idx):
        if self.data_split not in ["train", "trainval"]:
            corners_vis = np.ones(self.ncorners)
        else:
            corners_2d = self.get_corners_2d(idx)
            corners_vis = ((corners_2d[:, 0] >= 0) &
                           (corners_2d[:, 0] < self.raw_size[0])) & ((corners_2d[:, 1] >= 0) &
                                                                     (corners_2d[:, 1] < self.raw_size[1]))
        return corners_vis.astype(np.float32)

    def __getitem__(self, idx):
        idx = self.get_sample_idxs()[idx]
        sample = {}
        sample[SynthQueries.IS_SYNTH] = False
        sample[SynthQueries.OBJ_ID] = -1
        sample[SynthQueries.PERSP_ID] = -1
        sample[SynthQueries.GRASP_ID] = -1
        sample[Queries.FRAME_NUM] = self.frame_num
        frame_list = [i - (self.frame_num - 1) // 2 for i in range(self.frame_num)]

        gt_hand_side = self.get_sides(idx)
        flip = True if gt_hand_side != self.sides else False

        bbox_center, bbox_scale = self.get_center_scale_wrt_bbox(idx)
        bbox_scale *= self.bbox_expand_ratio

        img = self.get_image(idx)
        img_list = []
        for i in frame_list:
            img_list.append(self.get_image(idx, seq = i))
        cam_intr = self.get_cam_intr(idx)
        joints_3d = self.get_joints_3d(idx)
        joints_3d_list = []
        for i in frame_list:
            joints_3d_list.append(self.get_joints_3d(idx, seq = i))
        joints_2d = self.get_joints_2d(idx)
        corners_3d = self.get_corners_3d(idx)
        corners_3d_list = []
        for i in frame_list:
            corners_3d_list.append(self.get_corners_3d(idx, seq = i))
        corners_2d = self.get_corners_2d(idx)
        corners_2d_list = []
        for i in frame_list:
            corners_2d_list.append(self.get_corners_2d(idx, seq = i))
        corners_can = self.get_corners_can(idx)

        # Flip 2d if needed
        if flip:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
            for i in range(len(img_list)):
                img_list[i] = img_list[i].transpose(Image.FLIP_LEFT_RIGHT)
            bbox_center[0] = self.raw_size[0] - bbox_center[0]  # image center
            joints_3d = self._flip_3d(joints_3d)
            for i in range(len(joints_3d_list)):
                joints_3d_list[i] = self._flip_3d(joints_3d_list[i])
            corners_3d = self._flip_3d(corners_3d)
            for i in range(len(corners_3d_list)):
                corners_3d_list[i] = self._flip_3d(corners_3d_list[i])
            joints_2d = self._flip_2d(self.raw_size, joints_2d)
            corners_2d = self._flip_2d(self.raw_size, corners_2d)
            for i in range(len(corners_2d_list)):
                corners_2d_list[i] = self._flip_2d(self.raw_size, corners_2d_list[i])

        # Data augmentation
        if self.aug:
            # * Randomly jitter center
            # Center is located in square of size 2*center_jitter_factor
            # in center of cropped image
            center_jit = Uniform(low=-1, high=1).sample((2,)).numpy()
            center_offsets = self.center_jittering * bbox_scale * center_jit
            bbox_center = bbox_center + center_offsets.astype(int)

            # Scale jittering
            jittering_ = Normal(0, self.scale_jittering / 3.0).sample().item()  # (-3, 3)
            jittering = jittering_ + 1.0
            jittering = np.clip(jittering, 1 - self.scale_jittering, 1 + self.scale_jittering)
            bbox_scale = bbox_scale * jittering
            rot_rad = Uniform(low=-self.max_rot, high=self.max_rot).sample().item()
        else:
            rot_rad = 0

        rot_mat = np.array([
            [np.cos(rot_rad), -np.sin(rot_rad), 0],
            [np.sin(rot_rad), np.cos(rot_rad), 0],
            [0, 0, 1],
        ]).astype(np.float32)

        oc = [cam_intr[0, 2], cam_intr[1, 2]]
        affine_transf, post_rot_transf = transform.get_affine_transform(
            center=bbox_center,
            scale=bbox_scale,
            optical_center=oc,  # (cx, cy)
            out_res=self.image_size,  # (H, W)
            rot=rot_rad,
        )

        cam_intr = post_rot_transf.dot(cam_intr)
        sample[Queries.CAM_INTR] = cam_intr.astype(np.float32)

        joints_3d = rot_mat.dot(joints_3d.transpose(1, 0)).transpose()
        root_joint = joints_3d[self.center_idx]
        sample[Queries.ROOT_JOINT] = root_joint
        corners_3d = rot_mat.dot(corners_3d.transpose(1, 0)).transpose()
        sample[Queries.JOINTS_3D] = joints_3d - root_joint  # * make it root relative
        joints_2d = transform.transform_coords(joints_2d, affine_transf).astype(np.float32)
        sample[Queries.JOINTS_2D] = joints_2d

        root_joint_list = []
        for i in range(len(joints_3d_list)):
            _joints_3d = rot_mat.dot(joints_3d_list[i].transpose(1, 0)).transpose()
            _root_joint = _joints_3d[self.center_idx]
            root_joint_list.append(_root_joint)
            corners_3d_list[i] = rot_mat.dot(corners_3d_list[i].transpose(1, 0)).transpose()
            joints_3d_list[i] = _joints_3d - _root_joint
        sample[Queries.JOINTS_3D_LIST] = np.array(joints_3d_list)
        sample[Queries.ROOT_JOINT_LIST] = np.array(root_joint_list)
        

        joints_vis = self.get_joints_vis(idx)
        if self.data_split not in ["train", "trainval"]:
            sample[Queries.JOINTS_VIS] = np.full(CONST.NUM_JOINTS, 1.0, dtype=np.float32)
        elif joints_vis.sum() < CONST.NUM_JOINTS * 0.4:  # magic number
            # hand invisible in raw image
            sample[Queries.JOINTS_VIS] = np.full(CONST.NUM_JOINTS, 0.0, dtype=np.float32)
        else:
            joints_vis_aug = (((joints_2d[:, 0] >= 0) & (joints_2d[:, 0] < self.image_size[0])) &
                              ((joints_2d[:, 1] >= 0) & (joints_2d[:, 1] < self.image_size[1]))).astype(np.float32)
            if joints_vis_aug.sum() < CONST.NUM_JOINTS * 0.4:  # magic number
                sample[Queries.JOINTS_VIS] = np.full(CONST.NUM_JOINTS, 0.0, dtype=np.float32)
            else:
                sample[Queries.JOINTS_VIS] = joints_vis_aug

        sample[Queries.CORNERS_3D] = corners_3d - root_joint  # * make it root relative
        for i in range(len(corners_3d_list)):
            corners_3d_list[i] = corners_3d_list[i] - root_joint_list[i]
        corners_2d = transform.transform_coords(corners_2d, affine_transf).astype(np.float32)
        for i in range(len(corners_2d_list)):
            corners_2d_list[i] = transform.transform_coords(corners_2d_list[i], affine_transf).astype(np.float32)
        sample[Queries.CORNERS_2D] = corners_2d
        sample[Queries.CORNERS_CAN] = corners_can
        sample[Queries.OBJ_IDX] = self.get_obj_idx(idx)
        sample[Queries.GRASP_IDX] = self.get_grasp_idx(idx)
        sample[Queries.CORNERS_3D_LIST] = np.array(corners_3d_list)



        # object transform: T,   O3D = T @ O_CAN
        base_trasnf = self.get_obj_transf(idx)
        sample[Queries.OBJ_TRANSF] = base_transf2obj_transf(base_trasnf, rot_mat)
        sample[Queries.OBJ_TRANSF_LIST] = []
        for i in frame_list:
            _obj_transf = self.get_obj_transf(idx, i)
            sample[Queries.OBJ_TRANSF_LIST].append(base_transf2obj_transf(_obj_transf, rot_mat))
        sample[Queries.OBJ_TRANSF_LIST] = np.array(sample[Queries.OBJ_TRANSF_LIST])
        


        corners_vis = self.get_corners_vis(idx)
        get_corners_vis(corners_vis, sample, self.data_split, corners_2d, self.image_size, Queries.CORNERS_VIS)
        corners_vis_list = []
        for i in frame_list:
            _corners_vis = self.get_corners_vis(idx, i)
            if self.data_split not in ["train", "trainval"]:
                _corners_vis = np.full(CONST.NUM_CORNERS, 1.0, dtype=np.float32)  # all 1
            elif _corners_vis.sum() < CONST.NUM_CORNERS * 0.4:  # magic number
                # corners invisible in raw image
                _corners_vis = np.full(CONST.NUM_CORNERS, 0.0, dtype=np.float32)  # all 0
            else:
                corners_vis_aug = (((corners_2d_list[i][:, 0] >= 0) & (corners_2d_list[i][:, 0] < self.image_size[0])) &
                                    ((corners_2d_list[i][:, 1] >= 0) & (corners_2d_list[i][:, 1] < self.image_size[1]))).astype(np.float32)
                if corners_vis_aug.sum() < CONST.NUM_CORNERS * 0.4:  # magic number
                    _corners_vis = np.full(CONST.NUM_CORNERS, 0.0, dtype=np.float32)
                else:
                    _corners_vis = corners_vis_aug
            corners_vis_list.append(_corners_vis)
        sample[Queries.CORNER_VIS_LIST] = np.array(corners_vis_list)

        if self.aug:
            blur_radius = Uniform(low=0, high=1).sample().item() * self.blur_radius
            img = img.filter(ImageFilter.GaussianBlur(blur_radius))
            for i in range(len(img_list)):
                img_list[i] = img_list[i].filter(ImageFilter.GaussianBlur(blur_radius))
            B, C, S, H = img_augment.get_color_params(brightness=self.brightness,
                                                      saturation=self.saturation,
                                                      hue=self.hue,
                                                      contrast=self.contrast)
            img = img_augment.apply_jitter(img, brightness=B, contrast=C, saturation=S, hue=H)
            for i in range(len(img_list)):
                img_list[i] = img_augment.apply_jitter(img_list[i], brightness=B, contrast=C, saturation=S, hue=H)

        img = img_augment.transform_img(img, affine_transf, self.image_size)
        img = img.crop((0, 0, self.image_size[0], self.image_size[1]))
        img = tvF.to_tensor(img).float()
        img = tvF.normalize(img, [0.5, 0.5, 0.5], [1, 1, 1])
        for i in range(len(img_list)):
            img_list[i] = img_augment.transform_img(img_list[i], affine_transf, self.image_size)
            img_list[i] = img_list[i].crop((0, 0, self.image_size[0], self.image_size[1]))
            img_list[i] = tvF.to_tensor(img_list[i]).float()
            img_list[i] = tvF.normalize(img_list[i], [0.5, 0.5, 0.5], [1, 1, 1])


        sample[Queries.IMAGE] = img
        sample[Queries.IMAGE_LIST] = img_list
        sample[Queries.SAMPLE_IDX] = idx

        

        sample[Queries.TARGET_VEL] = self.get_real_vel(idx)
        sample[Queries.TARGET_OMEGA] = self.get_real_omega(idx)
        sample[Queries.TARGET_ACC] = self.get_real_acc(idx)
        sample[Queries.TARGET_BETA] = self.get_real_beta(idx)

        if self.name == "DexYCB":
            sample[Queries.LABEL_PATH] = self.get_label_path(idx)
        
        if self.name == "OakInkImage":
            sample[Queries.INFO_STR] = self.get_info_str(idx)
        
            # sample[Queries.PRED_TRANS_LIST] = []
            # sample[Queries.PRED_ROT_LIST] = []
            # sample[Queries.PRED_VEL_LIST] = []
            # sample[Queries.PRED_OMEGA_LIST] = []
            # for i in [-1, 0, 1]:
            #     sample[Queries.PRED_TRANS_LIST].append(self.get_pred_trans(idx, i))
            #     sample[Queries.PRED_ROT_LIST].append(self.get_pred_rot(idx, i))
            #     sample[Queries.PRED_VEL_LIST].append(self.get_pred_vel(idx, i))
            #     sample[Queries.PRED_OMEGA_LIST].append(self.get_pred_omega(idx, i))
            # sample[Queries.PRED_TRANS_LIST] = np.array(sample[Queries.PRED_TRANS_LIST])
            # sample[Queries.PRED_ROT_LIST] = np.array(sample[Queries.PRED_ROT_LIST])
            # sample[Queries.PRED_VEL_LIST] = np.array(sample[Queries.PRED_VEL_LIST])
            # sample[Queries.PRED_OMEGA_LIST] = np.array(sample[Queries.PRED_OMEGA_LIST])
            
        sample[Queries.KIN_DATA_MEAN], sample[Queries.KIN_DATA_STD] = self.get_kin_mean_std(idx)
            
        if self.frame_num >= 3:
            sample[Queries.TARGET_NEXT_VEL] = self.get_real_vel(idx, 1)
            sample[Queries.TARGET_NEXT_OMEGA] = self.get_real_omega(idx, 1)
            sample[Queries.TARGET_NEXT_ACC] = self.get_real_acc(idx, 1)
            sample[Queries.TARGET_NEXT_BETA] = self.get_real_beta(idx, 1)
        else:
            sample[Queries.TARGET_NEXT_VEL] = np.zeros(3)
            sample[Queries.TARGET_NEXT_OMEGA] = np.zeros(3)
            sample[Queries.TARGET_NEXT_ACC] = np.zeros(3)
            sample[Queries.TARGET_NEXT_BETA] = np.zeros(3)
        if self.frame_num >= 5:
            sample[Queries.TARGET_NNEXT_VEL] = self.get_real_vel(idx, 2)
            sample[Queries.TARGET_NNEXT_OMEGA] = self.get_real_omega(idx, 2)
            sample[Queries.TARGET_NNEXT_ACC] = self.get_real_acc(idx, 2)
            sample[Queries.TARGET_NNEXT_BETA] = self.get_real_beta(idx, 2)
        else:
            sample[Queries.TARGET_NNEXT_VEL] = np.zeros(3)
            sample[Queries.TARGET_NNEXT_OMEGA] = np.zeros(3)
            sample[Queries.TARGET_NNEXT_ACC] = np.zeros(3)
            sample[Queries.TARGET_NNEXT_BETA] = np.zeros(3)

        return sample

def base_transf2obj_transf(base_transf, rot_mat):
    base_rot = base_transf[:3, :3]  # (3, 3)
    base_tsl = base_transf[:3, 3:]  # (3, 1)
    trans_rot = rot_mat @ base_rot  # (3, 3)
    trans_tsl = rot_mat.dot(base_tsl)  # (3, 1)
    trans_transf = np.concatenate([trans_rot, trans_tsl], axis=1)  # (3, 4)
    trans_transf = np.concatenate([trans_transf, np.array([0.0, 0.0, 0.0, 1.0])[np.newaxis, :]], axis=0)
    return trans_transf.astype(np.float32)

def get_corners_vis(corners_vis, sample, data_split, corners_2d, image_size, queries):
    if data_split not in ["train", "trainval"]:
        sample[queries] = np.full(CONST.NUM_CORNERS, 1.0, dtype=np.float32)  # all 1
    elif corners_vis.sum() < CONST.NUM_CORNERS * 0.4:  # magic number
        # corners invisible in raw image
        sample[queries] = np.full(CONST.NUM_CORNERS, 0.0, dtype=np.float32)  # all 0
    else:
        corners_vis_aug = (((corners_2d[:, 0] >= 0) & (corners_2d[:, 0] < image_size[0])) &
                            ((corners_2d[:, 1] >= 0) & (corners_2d[:, 1] < image_size[1]))).astype(np.float32)
        if corners_vis_aug.sum() < CONST.NUM_CORNERS * 0.4:  # magic number
            sample[queries] = np.full(CONST.NUM_CORNERS, 0.0, dtype=np.float32)
        else:
            sample[queries] = corners_vis_aug