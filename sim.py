import json
import math
import os
import sys
import time
from datetime import datetime
from random import randint, random

import numpy as np
import pybullet as p
import pybullet_data

from scipy.spatial.transform import Rotation as R

from matplotlib import pyplot as plt

from math import cos

def getMOI(axis, inertia_tensor):
    # Normalize the direction vector
    axis = np.array(axis)
    if np.linalg.norm(axis) == 0:
        return 0
    axis = axis / np.linalg.norm(axis)

    # Calculate the moment of inertia about the axis
    I = np.dot(axis, np.dot(inertia_tensor, axis))

    return I

def cosine_similarity(vec1, vec2):
    """Compute the cosine similarity between two vectors."""
    dot_product = np.dot(vec1, vec2)
    norm_vec1 = np.linalg.norm(vec1)
    norm_vec2 = np.linalg.norm(vec2)
    return dot_product / (norm_vec1 * norm_vec2)

sim_freq = 30

p.connect(p.DIRECT)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.setGravity(0, 0, 0)
p.setRealTimeSimulation(0)
p.setTimeStep(1./sim_freq)
# p.loadURDF("plane.urdf")

print(pybullet_data.getDataPath())



shift = [0, -0.02, 0]
scale = [1, 1, 1]


def eval_object_pos(obj_name, acc_list, beta_list, gt_trans, 
                    gt_rot, id_str, mode, dataset = "oakink"):
    
    if dataset == "oakink":
        dataset_path = "/mnt/public/datasets/OakInk"
        obj_path = "OakInkObjectsV2"
        obj_shape_path = os.path.join(obj_path, obj_name, "align")
        obj_files = os.listdir(obj_shape_path)
        try:
            obj_file = [f for f in obj_files if f.endswith(".obj")][0]
        except:
            obj_file = [f for f in obj_files if f.endswith(".ply")][0]
        visual_file, collision_file = obj_file, obj_file
    elif dataset == "dexycb":
        obj_path = "models"
        obj_shape_path = os.path.join(obj_path, obj_name)
        obj_files = os.listdir(obj_shape_path)
        visual_file = [f for f in obj_files if f.endswith(".obj") and "simple" not in f][0]
        collision_file = [f for f in obj_files if f.endswith(".obj") and "simple" in f][0]
    
    original_trans = gt_trans[0]
    original_rot = gt_rot[0]

    target_trans = gt_trans[-1]
    target_rot = gt_rot[-1]

    
    visual_path = os.path.join(obj_shape_path, visual_file)
    collision_path = os.path.join(obj_shape_path, collision_file)

    original_rot_quat = R.from_matrix(original_rot).as_quat()


    visual_shape = p.createVisualShape(
        shapeType=p.GEOM_MESH,
        fileName=visual_path,
        rgbaColor=[1, 1, 1, 1],
        specularColor=[0.4, .4, 0],
        visualFramePosition=shift,
        meshScale=scale
    )

    collision_shape = p.createCollisionShape(
        shapeType=p.GEOM_MESH,
        fileName=collision_path,
        collisionFramePosition=shift,
        meshScale=scale
    )

    obj = p.createMultiBody(
        baseMass=1,
        baseCollisionShapeIndex=collision_shape,
        baseVisualShapeIndex=visual_shape,
        basePosition=original_trans,
        baseOrientation=original_rot_quat,
        useMaximalCoordinates=True
    )

    inertia_tensor = p.getDynamicsInfo(obj, -1)[2]
    #print(inertia_tensor)
    mass = p.getDynamicsInfo(obj, -1)[0]

# obj = p.loadURDF("cube_small.urdf", [0, 0, 0.025])

    frame_length = len(acc_list)

    predict_trans = [original_trans]
    predict_rot = [original_rot]

    p.stepSimulation()

    for i in range(frame_length):
        acc = acc_list[i].tolist()
        beta = beta_list[i].tolist()
        # acc = [acc[0], acc[2], -acc[1]]
        # beta = [beta[0], beta[2], -beta[1]]
        trans, quat = p.getBasePositionAndOrientation(obj)
        predict_trans.append(trans)
        predict_rot.append(R.from_quat(quat).as_matrix())
        force = acc * 1
        #print("force:", force)
        mom_of_inertia = getMOI(beta, inertia_tensor)
        axis = beta / np.linalg.norm(beta) if np.linalg.norm(beta) != 0 else np.array([0, 0, 0])
        torque = mom_of_inertia * axis
        #print("torque:", torque)
        p.applyExternalForce(obj, -1, force, trans, p.WORLD_FRAME)
        p.applyExternalTorque(obj, -1, torque, p.WORLD_FRAME)
        real_vel = p.getBaseVelocity(obj)
        #print(real_vel)
        p.stepSimulation()
    
    final_trans, final_quat = p.getBasePositionAndOrientation(obj)
    final_rot = R.from_quat(final_quat).as_matrix()
    predict_trans.append(final_trans)
    predict_rot.append(final_rot)

    #print(final_trans, target_trans)
    #print(final_rot, target_rot)

    predict_trans = np.array(predict_trans)

    cos_sim = []
    #print(len(gt_rot))
    #print(len(predict_rot))
    for i in range(len(gt_rot)):
        trans_rot = np.dot(gt_rot[i], predict_rot[i].T)
        rotvec = R.from_matrix(trans_rot).as_rotvec()
        cos_sim.append(cos(np.linalg.norm(rotvec)))

    mag1 = []
    mag2 = []
    for i in range(len(gt_rot)):
        rotvec1 = R.from_matrix(gt_rot[i]).as_rotvec()
        rotvec2 = R.from_matrix(predict_rot[i]).as_rotvec()
        mag1.append(np.linalg.norm(rotvec1))
        mag2.append(np.linalg.norm(rotvec2))
    
    rot_loss = 0
    #trans_loss = np.mean((target_trans - final_trans) ** 2)
    trans_loss = np.mean(np.linalg.norm(predict_trans - gt_trans, axis=1))
    for i in range(len(cos_sim)):
        rot_loss += 1 - cos_sim[i]
    rot_loss /= len(cos_sim)
    #print(rot_loss)

    # _axis = ["x", "y", "z"]

    # for i, axis in enumerate(_axis):
    #     plt.plot(gt_trans[:, i], label=f"gt_pos_{axis}")
    #     plt.plot(predict_trans[:, i], label=f"predict_pos_{axis}")
    #     plt.legend()
    #     plt.savefig(f"sim_images/{id_str}_{mode}_pos_{axis}.png")
    #     plt.cla()
    
    # plt.plot(cos_sim, label="cos_sim")
    # plt.legend()
    # plt.savefig(f"sim_images/{id_str}_{mode}_cos_sim.png")
    # plt.cla()

    # plt.plot(mag1, label="gt_mag")
    # plt.plot(mag2, label="predict_mag")
    # plt.legend()
    # plt.savefig(f"sim_images/{id_str}_{mode}_mag.png")
    # plt.cla()
    print(trans_loss)

    return trans_loss, rot_loss
