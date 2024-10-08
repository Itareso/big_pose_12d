from roma.mappings import rotmat_to_rotvec
from anakin.utils.transform import compute_rotation_matrix_from_ortho6d
import numpy as np
from scipy.interpolate import make_interp_spline
from scipy.signal import savgol_filter
from scipy.spatial.transform import Rotation as R

def compute_velocity_and_omega(corners3d1, corners3d2, ortho6d1, ortho6d2, fps):
    # Compute the center of the 3D bounding box for each frame
    center1 = corners3d1.mean(1)  # corners3d: [B, 8, 3] -> [B, 3]
    center2 = corners3d2.mean(1)

    # Compute the translation and rotation from frame1 to frame2
    rot1 = compute_rotation_matrix_from_ortho6d(ortho6d1)
    rot2 = compute_rotation_matrix_from_ortho6d(ortho6d2)
    rot = rot2 @ rot1.transpose(1, 2)

    # Compute the velocity
    velocity = (center2 - center1) * fps

    r = rotmat_to_rotvec(rot)
    omega = r * fps

    return velocity, omega

# compute position and quaternion from corner points and orthogonal 6d pose
def compute_pos_and_rot(corners3d, ortho6d):
    center = corners3d.mean(1)
    center = center.detach().cpu().numpy()
    rot = compute_rotation_matrix_from_ortho6d(ortho6d)
    rot = rot.detach().cpu().numpy()
    return center, rot

def spline_derivative(arr, fps=240., k_degree=3, bc_type=None):
    timestamps = np.linspace(0, len(arr)/fps, len(arr))
    spline = make_interp_spline(timestamps, arr, axis=0, k=k_degree, bc_type=bc_type)

    derivative = spline.derivative()

    return derivative(timestamps)

def angular_velocities(q1, q2, fps):
    return -(2 * fps) * np.array([
        q1[0]*q2[3] - q1[1]*q2[2] + q1[2]*q2[1] - q1[3]*q2[0],
        q1[0]*q2[2] + q1[1]*q2[3] - q1[2]*q2[0] - q1[3]*q2[1],
        -q1[0]*q2[1] + q1[1]*q2[0] + q1[2]*q2[3] - q1[3]*q2[2],
        ])

def angular_velocities_rot(r1, r2, fps):
    r21 = r2 @ r1.T
    rotvec21 = R.from_matrix(r21).as_rotvec()
    return rotvec21 * fps

def get_acc_beta_from_pose(predict_trans, predict_rot):
    vel = spline_derivative(predict_trans, fps=30)
    acc = spline_derivative(vel, fps=30)
    acc = acc[1:-1]

    omega = np.empty((len(predict_rot)-1, 3))
    for i in range(len(predict_rot)-1):
        omega[i] = angular_velocities_rot(predict_rot[i], predict_rot[i+1], fps=30)

    beta = np.diff(omega, axis=0) * 30

    return acc, beta

def get_acc_beta_from_vel(vel_list, omega_list):
    vel_list = np.array(vel_list)
    omega_list = np.array(omega_list)
    acc = spline_derivative(vel_list, fps=30)

    beta = np.diff(omega_list, axis=0) * 30
    beta = np.insert(beta, 0, beta[0], axis=0)
    #beta = spline_derivative(omega_list, fps=30)

    return acc, beta