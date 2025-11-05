import numpy as np
from typing import Dict, Tuple, List
from load_inputs import load_pair_pose

def invert_rel_transform(R_ab, t_ab):

    R_ba = R_ab.T
    t_ba = -R_ab.T @ t_ab
    return R_ba, t_ba

def camPose_to_extrinsic(R_c2w, t_c2w):

    R_w2c = R_c2w.T
    t_w2c = -R_c2w.T @ t_c2w
    return R_w2c, t_w2c

def accumulate_global_poses(pairs: List[tuple]):

    start_img = pairs[0][0]


    poses_cam2world = {
        start_img: (
            np.eye(3, dtype=np.float64),
            np.zeros((3,1), dtype=np.float64)
        )
    }

    rel_edges = {}
    for (a,b) in pairs:
        R_ab, t_ab = load_pair_pose(a,b)
        rel_edges[(a,b)] = (R_ab, t_ab)

        R_ba, t_ba = invert_rel_transform(R_ab, t_ab)
        rel_edges[(b,a)] = (R_ba, t_ba)

    changed = True
    while changed:
        changed = False
        for (a,b),(R_ab, t_ab) in rel_edges.items():
            if a in poses_cam2world and b not in poses_cam2world:

                R_a_c2w, t_a_c2w = poses_cam2world[a]

                R_b_c2w = R_a_c2w @ R_ab.T
                t_b_c2w = t_a_c2w - R_a_c2w @ (R_ab.T @ t_ab)

                poses_cam2world[b] = (R_b_c2w, t_b_c2w)
                changed = True


    poses_global = {}
    for img, (R_c2w, t_c2w) in poses_cam2world.items():
        R_w2c, t_w2c = camPose_to_extrinsic(R_c2w, t_c2w)
        poses_global[img] = (R_w2c, t_w2c)

    return poses_global