"""Example pipeline:
1) Load cameras.json, mouse template, and dlc_long.csv
2) Build an initialization from multi-view triangulation
3) Call the Ceres backend (mouse_ba.solve)

Note: You need to build/install the pybind11 module first.
"""

import numpy as np

from frontend import (
    load_cameras_json,
    load_template_points,
    load_dlc_long_csv,
    init_poses_from_triangulation,
    default_weight_pack,
)

import mouse_ba


def main():
    cam_quat, cam_trans, cam_intr = load_cameras_json("../cameras.json")
    template_pts = load_template_points("../mouse3_mesh_shortTail.txt")

    (obs_cam, obs_mouse, obs_joint, obs_frame,
     obs_x, obs_y, obs_l, num_frames) = load_dlc_long_csv("../dlc_long.csv")

    init_q, init_t = init_poses_from_triangulation(
        template_pts,
        obs_cam, obs_mouse, obs_joint, obs_frame, obs_x, obs_y, obs_l,
        cam_quat, cam_trans, cam_intr,
        likelihood_min=0.2,
    )

    wp = default_weight_pack().to_dict()

    solver_opts = {
        "max_num_iterations": 40,
        "num_threads": 8,
        "verbose": True,
        "optimize_cameras": False,
        "optimize_intrinsics": False,
    }

    out = mouse_ba.solve(
        cam_quat, cam_trans, cam_intr,
        template_pts,
        obs_cam, obs_mouse, obs_joint, obs_frame,
        obs_x, obs_y, obs_l,
        num_frames,
        init_q, init_t,
        wp,
        solver_opts,
    )

    print(out["summary"])
    # out contains mouse_quat, mouse_trans, slack, ctrl_p, ctrl_phi


if __name__ == "__main__":
    main()
