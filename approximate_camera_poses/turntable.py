import numpy as np
import matplotlib.pyplot as plt

# Setup parameters

num_views = 32
turntable_radius = 0.50  # meters

lid_angle_deg = 109.0
lid_tilt_from_vertical = lid_angle_deg - 90.0  # ~19 deg

hinge_height = 0.015
hinge_to_camera_along_lid = 0.18
camera_extra_height = 0.0265

# camera height (approx geometry)
camera_height = (
    hinge_height
    + hinge_to_camera_along_lid * np.sin(np.deg2rad(lid_angle_deg))
    + camera_extra_height
)

# horizontal offset from hinge to camera projection
camera_forward_offset = hinge_to_camera_along_lid * np.cos(np.deg2rad(lid_angle_deg))
camera_radius = turntable_radius - camera_forward_offset

world_up = np.array([0.0, 0.0, 1.0])

camera_positions = []
camera_dirs = []

# Camera pose generation

for i in range(num_views):

    theta = -2.0 * np.pi * i / num_views

    cam_pos = np.array([
        camera_radius * np.cos(theta),
        camera_radius * np.sin(theta),
        camera_height
    ])

    # direction toward object center in XY plane
    to_origin_xy = np.array([-cam_pos[0], -cam_pos[1], 0.0])
    to_origin_xy = to_origin_xy / np.linalg.norm(to_origin_xy)

    # enforce physical downward tilt (~19 deg)
    pitch = np.deg2rad(lid_tilt_from_vertical)

    forward = (
        np.cos(pitch) * to_origin_xy +
        np.sin(pitch) * np.array([0.0, 0.0, -1.0])
    )

    forward = forward / np.linalg.norm(forward)

    camera_positions.append(cam_pos)
    camera_dirs.append(forward)

camera_positions = np.array(camera_positions)
camera_dirs = np.array(camera_dirs)

# Visualization

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

scale = 0.15  # arrow length

for i in range(num_views):

    p = camera_positions[i]
    d = camera_dirs[i]

    ax.quiver(
        p[0], p[1], p[2],
        d[0], d[1], d[2],
        length=scale,
        normalize=True
    )

    ax.text(p[0], p[1], p[2], str(i + 1), fontsize=8)

# origin (turntable center)
ax.scatter(0, 0, 0, color='red', s=60)

# 40 cm jug vertical axis
ax.quiver(
    0, 0, 0,
    0, 0, 0.40,
    color='red',
    linewidth=2
)

# labels
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")
ax.set_title("Camera Pose Visualization (Physically Consistent Model)")

plt.show()

import pickle
import numpy as np

poses_R = []
poses_t = []
poses_T = []

for i in range(num_views):

    theta = -2.0 * np.pi * i / num_views

    cam_pos = np.array([
        camera_radius * np.cos(theta),
        camera_radius * np.sin(theta),
        camera_height
    ])
    to_origin_xy = np.array([-cam_pos[0], -cam_pos[1], 0.0])
    to_origin_xy = to_origin_xy / np.linalg.norm(to_origin_xy)

    pitch = np.deg2rad(lid_tilt_from_vertical)

    forward = (
        np.cos(pitch) * to_origin_xy +
        np.sin(pitch) * np.array([0.0, 0.0, -1.0])
    )
    forward = forward / np.linalg.norm(forward)

    world_up = np.array([0.0, 0.0, 1.0])

    right = np.cross(world_up, forward)
    right = right / np.linalg.norm(right)

    up = np.cross(forward, right)

    R_wc = np.stack([right, up, forward], axis=1)  # camera in world

    R_cw = R_wc.T
    t_cw = -R_cw @ cam_pos

    T_cw = np.eye(4)
    T_cw[:3, :3] = R_cw
    T_cw[:3, 3] = t_cw

    poses_R.append(R_cw)
    poses_t.append(t_cw)
    poses_T.append(T_cw)

poses_R = np.array(poses_R)
poses_t = np.array(poses_t)
poses_T = np.array(poses_T)

poses_data = {
    "num_views": num_views,
    "turntable_radius": turntable_radius,
    "camera_height": camera_height,

    # voxel carving uses THIS:
    "R_world_to_cam": poses_R,
    "t_world_to_cam": poses_t,
    "T_world_to_cam": poses_T,

    # optional metadata
    "lid_tilt_deg": lid_tilt_from_vertical,
}

with open("camera_poses.pkl", "wb") as f:
    pickle.dump(poses_data, f)

print("Saved voxel-carving-ready poses to camera_poses.pkl")