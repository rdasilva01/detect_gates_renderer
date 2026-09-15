#!/usr/bin/env python3
"""See the detections' 3D corners, and each gate's pose as PnP recovers it from them.

For one drone pose:

- left, the rendered mask with each detection's keypoints (green visible,
  red hidden);
- right, a rotatable 3D plot of the same corners at their `world` positions,
  the true camera (black), and, for every gate with at least --min-pnp-points
  visible corners, the gate's pose recovered by solvePnP from `gate_local`
  and the keypoint pixels: its axes -- x lateral (red), y up (green), z along
  the facing normal (blue) -- placed in the world through the true camera.
  Each detected gate's configured frame, the one `gate_local` is expressed in,
  is drawn faintly underneath, so a PnP error shows as the bold axes moving
  off it; the position and orientation errors are printed.

It also runs one global PnP over the whole circuit: every visible corner of
every detected gate, paired with its `world` point, solved at once for the
camera pose in the world, then turned into the drone pose by undoing the
calibration's camera_transform. The estimated drone (magenta star, with the
estimated camera's viewing direction) is drawn next to the true one (black
dot), and its position, roll/pitch/yaw and errors are printed.

--pixel-noise SIGMA simulates an imperfect detector: every keypoint's pixel is
moved by Gaussian noise with a standard deviation of SIGMA pixels on each axis
(at the renderer's output resolution), while the 3D points stay exact, as they
would for a known track. Each keypoint gets one noisy pixel, used by both the
per-gate PnPs and the global one, so their errors compare like for like; the
noisy pixels are drawn as yellow crosses. --seed repeats a draw.

PnP needs pinhole pixels. Unless -r, the pixels are fisheye, so they are first
undistorted with the calibration's model, its intrinsics scaled to the output
size, and corners more than 85 deg off-axis are left out of the solve. A
rectified renderer's pinhole matrix is rebuilt with
cv2.fisheye.estimateNewCameraMatrixForUndistortRectify, as the library does.
The script checks its camera model first: the `world` points must reproject
onto the keypoints.

A small, distant gate is a flat target a few pixels across, where very
different camera poses fit the pixels almost equally well; expect PnP to go
wrong there even though the points are exact.

Requires matplotlib and PyYAML.
"""
from __future__ import annotations

import argparse
import math
import os

import cv2
import numpy as np
import yaml

from detect_gates_renderer import DronePose, GateRenderer

_MAX_OFF_AXIS = math.radians(85.0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--gates-config", default="config/gates_config.yaml")
    parser.add_argument("--drone-config", default="config/config.yaml")
    parser.add_argument("--camera-config", default="config/camera_calibration.yaml")
    parser.add_argument("-r", "--rectified", action="store_true", help="use the rectified (pinhole) view")
    parser.add_argument("--x", type=float, default=19.0)
    parser.add_argument("--y", type=float, default=2.0)
    parser.add_argument("--z", type=float, default=0.155)
    parser.add_argument("--roll", type=float, default=0.0)
    parser.add_argument("--pitch", type=float, default=0.0)
    parser.add_argument("--yaw", type=float, default=3.13)
    parser.add_argument("--min-pnp-points", type=int, default=6,
                        help="visible corners a gate needs before PnP is run on it (default: 6)")
    parser.add_argument("--min-global-points", type=int, default=6,
                        help="visible corners, over all gates, the global PnP needs (default: 6)")
    parser.add_argument("--pixel-noise", type=float, default=0.0, metavar="SIGMA",
                        help="Gaussian noise added to each keypoint's x and y before PnP, standard deviation "
                             "in output pixels (default: 0, exact detections)")
    parser.add_argument("--seed", type=int, default=0, help="random seed for --pixel-noise (default: 0)")
    parser.add_argument("--output", help="save the figure here instead of opening a window")
    return parser.parse_args()


def rpy_to_matrix(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """Same convention as the library: Rz(yaw) @ Ry(pitch) @ Rx(roll)."""
    cr, sr, cp, sp, cy, sy = math.cos(roll), math.sin(roll), math.cos(pitch), math.sin(pitch), math.cos(yaw), math.sin(yaw)
    rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    return rz @ ry @ rx


def load_calibration(path: str) -> dict:
    root = yaml.safe_load(open(path))
    return root["/**"]["ros__parameters"] if "/**" in root else root


def camera_in_world(pose: DronePose, calib: dict) -> tuple[np.ndarray, np.ndarray]:
    """(R, t) of the camera in the world: the drone pose composed with the calibration's camera_transform."""
    tf = calib["camera_transform"]
    r_base, t_base = rpy_to_matrix(pose.roll, pose.pitch, pose.yaw), np.array([pose.x, pose.y, pose.z])
    r_cam, t_cam = rpy_to_matrix(tf["roll"], tf["pitch"], tf["yaw"]), np.array([tf["x"], tf["y"], tf["z"]])
    return r_base @ r_cam, t_base + r_base @ t_cam


def intrinsics(calib: dict, renderer: GateRenderer, rectified: bool) -> tuple[np.ndarray, np.ndarray, bool]:
    """Camera matrix, distortion and model of the pixels the renderer returns, at its output resolution."""
    width, height = calib["image_width"], calib["image_height"]
    k = np.array(calib["camera_matrix"]["data"], dtype=float).reshape(3, 3)
    d = np.array(calib["distortion_coefficients"]["data"], dtype=float).reshape(-1)[:4]
    fisheye = calib.get("distortion_model", "fisheye") in ("fisheye", "equidistant")
    if rectified:
        if fisheye:
            k = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(k, d, (width, height), np.eye(3), balance=0.0)
        d, fisheye = np.zeros(4), False
    scale = np.diag([renderer.image_width / width, renderer.image_height / height, 1.0])
    return scale @ k, d, fisheye


def project(points_cam: np.ndarray, k: np.ndarray, d: np.ndarray, fisheye: bool) -> np.ndarray:
    """Camera-frame points to pixels, with the library's models (full-range equidistant, or pinhole)."""
    x, y, z = points_cam.T
    if fisheye:
        r = np.hypot(x, y)
        theta = np.arctan2(r, z)
        t2 = theta * theta
        theta_d = theta * (1 + t2 * (d[0] + t2 * (d[1] + t2 * (d[2] + t2 * d[3]))))
        scale = np.where(r > 1e-12, theta_d / np.maximum(r, 1e-12), 0.0)
        u, v = scale * x, scale * y
    else:
        u, v = x / z, y / z
    return np.stack([k[0, 0] * u + k[0, 2], k[1, 1] * v + k[1, 2]], axis=1)


def normalized(pixels: np.ndarray, k: np.ndarray, d: np.ndarray, fisheye: bool) -> tuple[np.ndarray, np.ndarray]:
    """Pixels to the pinhole image plane (x/z, y/z), with each ray's angle off the optical axis."""
    mx, my = (pixels[:, 0] - k[0, 2]) / k[0, 0], (pixels[:, 1] - k[1, 2]) / k[1, 1]
    if not fisheye:
        return np.stack([mx, my], axis=1), np.arctan(np.hypot(mx, my))
    theta_d = np.hypot(mx, my)
    theta = theta_d.copy()
    for _ in range(20):  # invert theta_d(theta) by Newton
        t2 = theta * theta
        f = theta * (1 + t2 * (d[0] + t2 * (d[1] + t2 * (d[2] + t2 * d[3])))) - theta_d
        df = 1 + t2 * (3 * d[0] + t2 * (5 * d[1] + t2 * (7 * d[2] + t2 * 9 * d[3])))
        theta -= f / df
    scale = np.where(theta_d > 1e-12, np.tan(theta) / np.maximum(theta_d, 1e-12), 1.0)
    return np.stack([scale * mx, scale * my], axis=1), theta


def gate_frame(gate: dict) -> tuple[np.ndarray, np.ndarray]:
    """Origin and axes (rows: x lateral, y up, z normal) of the frame `gate_local` is expressed in."""
    x, y, z, yaw = gate["pose"]
    axes = np.array([[-math.sin(yaw), math.cos(yaw), 0.0], [0.0, 0.0, 1.0], [math.cos(yaw), math.sin(yaw), 0.0]])
    return np.array([x, y, z]), axes


def rings(count: int) -> list[list[int]]:
    """Keypoint index loops to draw: each square/octagon ring, or a double's two inner squares."""
    if count == 12:
        return [[0, 1, 2, 3], [6, 7, 8, 9]]
    half = count // 2
    return [list(range(half)), list(range(half, count))]


def matrix_to_rpy(r: np.ndarray) -> tuple[float, float, float]:
    """Inverse of rpy_to_matrix (away from pitch = +-90 deg)."""
    return (math.atan2(r[2, 1], r[2, 2]), math.asin(float(np.clip(-r[2, 0], -1.0, 1.0))), math.atan2(r[1, 0], r[0, 0]))


def drone_from_camera(r_wc: np.ndarray, t_wc: np.ndarray, calib: dict) -> tuple[np.ndarray, np.ndarray]:
    """The drone (base) pose whose camera sits at (r_wc, t_wc): camera_in_world with camera_transform undone."""
    tf = calib["camera_transform"]
    r_bc, t_bc = rpy_to_matrix(tf["roll"], tf["pitch"], tf["yaw"]), np.array([tf["x"], tf["y"], tf["z"]])
    r_wb = r_wc @ r_bc.T
    return r_wb, t_wc - r_wb @ t_bc


def noisy_pixels(detections, sigma: float, rng: np.random.Generator) -> dict[str, np.ndarray]:
    """Each detection's keypoint pixels, by gate, with Gaussian noise of `sigma` px per axis (0: exact)."""
    pixels = {}
    for det in detections:
        exact = np.array([(kp.x, kp.y) for kp in det.keypoints], dtype=float)
        pixels[det.gate] = exact + rng.normal(0.0, sigma, exact.shape) if sigma > 0 else exact
    return pixels


def gate_pnp(keypoints, pixels: np.ndarray, k: np.ndarray, d: np.ndarray, fisheye: bool, min_points: int):
    """One gate's pose in the camera frame from its visible corners' `gate_local` points and `pixels`.

    Returns (r_cg, t_cg, corners used); the pose is None when fewer than `min_points` corners are usable.
    """
    visible = [i for i, kp in enumerate(keypoints) if kp.visible]
    if not visible:
        return None, None, 0
    plane, off_axis = normalized(pixels[visible], k, d, fisheye)
    keep = off_axis < _MAX_OFF_AXIS
    obj = np.array([keypoints[i].gate_local for i in visible])[keep]
    if len(obj) < min_points:
        return None, None, len(obj)
    ok, rvec, tvec = cv2.solvePnP(obj, plane[keep], np.eye(3), None, flags=cv2.SOLVEPNP_SQPNP)
    if not ok:
        return None, None, len(obj)
    return cv2.Rodrigues(rvec)[0], tvec.reshape(3), len(obj)


def global_pnp(detections, pixels_by_gate: dict[str, np.ndarray], k: np.ndarray, d: np.ndarray, fisheye: bool,
               min_points: int):
    """Camera pose in the world from every visible corner of every detected gate, solved at once.

    `pixels_by_gate` holds each detection's keypoint pixels (exact, or from `noisy_pixels`).
    Returns (r_wc, t_wc, corners used, gates used), or None with too few usable corners.
    """
    world, pixels, owner = [], [], []
    for det in detections:
        for i, kp in enumerate(det.keypoints):
            if kp.visible:
                world.append(kp.world)
                pixels.append(pixels_by_gate[det.gate][i])
                owner.append(det.gate)
    if not world:
        return None
    plane, off_axis = normalized(np.array(pixels), k, d, fisheye)
    keep = off_axis < _MAX_OFF_AXIS
    if keep.sum() < min_points:
        return None
    ok, rvec, tvec = cv2.solvePnP(np.array(world)[keep], plane[keep], np.eye(3), None, flags=cv2.SOLVEPNP_SQPNP)
    if not ok:
        return None
    r_cw = cv2.Rodrigues(rvec)[0]
    return r_cw.T, -r_cw.T @ tvec.reshape(3), int(keep.sum()), len(set(np.array(owner)[keep]))


def main() -> None:
    args = parse_args()
    import matplotlib
    if args.output:
        matplotlib.use("Agg")
    else:
        # opencv-python bundles its own Qt and points QT_QPA_PLATFORM_PLUGIN_PATH
        # at it when cv2 is imported, so matplotlib's default Qt window would load
        # a second, clashing Qt build and abort. Tk avoids Qt altogether; without
        # it, clearing the variable is the next best thing.
        try:
            import tkinter  # noqa: F401
            matplotlib.use("TkAgg")
        except ImportError:
            os.environ.pop("QT_QPA_PLATFORM_PLUGIN_PATH", None)
    import matplotlib.pyplot as plt

    renderer = GateRenderer(args.gates_config, args.drone_config, args.camera_config, args.rectified)
    pose = DronePose(x=args.x, y=args.y, z=args.z, roll=args.roll, pitch=args.pitch, yaw=args.yaw)
    calib = load_calibration(args.camera_config)
    gates_cfg = yaml.safe_load(open(args.gates_config))["gates"]
    r_wc, t_wc = camera_in_world(pose, calib)
    k, d, fisheye = intrinsics(calib, renderer, args.rectified)
    detections = renderer.render_detections(pose)

    # Camera model check: the script's own projection of `world` must land on the keypoints.
    worst = 0.0
    for det in detections:
        for kp in det.keypoints:
            if kp.in_frustum:
                cam = (np.array(kp.world) - t_wc) @ r_wc
                worst = max(worst, float(np.hypot(*(project(cam[None], k, d, fisheye)[0] - (kp.x, kp.y)))))
    print(f"camera model check: world points reproject onto keypoints within {worst:.2e} px")
    pixels_by_gate = noisy_pixels(detections, args.pixel_noise, np.random.default_rng(args.seed))
    if args.pixel_noise > 0:
        print(f"pixel noise: sigma {args.pixel_noise:g} px per axis on every keypoint (seed {args.seed})")

    image = cv2.cvtColor(renderer.render(pose), cv2.COLOR_GRAY2RGB)
    fig = plt.figure(figsize=(15, 7))
    ax_img = fig.add_subplot(1, 2, 1)
    ax3d = fig.add_subplot(1, 2, 2, projection="3d")
    colours = plt.cm.tab10(np.arange(len(detections)) % 10)
    extent = [t_wc]

    print(f"{'gate':<10} {'visible':>7} {'PnP points':>10} {'PnP gate pose error':>24}")
    for det, colour in zip(detections, colours):
        keypoints = det.keypoints
        pixels = pixels_by_gate[det.gate]
        world = np.array([kp.world for kp in keypoints])
        visible = np.array([kp.visible for kp in keypoints])
        extent.append(world)

        # 2D: keypoints on the mask, and where the noise moved them
        for kp, (nx, ny) in zip(keypoints, pixels):
            if kp.in_frustum:
                ax_img.plot(kp.x, kp.y, "o", ms=3, color="lime" if kp.visible else "red")
                if args.pixel_noise > 0 and kp.visible:
                    ax_img.plot(nx, ny, "x", ms=4, mew=1, color="yellow")
        box = det.mask_bounding_box
        if math.isfinite(box.x1):
            ax_img.add_patch(plt.Rectangle((box.x1, box.y1), box.x2 - box.x1, box.y2 - box.y1, fill=False, ec=colour, lw=1))
            ax_img.text(box.x1, box.y1 - 3, det.gate, color=colour, fontsize=8)

        # 3D: corners, their rings, and the gate's own axes
        ax3d.scatter(*world[visible].T, color="green", s=12)
        ax3d.scatter(*world[~visible].T, color="red", s=12)
        for loop in rings(len(world)):
            pts = world[loop + loop[:1]]
            ax3d.plot(*pts.T, color=colour, lw=1)
        origin, axes = gate_frame(gates_cfg[det.gate])
        for axis, axis_colour in zip(axes, ("r", "g", "b")):  # configured frame, faint
            ax3d.quiver(*origin, *axis, length=1.0, color=axis_colour, lw=1, alpha=0.25)
        ax3d.text(*(origin + [0, 0, 0.3]), det.gate, color=colour, fontsize=8)

        # PnP on gate_local vs the (possibly noisy) keypoint pixels
        r_cg, t_cg, used = gate_pnp(keypoints, pixels, k, d, fisheye, args.min_pnp_points)
        if r_cg is not None:
            # PnP gives the gate in the camera frame; the true camera puts it in the world.
            r_wg = r_wc @ r_cg
            origin_pnp = r_wc @ t_cg + t_wc
            for axis, axis_colour in zip(r_wg.T, ("r", "g", "b")):
                ax3d.quiver(*origin_pnp, *axis, length=1.0, color=axis_colour, lw=2.5)
            extent.append(origin_pnp)
            position_error = float(np.linalg.norm(origin_pnp - origin))
            angle_error = math.degrees(math.acos(float(np.clip((np.trace(axes @ r_wg) - 1) / 2, -1, 1))))
            result = f"{100 * position_error:.2f} cm, {angle_error:.2f} deg"
        elif not visible.any():
            result = "not enough visible corners"
        else:
            result = f"{used} usable corners < {args.min_pnp_points}"
        print(f"{det.gate:<10} {int(visible.sum()):>7} {used:>10} {result:>24}")

    # global PnP: the whole circuit at once, for the drone pose
    true_drone = np.array([pose.x, pose.y, pose.z])
    ax3d.scatter(*true_drone, marker="o", s=40, color="black")
    solution = global_pnp(detections, pixels_by_gate, k, d, fisheye, args.min_global_points)
    if solution is None:
        print(f"global PnP: fewer than {args.min_global_points} usable corners")
    else:
        r_wc_pnp, t_wc_pnp, corners, gates_used = solution
        r_wb, drone = drone_from_camera(r_wc_pnp, t_wc_pnp, calib)
        rpy = matrix_to_rpy(r_wb)
        r_true = rpy_to_matrix(pose.roll, pose.pitch, pose.yaw)
        angle_error = math.degrees(math.acos(float(np.clip((np.trace(r_true.T @ r_wb) - 1) / 2, -1, 1))))
        print(f"global PnP on {corners} corners from {gates_used} gates:")
        print(f"   drone at ({drone[0]:.3f}, {drone[1]:.3f}, {drone[2]:.3f}) m, roll/pitch/yaw "
              f"({', '.join(f'{math.degrees(a):.2f}' for a in rpy)}) deg")
        print(f"   true     ({pose.x:.3f}, {pose.y:.3f}, {pose.z:.3f}) m, roll/pitch/yaw "
              f"({', '.join(f'{math.degrees(a):.2f}' for a in (pose.roll, pose.pitch, pose.yaw))}) deg")
        print(f"   error {100 * float(np.linalg.norm(drone - true_drone)):.2f} cm, {angle_error:.2f} deg")
        ax3d.scatter(*drone, marker="*", s=180, color="magenta")
        ax3d.quiver(*t_wc_pnp, *r_wc_pnp[:, 2], length=1.5, color="magenta", lw=2)
        extent.append(drone)

    # the true camera and where it looks
    ax3d.scatter(*t_wc, marker="^", s=80, color="black")
    ax3d.quiver(*t_wc, *r_wc[:, 2], length=1.5, color="black", lw=2)
    ax3d.text(*(t_wc + [0, 0, 0.3]), "camera", color="black", fontsize=8)

    pts = np.vstack([np.atleast_2d(e) for e in extent])
    centre, half = (pts.max(0) + pts.min(0)) / 2, max((pts.max(0) - pts.min(0)).max() / 2, 1.0)
    ax3d.set_xlim(centre[0] - half, centre[0] + half)
    ax3d.set_ylim(centre[1] - half, centre[1] + half)
    ax3d.set_zlim(max(0.0, centre[2] - half), centre[2] + half)
    ax3d.set_box_aspect((1, 1, 1))
    ax3d.set_xlabel("x (m, east)"); ax3d.set_ylabel("y (m, north)"); ax3d.set_zlabel("z (m, up)")
    ax3d.set_title("world corners (green visible, red hidden), true camera (black triangle) and drone (black dot),\n"
                   "gate poses from PnP (bold axes: x red, y green, z blue) over configured frames (faint),\n"
                   "drone from global PnP (magenta star, arrow = its camera's view)", fontsize=9)
    ax_img.imshow(image)
    noise_note = f", pixel noise sigma {args.pixel_noise:g} px (yellow x)" if args.pixel_noise > 0 else ""
    ax_img.set_title(f"{'rectified' if args.rectified else 'fisheye'} view, "
                     f"{renderer.image_width}x{renderer.image_height}{noise_note}", fontsize=9)
    ax_img.axis("off")
    fig.tight_layout()
    if args.output:
        fig.savefig(args.output, dpi=110)
    else:
        plt.show()


if __name__ == "__main__":
    main()
