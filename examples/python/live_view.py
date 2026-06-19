#!/usr/bin/env python3
"""Live interactive viewer: fly around in body frame with the keyboard.

Starts at the same demo pose as the other examples and continuously renders
the segmentation mask as fast as possible. Controls (all moves/rotations are
applied in the drone's current body frame):

  Up/Down arrows    : move forward/backward 1 m
  Left/Right arrows : move left/right 1 m
  w/s               : move up/down 1 m
  a/d               : yaw +/-15 deg
  q/e               : roll -/+15 deg
  r/f               : pitch -/+15 deg
  Tab               : toggle pose-detection overlay
  Space             : toggle rectified (pinhole) vs. raw fisheye view
  Esc               : quit

Requires a GUI-enabled OpenCV build (plain `opencv-python`, not
`opencv-python-headless`) and a display.
"""
from __future__ import annotations

import argparse
import time

import cv2
import numpy as np

from detect_gates_renderer import GateRenderer

_STEP_TRANSLATION = 1.0  # meters
_STEP_ROTATION = np.radians(15.0)

# Candidate extended key codes for arrow keys across common OpenCV GUI
# backends/platforms (X11/GTK, Windows, and a GTK3 build seen in some
# environments that offsets GDK keyvals by 0x100000). cv2.waitKeyEx's codes
# for non-printable keys aren't standardized across builds -- if arrows
# don't move anything, run with --debug-keys to print the codes your build
# actually sends and add them to the matching set below.
_LEFT = {65361, 2424832, 1113937}
_UP = {65362, 2490368, 1113938}
_RIGHT = {65363, 2555904, 1113939}
_DOWN = {65364, 2621440, 1113940}


def rpy_to_matrix(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """R = Rz(yaw) @ Ry(pitch) @ Rx(roll), matching the library's convention."""
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)
    rx = np.array([[1.0, 0.0, 0.0], [0.0, cr, -sr], [0.0, sr, cr]])
    ry = np.array([[cp, 0.0, sp], [0.0, 1.0, 0.0], [-sp, 0.0, cp]])
    rz = np.array([[cy, -sy, 0.0], [sy, cy, 0.0], [0.0, 0.0, 1.0]])
    return rz @ ry @ rx


def matrix_to_rpy(r: np.ndarray) -> tuple[float, float, float]:
    """Inverse of rpy_to_matrix."""
    roll = float(np.arctan2(r[2, 1], r[2, 2]))
    pitch = float(np.arctan2(-r[2, 0], np.hypot(r[2, 1], r[2, 2])))
    yaw = float(np.arctan2(r[1, 0], r[0, 0]))
    return roll, pitch, yaw


def abbreviate(name: str) -> str:
    """"top_left_inner" -> "tl_i": initials of the side, plus the ring's first letter."""
    side, _, ring = name.rpartition("_")
    abbrev = "".join(word[0] for word in side.split("_"))
    return f"{abbrev}_{ring[0]}"


def draw_pose_overlay(frame: np.ndarray, renderer: GateRenderer, x: float, y: float, z: float, roll: float,
                       pitch: float, yaw: float) -> None:
    for d in renderer.render_detections(x, y, z, roll, pitch, yaw):
        top_left = (int(d.bounding_box.x1), int(d.bounding_box.y1))
        bottom_right = (int(d.bounding_box.x2), int(d.bounding_box.y2))
        cv2.rectangle(frame, top_left, bottom_right, (0, 255, 255), 1)
        cv2.putText(frame, d.gate, (top_left[0], top_left[1] - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)

        for k in d.keypoints:
            if not k.in_frustum:
                continue
            color = (0, 255, 0) if k.visible else (0, 0, 255)
            center = (int(k.x), int(k.y))
            cv2.circle(frame, center, 3, color, cv2.FILLED)
            cv2.putText(frame, abbreviate(k.name), (center[0] + 4, center[1] + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.3,
                        color, 1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--gates-config", default="config/gates_config.yaml")
    parser.add_argument("--drone-config", default="config/config.yaml")
    parser.add_argument("--camera-config", default="config/camera_calibration.yaml")
    parser.add_argument("-r", "--rectified", action="store_true", help="start in rectified view (toggle with Space)")
    parser.add_argument("--x", type=float, default=19.0)
    parser.add_argument("--y", type=float, default=2.0)
    parser.add_argument("--z", type=float, default=0.155)
    parser.add_argument("--roll", type=float, default=0.0)
    parser.add_argument("--pitch", type=float, default=0.0)
    parser.add_argument("--yaw", type=float, default=3.13)
    parser.add_argument("--debug-keys", action="store_true",
                         help="print every key code received, to help map unrecognized arrow keys")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    # GateRenderer's `rectified` flag is constructor-only, so build both up
    # front and just switch which one we call each frame.
    renderers = {
        False: GateRenderer(args.gates_config, args.drone_config, args.camera_config, False),
        True: GateRenderer(args.gates_config, args.drone_config, args.camera_config, True),
    }

    pos = np.array([args.x, args.y, args.z])
    roll, pitch, yaw = args.roll, args.pitch, args.yaw
    show_pose = True
    rectified = args.rectified

    window = "live_view"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    print(__doc__)

    ema_fps = 0.0
    prev_time = time.perf_counter()

    while True:
        renderer = renderers[rectified]
        x, y, z = float(pos[0]), float(pos[1]), float(pos[2])
        mask = renderer.render(x, y, z, roll, pitch, yaw)
        frame = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR) if show_pose else mask

        now = time.perf_counter()
        dt = now - prev_time
        prev_time = now
        if dt > 0:
            ema_fps = 0.9 * ema_fps + 0.1 * (1.0 / dt) if ema_fps > 0 else 1.0 / dt

        if show_pose:
            draw_pose_overlay(frame, renderer, x, y, z, roll, pitch, yaw)
            cv2.putText(frame, f"FPS: {ema_fps:.1f}", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        else:
            cv2.putText(frame, f"FPS: {ema_fps:.1f}", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        cv2.imshow(window, frame)
        print(f"\rpos=({pos[0]:.2f}, {pos[1]:.2f}, {pos[2]:.2f})  rpy=({roll:.2f}, {pitch:.2f}, {yaw:.2f})  "
              f"fps={ema_fps:.1f}  rectified={rectified}  ", end="", flush=True)

        key = cv2.waitKeyEx(1)
        if key == -1:
            continue
        if args.debug_keys:
            print(f"\nkey code: {key}")

        ascii_key = key & 0xFF
        if ascii_key == 9:  # Tab
            show_pose = not show_pose
            continue
        if ascii_key == ord(" "):
            rectified = not rectified
            continue

        r_old = rpy_to_matrix(roll, pitch, yaw)

        delta_body = np.zeros(3)
        if ascii_key == ord("w"):
            delta_body[2] += _STEP_TRANSLATION
        elif ascii_key == ord("s"):
            delta_body[2] -= _STEP_TRANSLATION
        elif key in _UP:
            delta_body[0] += _STEP_TRANSLATION
        elif key in _DOWN:
            delta_body[0] -= _STEP_TRANSLATION
        elif key in _LEFT:
            delta_body[1] += _STEP_TRANSLATION
        elif key in _RIGHT:
            delta_body[1] -= _STEP_TRANSLATION
        if np.any(delta_body):
            pos = pos + r_old @ delta_body

        r_delta = None
        if ascii_key == ord("a"):
            r_delta = rpy_to_matrix(0.0, 0.0, _STEP_ROTATION)
        elif ascii_key == ord("d"):
            r_delta = rpy_to_matrix(0.0, 0.0, -_STEP_ROTATION)
        elif ascii_key == ord("q"):
            r_delta = rpy_to_matrix(-_STEP_ROTATION, 0.0, 0.0)
        elif ascii_key == ord("e"):
            r_delta = rpy_to_matrix(_STEP_ROTATION, 0.0, 0.0)
        elif ascii_key == ord("r"):
            r_delta = rpy_to_matrix(0.0, -_STEP_ROTATION, 0.0)
        elif ascii_key == ord("f"):
            r_delta = rpy_to_matrix(0.0, _STEP_ROTATION, 0.0)
        if r_delta is not None:
            roll, pitch, yaw = matrix_to_rpy(r_old @ r_delta)

        if ascii_key == 27:  # Esc
            break

    cv2.destroyAllWindows()
    cv2.waitKey(1)
    print()


if __name__ == "__main__":
    main()
