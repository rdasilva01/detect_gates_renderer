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
  1                 : toggle the on-screen FPS readout (FPS | GEN_FPS)
  2                 : toggle full-resolution mask vs. the configured output size
  Space             : toggle rectified (pinhole) vs. raw fisheye view
  Esc               : quit

The mask is upscaled to --display-size (default 820x616) for viewing only, so
that a renderer configured for small masks -- config.yaml's output_width /
output_height, e.g. 64x64 -- still gives a window big enough to read the
overlay and text on. Nothing rendered or measured is affected; the mask really
is whatever `renderer.image_width` x `renderer.image_height` says, and the
status line reports it.

`2` switches to a mask rendered at the camera calibration's own resolution, to
see what the downscale throws away. It does nothing if config.yaml sets no
output size, since both are then the same. Note the full-resolution view is
not necessarily the slower one: with `native_inter: false` the downscaled mask
is rendered at full resolution and *then* resampled, so it costs strictly more.

Two rates are shown. FPS is the whole loop: render, overlay, imshow and the
GUI event pump. Because the window is resizable, imshow rescales the frame to
it every frame, so FPS drops as you enlarge the window even though nothing
about the mask changed. GEN_FPS times `renderer.render()` alone -- that is the
one to read when sizing a dataset run.

Requires a GUI-enabled OpenCV build (plain `opencv-python`, not
`opencv-python-headless`) and a display.
"""
from __future__ import annotations

import argparse
import os
import tempfile
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
                       pitch: float, yaw: float, scale_x: float = 1.0, scale_y: float = 1.0) -> None:
    """Draw detections onto `frame`.

    Detections come back in the renderer's *output* pixels, so when the frame
    has been upscaled for viewing they have to be scaled the same way or the
    whole overlay bunches up in the top-left corner.
    """
    for d in renderer.render_detections(x, y, z, roll, pitch, yaw):
        top_left = (int(d.bounding_box.x1 * scale_x), int(d.bounding_box.y1 * scale_y))
        bottom_right = (int(d.bounding_box.x2 * scale_x), int(d.bounding_box.y2 * scale_y))
        cv2.rectangle(frame, top_left, bottom_right, (0, 255, 255), 1)
        cv2.putText(frame, d.gate, (top_left[0], top_left[1] - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)

        for k in d.keypoints:
            if not k.in_frustum:
                continue
            color = (0, 255, 0) if k.visible else (0, 0, 255)
            center = (int(k.x * scale_x), int(k.y * scale_y))
            cv2.circle(frame, center, 3, color, cv2.FILLED)
            cv2.putText(frame, abbreviate(k.name), (center[0] + 4, center[1] + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.3,
                        color, 1)


def config_without_output_size(config_path: str) -> str | None:
    """Write a copy of `config_path` with the output size keys removed.

    Returns its path, or None if the config sets no output size (in which case
    it already renders at the calibration resolution). The caller may delete
    the copy as soon as the renderer is built -- it is only read once.
    """
    import yaml  # only needed for the full-resolution toggle; keep it off the import path

    with open(config_path) as handle:
        config = yaml.safe_load(handle)
    if not any(key in config for key in ("output_width", "output_height")):
        return None

    for key in ("output_width", "output_height"):
        config.pop(key, None)
    handle = tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False)
    with handle:
        yaml.safe_dump(config, handle)
    return handle.name


def build_renderers(args: argparse.Namespace) -> dict[tuple[bool, bool], GateRenderer]:
    """One renderer per (rectified, full_res) combination.

    Both the rectified flag and the output resolution are fixed at
    construction, so switching either at runtime means having built the
    alternative up front.
    """
    full_res_config = args.drone_config
    stripped = None
    try:
        stripped = config_without_output_size(args.drone_config)
    except ImportError:
        print("note: PyYAML not installed -- the full-resolution toggle (2) will do nothing")
    if stripped is not None:
        full_res_config = stripped

    renderers = {
        (rectified, full_res): GateRenderer(args.gates_config,
                                             full_res_config if full_res else args.drone_config,
                                             args.camera_config, rectified)
        for rectified in (False, True)
        for full_res in (False, True)
    }

    if stripped is not None:
        os.unlink(stripped)
    return renderers


def parse_size(text: str) -> tuple[int, int]:
    """"820x616" -> (820, 616)."""
    width, _, height = text.lower().partition("x")
    try:
        size = (int(width), int(height))
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"expected WIDTHxHEIGHT, got '{text}'") from exc
    if size[0] <= 0 or size[1] <= 0:
        raise argparse.ArgumentTypeError(f"display size must be positive, got '{text}'")
    return size


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
    parser.add_argument("--display-size", type=parse_size, default=(820, 616), metavar="WxH",
                         help="upscale the mask to this size for viewing only (default: 820x616). "
                              "Masks already this size or larger are shown untouched.")
    parser.add_argument("--debug-keys", action="store_true",
                         help="print every key code received, to help map unrecognized arrow keys")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    renderers = build_renderers(args)

    pos = np.array([args.x, args.y, args.z])
    roll, pitch, yaw = args.roll, args.pitch, args.yaw
    show_pose = True
    show_fps = True
    full_res = False
    rectified = args.rectified
    display_width, display_height = args.display_size

    window = "live_view"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    print(__doc__)

    ema_fps = 0.0
    ema_gen_fps = 0.0
    prev_time = time.perf_counter()

    while True:
        renderer = renderers[(rectified, full_res)]
        x, y, z = float(pos[0]), float(pos[1]), float(pos[2])
        render_start = time.perf_counter()
        mask = renderer.render(x, y, z, roll, pitch, yaw)
        render_dt = time.perf_counter() - render_start

        # Viewing only. Nearest-neighbour so the upscale shows the mask's real
        # pixel grid rather than inventing a smooth edge that isn't in the data
        # -- and so a soft mask (`inter_method: area`) keeps its exact values.
        scale_x = display_width / renderer.image_width
        scale_y = display_height / renderer.image_height
        view = mask
        if scale_x > 1.0 or scale_y > 1.0:
            view = cv2.resize(mask, (display_width, display_height), interpolation=cv2.INTER_NEAREST)
        else:
            scale_x = scale_y = 1.0

        frame = cv2.cvtColor(view, cv2.COLOR_GRAY2BGR) if show_pose else view

        now = time.perf_counter()
        dt = now - prev_time
        prev_time = now
        # FPS is the whole loop -- render, overlay, imshow, the GUI event pump --
        # so it moves when you resize the window. GEN_FPS is `render()` alone,
        # which is the number that says how fast masks can actually be produced.
        if dt > 0:
            ema_fps = 0.9 * ema_fps + 0.1 * (1.0 / dt) if ema_fps > 0 else 1.0 / dt
        if render_dt > 0:
            gen_fps = 1.0 / render_dt
            ema_gen_fps = 0.9 * ema_gen_fps + 0.1 * gen_fps if ema_gen_fps > 0 else gen_fps

        if show_pose:
            draw_pose_overlay(frame, renderer, x, y, z, roll, pitch, yaw, scale_x, scale_y)
        if show_fps:
            color = (0, 255, 255) if show_pose else (255, 255, 255)
            cv2.putText(frame, f"FPS: {ema_fps:.1f} | GEN_FPS: {ema_gen_fps:.1f}", (10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        cv2.imshow(window, frame)
        print(f"\rpos=({pos[0]:.2f}, {pos[1]:.2f}, {pos[2]:.2f})  rpy=({roll:.2f}, {pitch:.2f}, {yaw:.2f})  "
              f"fps={ema_fps:.1f}  gen_fps={ema_gen_fps:.1f}  rectified={rectified}  "
              f"mask={renderer.image_width}x{renderer.image_height}  ", end="", flush=True)

        key = cv2.waitKeyEx(1)
        if key == -1:
            continue
        if args.debug_keys:
            print(f"\nkey code: {key}")

        ascii_key = key & 0xFF
        if ascii_key == 9:  # Tab
            show_pose = not show_pose
            continue
        if ascii_key == ord("1"):
            show_fps = not show_fps
            continue
        if ascii_key == ord("2"):
            full_res = not full_res
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
