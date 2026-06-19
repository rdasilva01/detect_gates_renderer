#!/usr/bin/env python3
"""Render a gate-segmentation mask from Python.

Python port of examples/render_segmentation.cpp: loads the same three YAML
configs, renders a mask for a hardcoded drone pose, and writes it as a
grayscale PNG.
"""
import argparse

import cv2

from segment_gate_renderer import DronePose, GateRenderer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gates-config", default="config/gates_config.yaml")
    parser.add_argument("--drone-config", default="config/config.yaml")
    parser.add_argument("--camera-config", default="config/camera_calibration.yaml")
    parser.add_argument("--output", required=True)
    parser.add_argument("-r", "--rectified", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    renderer = GateRenderer(args.gates_config, args.drone_config, args.camera_config, args.rectified)

    pose = DronePose(x=19.0, y=2.0, z=0.155, roll=0.0, pitch=0.0, yaw=3.13)
    mask = renderer.render(pose)

    if not cv2.imwrite(args.output, mask):
        raise SystemExit(f"error: failed to write {args.output}")


if __name__ == "__main__":
    main()
