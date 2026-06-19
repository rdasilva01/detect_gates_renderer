#!/usr/bin/env python3
"""Render a gate-segmentation mask or pose/keypoint detections from Python.

Python port of examples/detect_gates.cpp: loads the same three YAML configs
and renders for a hardcoded drone pose. --mode segment (default) writes a
grayscale PNG; --mode pose writes a JSON array of per-gate keypoint/bbox
detections instead.
"""
import argparse
import json

import cv2

from detect_gates_renderer import DronePose, GateRenderer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gates-config", default="config/gates_config.yaml")
    parser.add_argument("--drone-config", default="config/config.yaml")
    parser.add_argument("--camera-config", default="config/camera_calibration.yaml")
    parser.add_argument("--output", required=True)
    parser.add_argument("--mode", choices=["segment", "pose"], default="segment")
    parser.add_argument("-r", "--rectified", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    renderer = GateRenderer(args.gates_config, args.drone_config, args.camera_config, args.rectified)
    pose = DronePose(x=19.0, y=2.0, z=0.155, roll=0.0, pitch=0.0, yaw=3.13)

    if args.mode == "segment":
        mask = renderer.render(pose)
        if not cv2.imwrite(args.output, mask):
            raise SystemExit(f"error: failed to write {args.output}")
    else:
        detections = renderer.render_detections(pose)
        payload = [
            {
                "gate": d.gate,
                "bounding_box": {
                    "x1": d.bounding_box.x1,
                    "y1": d.bounding_box.y1,
                    "x2": d.bounding_box.x2,
                    "y2": d.bounding_box.y2,
                },
                "keypoints": [{"name": k.name, "x": k.x, "y": k.y, "visible": k.visible} for k in d.keypoints],
            }
            for d in detections
        ]
        with open(args.output, "w") as f:
            json.dump(payload, f, indent=2)


if __name__ == "__main__":
    main()
