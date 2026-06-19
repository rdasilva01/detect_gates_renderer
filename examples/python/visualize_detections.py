#!/usr/bin/env python3
"""Overlay pose-mode detections on top of the segment-mode mask from Python.

Python port of examples/visualize_detections.cpp: visual cross-check that
render() and render_detections() agree for the same pose -- keypoints should
sit right on the mask's frame edges, and bounding boxes should hug each
gate's silhouette.
"""
import argparse

import cv2

from detect_gates_renderer import DronePose, GateRenderer


def abbreviate(name: str) -> str:
    """"top_left_inner" -> "tl_i": initials of the side, plus the ring's first letter."""
    side, _, ring = name.rpartition("_")
    abbrev = "".join(word[0] for word in side.split("_"))
    return f"{abbrev}_{ring[0]}"


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
    overlay = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)

    detections = renderer.render_detections(pose)
    for d in detections:
        top_left = (int(d.bounding_box.x1), int(d.bounding_box.y1))
        bottom_right = (int(d.bounding_box.x2), int(d.bounding_box.y2))
        cv2.rectangle(overlay, top_left, bottom_right, (0, 255, 255), 1)
        cv2.putText(overlay, d.gate, (top_left[0], top_left[1] - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)

        for k in d.keypoints:
            if not k.in_frustum:
                # Outside the camera's field of view: (x, y) is not a
                # meaningful pixel location, so there's nothing to draw.
                continue
            color = (0, 255, 0) if k.visible else (0, 0, 255)
            center = (int(k.x), int(k.y))
            cv2.circle(overlay, center, 3, color, cv2.FILLED)
            cv2.putText(overlay, abbreviate(k.name), (center[0] + 4, center[1] + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.3,
                        color, 1)

    if not cv2.imwrite(args.output, overlay):
        raise SystemExit(f"error: failed to write {args.output}")


if __name__ == "__main__":
    main()
