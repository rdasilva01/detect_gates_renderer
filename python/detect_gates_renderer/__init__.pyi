from typing import overload

import numpy as np
import numpy.typing as npt

class DronePose:
    x: float
    y: float
    z: float
    roll: float
    pitch: float
    yaw: float

    def __init__(self, x: float = ..., y: float = ..., z: float = ..., roll: float = ..., pitch: float = ...,
                 yaw: float = ...) -> None: ...

class Keypoint:
    name: str
    x: float
    y: float
    visible: bool
    in_frustum: bool

class BoundingBox:
    x1: float
    y1: float
    x2: float
    y2: float

class GateDetection:
    gate: str
    # Spans the visible keypoints only: empty (inf, inf, -inf, -inf) for a gate
    # that is fully occluded or has no corner in the frustum.
    bounding_box: BoundingBox
    keypoints: list[Keypoint]
    # This gate's own silhouette, before other gates occlude it. Always the full
    # image size; all zeros when the gate projects nowhere.
    @property
    def mask(self) -> npt.NDArray[np.uint8]: ...
    # Bounding box of `mask`, following the fisheye-curved silhouette. Unlike
    # `bounding_box` it stays correct when the edges bow outside the corners and
    # when no corner is in the frustum.
    mask_bounding_box: BoundingBox

class GateRenderer:
    def __init__(self, gates_config_path: str, drone_config_path: str, camera_config_path: str,
                 rectified: bool = False) -> None: ...
    @overload
    def render(self, pose: DronePose) -> npt.NDArray[np.uint8]: ...
    @overload
    def render(self, x: float, y: float, z: float, roll: float, pitch: float,
               yaw: float) -> npt.NDArray[np.uint8]: ...
    @overload
    def render_detections(self, pose: DronePose, min_visible_corners: int = ...) -> list[GateDetection]: ...
    @overload
    def render_detections(self, x: float, y: float, z: float, roll: float, pitch: float, yaw: float,
                           min_visible_corners: int = ...) -> list[GateDetection]: ...
    @property
    def image_width(self) -> int: ...
    @property
    def image_height(self) -> int: ...
