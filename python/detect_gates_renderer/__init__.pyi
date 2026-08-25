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
    @staticmethod
    def from_quaternion(x: float, y: float, z: float, qw: float, qx: float, qy: float,
                        qz: float) -> DronePose: ...

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
    bounding_box: BoundingBox
    keypoints: list[Keypoint]

class GateRenderer:
    def __init__(self, gates_config_path: str, drone_config_path: str, camera_config_path: str,
                 rectified: bool = False) -> None: ...
    @overload
    def render(self, pose: DronePose) -> npt.NDArray[np.uint8]: ...
    @overload
    def render(self, x: float, y: float, z: float, roll: float, pitch: float,
               yaw: float) -> npt.NDArray[np.uint8]: ...
    @overload
    def render_segmented(self, pose: DronePose) -> tuple[npt.NDArray[np.uint8], npt.NDArray[np.uint8]]: ...
    @overload
    def render_segmented(self, x: float, y: float, z: float, roll: float, pitch: float,
                         yaw: float) -> tuple[npt.NDArray[np.uint8], npt.NDArray[np.uint8]]: ...
    @property
    def gate_names(self) -> list[str]: ...
    @overload
    def render_detections(self, pose: DronePose, min_visible_corners: int = ...) -> list[GateDetection]: ...
    @overload
    def render_detections(self, x: float, y: float, z: float, roll: float, pitch: float, yaw: float,
                           min_visible_corners: int = ...) -> list[GateDetection]: ...
    @property
    def image_width(self) -> int: ...
    @property
    def image_height(self) -> int: ...
