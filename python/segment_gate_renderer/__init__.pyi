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

class GateRenderer:
    def __init__(self, gates_config_path: str, drone_config_path: str, camera_config_path: str,
                 rectified: bool = False) -> None: ...
    @overload
    def render(self, pose: DronePose) -> npt.NDArray[np.uint8]: ...
    @overload
    def render(self, x: float, y: float, z: float, roll: float, pitch: float,
               yaw: float) -> npt.NDArray[np.uint8]: ...
    @property
    def image_width(self) -> int: ...
    @property
    def image_height(self) -> int: ...
