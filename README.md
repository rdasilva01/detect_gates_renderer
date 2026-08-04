# detect_gates_renderer

A standalone C++ library for rendering gate-segmentation masks and pose/
keypoint detections: given a gate layout, a drone pose, and a fisheye camera
calibration, it can render either a binary segmentation mask of what the
camera would see (255 = gate frame, 0 = background), or per-gate keypoint
(4 inner + 4 outer corner) and bounding-box detections with cross-gate
occlusion handling.

The segmentation pipeline is a C++ port of the geometry/projection/rendering
pipeline from the `map_error_net` Python project (`transforms.py`,
`gates.py`, `projection.py`, `render.py`, `scene.py`), verified
pixel-for-pixel against the original Python implementation. The pose/
keypoint detection mode is a from-scratch C++ addition (not present in
`map_error_net`), porting the core geometry of a reference ROS2 gate-
projection node, adapted to this library's real 3D gate geometry (the
reference node assumes a flat, zero-thickness gate).

## Dependencies

- CMake >= 3.16
- A C++17 compiler
- OpenCV (`core`, `imgproc`, `calib3d`, `imgcodecs`) — fisheye projection and rasterization
- Eigen3 — linear algebra
- yaml-cpp — config file parsing
- (optional, for the Python bindings) Python >= 3.8, [nanobind](https://github.com/wjakob/nanobind), [scikit-build-core](https://github.com/scikit-build/scikit-build-core) — installed automatically by `pip install .`

## Build

```sh
cmake -B build
cmake --build build
```

This builds the `detect_gates_renderer` static library and (by default) the
`detect_gates` example. Set `-DDETECT_GATES_RENDERER_BUILD_EXAMPLES=OFF`
to skip the example when embedding this project via `add_subdirectory`.

## Library layout

| Module | Header | Responsibility |
| --- | --- | --- |
| `transforms` | `include/detect_gates/transforms.hpp` | RPY <-> rotation matrix, pose composition/inversion |
| `gates` | `include/detect_gates/gates.hpp` | Gate corner/face geometry, edge subdivision |
| `projection` | `include/detect_gates/projection.hpp` | Fisheye/pinhole projection, 3D cone clipping |
| `render` | `include/detect_gates/render.hpp` | Mask compositing (per-gate outer/inner face OR/AND) |
| `scene` | `include/detect_gates/scene.hpp` | YAML config loading + `renderPose()`/`detectGates()` orchestration |
| `GateRenderer` | `include/detect_gates/gate_renderer.hpp` | Loads all configs once, then renders masks/detections for many poses |

## Usage from other C++ code

`GateRenderer` is the recommended entry point: construct it once with the
three config file paths, then call `render()` (segmentation mask) or
`renderDetections()` (keypoint/bbox detections) per pose — both are always
available on the same instance.

```cpp
#include "detect_gates/gate_renderer.hpp"

detect_gates::GateRenderer renderer("config/gates_config.yaml",
                                     "config/config.yaml",
                                     "config/camera_calibration.yaml");

cv::Mat mask = renderer.render(detect_gates::DronePose{19.0, 2.0, 0.155, 0.0, 0.0, 3.13});
// or: renderer.render(x, y, z, roll, pitch, yaw);

std::vector<detect_gates::GateDetection> detections = renderer.renderDetections(
    detect_gates::DronePose{19.0, 2.0, 0.155, 0.0, 0.0, 3.13});
// Optional 2nd arg: minVisibleCorners (default 3) -- gates with fewer than
// this many visible keypoints (before or after cross-gate occlusion) are
// omitted. Each GateDetection has `gate` (source gate name), `boundingBox`
// (x1,y1,x2,y2 over visible keypoints only), 8 `keypoints`
// (4 *_inner + 4 *_outer, each with name/x/y/visible), plus `mask` (this
// gate's own silhouette) and `maskBoundingBox` (the box of that silhouette).
```

`boundingBox` and `maskBoundingBox` are not the same box, and the difference
matters:

- `boundingBox` spans the **visible keypoints**, so it is empty
  (`inf, inf, -inf, -inf`) for a gate that is fully occluded or has no corner in
  the frustum.
- `maskBoundingBox` is the box of the rendered silhouette. A fisheye bows a
  gate's straight edges **outside** the straight lines joining its corners --
  which is why the faces are subdivided before projection -- so a box built from
  the corners under-covers the real shape, by over 150 px on a close gate at
  820x616. Use this one for anything that has to contain the gate.

`mask` is the same silhouette `render()` would draw for this gate alone, before
other gates occlude it; OR-ing every detection's `mask` reproduces `render()`
exactly. It costs nothing extra: it is the footprint the cross-gate occlusion
test already builds.

Pass `rectified = true` as the 4th constructor argument to render as seen by
a rectified (pinhole) view of the fisheye camera instead of the raw fisheye
projection (applies to both `render()` and `renderDetections()`).

The lower-level free functions in `scene.hpp` (`loadGatesConfig`,
`loadCameraCalibration`, `renderPose`, `detectGates`, ...) are still
available directly if you need more control (e.g. reloading a gate layout
without re-reading the camera calibration).

## Example

```sh
./build/examples/detect_gates \
    --gates-config config/gates_config.yaml \
    --drone-config config/config.yaml \
    --camera-config config/camera_calibration.yaml \
    --mode segment \
    --output mask.png
```

Pass `--mode pose` to write a JSON array of per-gate keypoint/bbox
detections instead of a PNG mask (default mode is `segment`). Pass
`-r`/`--rectified` to render as seen by a rectified (pinhole) view of the
fisheye camera instead of the raw fisheye projection.

To visually cross-check that `--mode segment` and `--mode pose` agree for
the same pose (keypoints should sit on the mask's frame edges, bounding
boxes should hug each gate's silhouette), run:

```sh
./build/examples/visualize_detections --output overlay.png
```

This draws the detections (green = visible keypoint, red = occluded/out of
view, yellow = bounding box + gate name) directly on top of the segmentation
mask.

## Using as a dependency

After `cmake --install build`, downstream projects can:

```cmake
find_package(detect_gates_renderer REQUIRED)
target_link_libraries(my_target PRIVATE detect_gates_renderer::detect_gates_renderer)
```

## Python bindings

Python bindings for `GateRenderer` are built with
[nanobind](https://github.com/wjakob/nanobind) and packaged with
[scikit-build-core](https://github.com/scikit-build/scikit-build-core).

```sh
pip install .
```

```python
from detect_gates_renderer import DronePose, GateRenderer

renderer = GateRenderer("config/gates_config.yaml", "config/config.yaml", "config/camera_calibration.yaml")
pose = DronePose(x=19.0, y=2.0, z=0.155, roll=0.0, pitch=0.0, yaw=3.13)

mask = renderer.render(pose)
# `mask` is a (height, width) uint8 numpy array. `renderer.render(x, y, z, roll, pitch, yaw)` also works.

detections = renderer.render_detections(pose)  # optional 2nd arg: min_visible_corners (default 3)
for d in detections:
    print(d.gate, d.bounding_box, [(k.name, k.x, k.y, k.visible) for k in d.keypoints])
    # d.mask is this gate's own silhouette, a (height, width) uint8 array.
    # d.mask_bounding_box is its box -- see the C++ note above on why it is not
    # d.bounding_box.
    print(d.mask_bounding_box, d.mask.sum() > 0)
```

See `examples/python/detect_gates.py` for a runnable example (mirrors
`examples/detect_gates.cpp`):

```sh
python examples/python/detect_gates.py --mode segment --output mask.png
python examples/python/detect_gates.py --mode pose --output detections.json
```

And `examples/python/visualize_detections.py` (mirrors
`examples/visualize_detections.cpp`) to visually cross-check both modes
agree for the same pose:

```sh
python examples/python/visualize_detections.py --output overlay.png
```

`examples/python/live_view.py` is an interactive viewer (Python-only, no C++
equivalent): it renders continuously in a window (with an FPS counter) and
lets you fly around in the drone's body frame with the keyboard (arrows =
forward/back/left/right, w/s = up/down, a/d = yaw, q/e = roll, r/f = pitch,
Tab = toggle the pose-detection overlay on/off, Space = toggle rectified
vs. raw fisheye view, Esc = quit). Requires a
GUI-enabled OpenCV build (`opencv-python`, not `opencv-python-headless`).

```sh
python examples/python/live_view.py
```
