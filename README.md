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
// (x1,y1,x2,y2 over visible keypoints only), and 8 `keypoints`
// (4 *_inner + 4 *_outer, each with name/x/y/visible).
```

Poses can be given as a quaternion instead of roll/pitch/yaw. Frames are
REP-103 throughout — world ENU (x east, y north, z up), body FLU (x forward,
y left, z up) — so this is only a change of parameterization:

```cpp
detect_gates::DronePose pose =
    detect_gates::poseFromQuaternion(x, y, z, qw, qx, qy, qz);
cv::Mat mask = renderer.render(pose);            // and renderDetections(pose)
```

```python
pose = DronePose.from_quaternion(x, y, z, qw, qx, qy, qz)
mask = renderer.render(pose)
```

The quaternion is **scalar first, `(w, x, y, z)`** — note that ROS's
`geometry_msgs/Quaternion` orders its *fields* `x, y, z, w`. It need not be
normalized. Two conventions worth stating, since both are easy to get backwards:
yaw is zero pointing **east** and increases counter-clockwise, and because the
body y-axis points *left*, positive pitch is **nose down** (the opposite of the
NED/FRD aerospace convention).

Pass `rectified = true` as the 4th constructor argument to render as seen by
a rectified (pinhole) view of the fisheye camera instead of the raw fisheye
projection (applies to both `render()` and `renderDetections()`).

The lower-level free functions in `scene.hpp` (`loadGatesConfig`,
`loadCameraCalibration`, `renderPose`, `detectGates`, ...) are still
available directly if you need more control (e.g. reloading a gate layout
without re-reading the camera calibration).

## Output resolution

By default a mask comes out at the camera calibration's `image_width` /
`image_height`. Four optional keys in `config.yaml` change that:

| key | meaning |
| --- | --- |
| `output_width`, `output_height` | mask size handed back. Set together, or leave both out for the calibration resolution. |
| `inter_method` | `nearest` \| `linear` \| `area` — how the mask is resampled down. Ignored when `native_inter` is true. |
| `native_inter` | rasterize straight at the output resolution instead of rendering large and resampling. |

Do **not** get this by editing `image_width`/`image_height` in the camera
calibration: intrinsics are tied to the resolution, so shrinking the image
without scaling `fx, fy, cx, cy` yields a small crop of the view rather than a
downscaled one (at 64×64 that is an empty mask on ~95% of poses). The keys
above scale the intrinsics for you, leaving the field of view untouched.

`renderDetections()` keypoints and bounding boxes are returned in
output-resolution pixels, so they always line up with `render()`'s mask.

Two things worth knowing when downscaling hard (e.g. 820×616 → 64×64):

- `area` (the default) makes the mask **soft**: each output pixel carries the
  fraction of itself covered by gate, so a frame thinner than one output pixel
  survives as a partial value instead of being hit or missed at random.
  Threshold it yourself if your loss needs hard labels. `nearest` stays binary
  but point-samples one pixel per block and breaks thin frames into dots
  (measured over 54 poses at 0.3–26 m: IoU 0.55 against true coverage, vs 0.71
  for `area`).
- `native_inter: true` is ~18× faster (0.25 vs 4.45 ms/frame at 64×64) and
  worth it for very large datasets, but a rasterizer only answers yes/no per
  pixel: sub-pixel frames get rounded up to a whole one (~+15% mask area at
  64×64) and a soft mask is not possible.

The aspect ratio is not preserved for you — 820×616 → 64×64 squashes
horizontally (12.8×) more than vertically (9.6×). That is fine provided the
real camera images are resized identically; if you letterbox or crop those, do
the same to the mask.

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
Tab = toggle the pose-detection overlay on/off, `1` = toggle the FPS readout,
`2` = toggle the full-resolution mask vs. the configured output size,
Space = toggle rectified vs. raw fisheye view, Esc = quit). Requires a
GUI-enabled OpenCV build (`opencv-python`, not `opencv-python-headless`).

```sh
python examples/python/live_view.py
```

Masks smaller than `--display-size` (default 820x616) are upscaled
nearest-neighbour for viewing, so a renderer configured for 64x64 output still
gives a readable window — the overlay and text are drawn at full size over the
mask's real pixel grid. This affects the preview only.
