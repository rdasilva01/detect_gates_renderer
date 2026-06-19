# segment_gate

A standalone C++ library for rendering gate-segmentation masks: given a gate
layout, a drone pose, and a fisheye camera calibration, it renders a binary
mask of what the camera would see (255 = gate frame, 0 = background).

This is a C++ port of the geometry/projection/rendering pipeline from the
`map_error_net` Python project (`transforms.py`, `gates.py`,
`projection.py`, `render.py`, `scene.py`). It does not include any of the
training/ML/dataset-generation code from that project — just the
deterministic mask-rendering pipeline, verified pixel-for-pixel against the
original Python implementation.

## Dependencies

- CMake >= 3.16
- A C++17 compiler
- OpenCV (`core`, `imgproc`, `calib3d`, `imgcodecs`) — fisheye projection and rasterization
- Eigen3 — linear algebra
- yaml-cpp — config file parsing

## Build

```sh
cmake -B build
cmake --build build
```

This builds the `segment_gate` static library and (by default) the
`render_segmentation` example. Set `-DSEGMENT_GATE_BUILD_EXAMPLES=OFF` to
skip the example when embedding this project via `add_subdirectory`.

## Library layout

| Module | Header | Responsibility |
| --- | --- | --- |
| `transforms` | `include/segment_gate/transforms.hpp` | RPY <-> rotation matrix, pose composition/inversion |
| `gates` | `include/segment_gate/gates.hpp` | Gate corner/face geometry, edge subdivision |
| `projection` | `include/segment_gate/projection.hpp` | Fisheye/pinhole projection, 3D cone clipping |
| `render` | `include/segment_gate/render.hpp` | Mask compositing (per-gate outer/inner face OR/AND) |
| `scene` | `include/segment_gate/scene.hpp` | YAML config loading + `renderPose()` orchestration |
| `GateRenderer` | `include/segment_gate/gate_renderer.hpp` | Loads all configs once, then renders masks for many poses |

## Usage from other C++ code

`GateRenderer` is the recommended entry point: construct it once with the
three config file paths, then call `render()` per pose.

```cpp
#include "segment_gate/gate_renderer.hpp"

segment_gate::GateRenderer renderer("config/gates_config.yaml",
                                     "config/config.yaml",
                                     "config/camera_calibration.yaml");

cv::Mat mask = renderer.render(segment_gate::DronePose{19.0, 2.0, 0.155, 0.0, 0.0, 3.13});
// or: renderer.render(x, y, z, roll, pitch, yaw);
```

Pass `rectified = true` as the 4th constructor argument to render as seen by
a rectified (pinhole) view of the fisheye camera instead of the raw fisheye
projection.

The lower-level free functions in `scene.hpp` (`loadGatesConfig`,
`loadCameraCalibration`, `renderPose`, ...) are still available directly if
you need more control (e.g. reloading a gate layout without re-reading the
camera calibration).

## Example

```sh
./build/examples/render_segmentation \
    --gates-config config/gates_config.yaml \
    --drone-config config/config.yaml \
    --camera-config config/camera_calibration.yaml \
    --output mask.png
```

Pass `-r`/`--rectified` to render as seen by a rectified (pinhole) view of
the fisheye camera instead of the raw fisheye projection.

## Using as a dependency

After `cmake --install build`, downstream projects can:

```cmake
find_package(segment_gate REQUIRED)
target_link_libraries(my_target PRIVATE segment_gate::segment_gate)
```
