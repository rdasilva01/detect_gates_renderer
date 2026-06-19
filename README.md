# segment_gate_renderer

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
- (optional, for the Python bindings) Python >= 3.8, [nanobind](https://github.com/wjakob/nanobind), [scikit-build-core](https://github.com/scikit-build/scikit-build-core) — installed automatically by `pip install .`

## Build

```sh
cmake -B build
cmake --build build
```

This builds the `segment_gate_renderer` static library and (by default) the
`render_segmentation` example. Set `-DSEGMENT_GATE_RENDERER_BUILD_EXAMPLES=OFF`
to skip the example when embedding this project via `add_subdirectory`.

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
find_package(segment_gate_renderer REQUIRED)
target_link_libraries(my_target PRIVATE segment_gate_renderer::segment_gate_renderer)
```

## Python bindings

Python bindings for `GateRenderer` are built with
[nanobind](https://github.com/wjakob/nanobind) and packaged with
[scikit-build-core](https://github.com/scikit-build/scikit-build-core).

```sh
pip install .
```

```python
from segment_gate_renderer import DronePose, GateRenderer

renderer = GateRenderer("config/gates_config.yaml", "config/config.yaml", "config/camera_calibration.yaml")
mask = renderer.render(DronePose(x=19.0, y=2.0, z=0.155, roll=0.0, pitch=0.0, yaw=3.13))
# `mask` is a (height, width) uint8 numpy array. `renderer.render(x, y, z, roll, pitch, yaw)` also works.
```

See `examples/python/render_segmentation.py` for a runnable example (mirrors
`examples/render_segmentation.cpp`):

```sh
python examples/python/render_segmentation.py --output mask.png
```
