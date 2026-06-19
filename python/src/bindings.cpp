// nanobind module exposing segment_gate::GateRenderer to Python.
#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/string.h>

#include <opencv2/core.hpp>

#include "segment_gate/gate_renderer.hpp"
#include "segment_gate/scene.hpp"

namespace nb = nanobind;
using namespace nb::literals;
using namespace segment_gate;

namespace {

// Wrap a CV_8UC1 cv::Mat as a zero-copy numpy array. The returned capsule
// holds a copy of the cv::Mat (a cheap, refcounted header) so the
// underlying pixel buffer stays alive as long as the numpy array does.
nb::ndarray<nb::numpy, uint8_t, nb::shape<-1, -1>> matToNdarray(const cv::Mat& mat) {
    if (mat.type() != CV_8UC1 || !mat.isContinuous()) {
        throw std::runtime_error("expected a continuous CV_8UC1 mask");
    }
    cv::Mat* owner = new cv::Mat(mat);
    nb::capsule owner_capsule(owner, [](void* p) noexcept { delete static_cast<cv::Mat*>(p); });
    return nb::ndarray<nb::numpy, uint8_t, nb::shape<-1, -1>>(
        owner->data, {static_cast<size_t>(owner->rows), static_cast<size_t>(owner->cols)}, owner_capsule);
}

}  // namespace

NB_MODULE(_segment_gate_renderer, m) {
    m.doc() = "Python bindings for segment_gate_renderer";

    nb::class_<DronePose>(m, "DronePose")
        .def(nb::init<>())
        .def(nb::init<double, double, double, double, double, double>(), "x"_a, "y"_a, "z"_a, "roll"_a,
             "pitch"_a, "yaw"_a)
        .def_rw("x", &DronePose::x)
        .def_rw("y", &DronePose::y)
        .def_rw("z", &DronePose::z)
        .def_rw("roll", &DronePose::roll)
        .def_rw("pitch", &DronePose::pitch)
        .def_rw("yaw", &DronePose::yaw)
        .def("__repr__", [](const DronePose& p) {
            return "DronePose(x=" + std::to_string(p.x) + ", y=" + std::to_string(p.y) +
                   ", z=" + std::to_string(p.z) + ", roll=" + std::to_string(p.roll) +
                   ", pitch=" + std::to_string(p.pitch) + ", yaw=" + std::to_string(p.yaw) + ")";
        });

    nb::class_<GateRenderer>(m, "GateRenderer")
        .def(nb::init<const std::string&, const std::string&, const std::string&, bool>(), "gates_config_path"_a,
             "drone_config_path"_a, "camera_config_path"_a, "rectified"_a = false,
             "Load a gates layout, gate dimensions, and camera calibration once, then render masks for "
             "arbitrary drone poses without re-parsing config files on every call.")
        .def(
            "render", [](const GateRenderer& self, const DronePose& pose) { return matToNdarray(self.render(pose)); },
            "pose"_a, "Render the segmentation mask for a DronePose, as a (height, width) uint8 numpy array.")
        .def(
            "render",
            [](const GateRenderer& self, double x, double y, double z, double roll, double pitch, double yaw) {
                return matToNdarray(self.render(x, y, z, roll, pitch, yaw));
            },
            "x"_a, "y"_a, "z"_a, "roll"_a, "pitch"_a, "yaw"_a)
        .def_prop_ro("image_width", &GateRenderer::imageWidth)
        .def_prop_ro("image_height", &GateRenderer::imageHeight);
}
