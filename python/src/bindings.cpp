// nanobind module exposing detect_gates::GateRenderer to Python.
#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

#include <opencv2/core.hpp>

#include "detect_gates/gate_renderer.hpp"
#include "detect_gates/scene.hpp"

namespace nb = nanobind;
using namespace nb::literals;
using namespace detect_gates;

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

NB_MODULE(_detect_gates_renderer, m) {
    m.doc() = "Python bindings for detect_gates_renderer";

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

    nb::class_<Keypoint>(m, "Keypoint")
        .def_rw("name", &Keypoint::name)
        .def_rw("x", &Keypoint::x)
        .def_rw("y", &Keypoint::y)
        .def_rw("visible", &Keypoint::visible)
        .def_rw("in_frustum", &Keypoint::inFrustum)
        .def("__repr__", [](const Keypoint& k) {
            return "Keypoint(name='" + k.name + "', x=" + std::to_string(k.x) + ", y=" + std::to_string(k.y) +
                   ", visible=" + (k.visible ? "True" : "False") +
                   ", in_frustum=" + (k.inFrustum ? "True" : "False") + ")";
        });

    nb::class_<BoundingBox>(m, "BoundingBox")
        .def_rw("x1", &BoundingBox::x1)
        .def_rw("y1", &BoundingBox::y1)
        .def_rw("x2", &BoundingBox::x2)
        .def_rw("y2", &BoundingBox::y2)
        .def("__repr__", [](const BoundingBox& b) {
            return "BoundingBox(x1=" + std::to_string(b.x1) + ", y1=" + std::to_string(b.y1) +
                   ", x2=" + std::to_string(b.x2) + ", y2=" + std::to_string(b.y2) + ")";
        });

    nb::class_<GateDetection>(m, "GateDetection")
        .def_rw("gate", &GateDetection::gate)
        .def_rw("bounding_box", &GateDetection::boundingBox)
        .def_rw("keypoints", &GateDetection::keypoints)
        .def_prop_ro(
            "mask", [](const GateDetection& d) { return matToNdarray(d.mask); },
            // matToNdarray hands back an array that already owns its buffer through
            // a capsule, so the default reference_internal policy would be both
            // wrong and rejected.
            nb::rv_policy::automatic,
            "This gate's own silhouette as a (height, width) uint8 array, before other "
            "gates occlude it. All zeros when the gate projects nowhere, which is the "
            "exact test for whether it is in the picture at all.")
        .def_ro("mask_bounding_box", &GateDetection::maskBoundingBox,
                "Bounding box of `mask`. Follows the fisheye-curved silhouette, so unlike "
                "`bounding_box` it stays correct when the edges bow outside the corners and "
                "when no corner is in the frustum.")
        .def("__repr__", [](const GateDetection& d) {
            return "GateDetection(gate='" + d.gate + "', keypoints=" + std::to_string(d.keypoints.size()) + ")";
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
        .def(
            "render_detections",
            [](const GateRenderer& self, const DronePose& pose, int minVisibleCorners) {
                return self.renderDetections(pose, minVisibleCorners);
            },
            "pose"_a, "min_visible_corners"_a = 3,
            "Detect per-gate keypoints/bounding boxes for a DronePose.")
        .def(
            "render_detections",
            [](const GateRenderer& self, double x, double y, double z, double roll, double pitch, double yaw,
               int minVisibleCorners) { return self.renderDetections(x, y, z, roll, pitch, yaw, minVisibleCorners); },
            "x"_a, "y"_a, "z"_a, "roll"_a, "pitch"_a, "yaw"_a, "min_visible_corners"_a = 3)
        .def_prop_ro("image_width", &GateRenderer::imageWidth)
        .def_prop_ro("image_height", &GateRenderer::imageHeight);
}
