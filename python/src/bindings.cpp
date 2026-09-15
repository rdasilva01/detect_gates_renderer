// nanobind module exposing detect_gates::GateRenderer to Python.
#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/array.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

#include <cstring>

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

// Copy N equally-sized CV_8UC1 masks into one contiguous (N, H, W) numpy array.
//
// One allocation and one owner rather than N of each: the caller wants a batch,
// and handing back a list of N zero-copy views would leave N cv::Mat capsules
// alive and force the consumer to stack them anyway. The copy is 4 KB a mask at
// the 64x64 the models use -- far below the render it just paid for.
nb::ndarray<nb::numpy, uint8_t, nb::shape<-1, -1, -1>> matsToNdarray(const std::vector<cv::Mat>& masks) {
    const size_t count = masks.size();
    const size_t rows = count ? static_cast<size_t>(masks[0].rows) : 0;
    const size_t cols = count ? static_cast<size_t>(masks[0].cols) : 0;
    auto* buffer = new std::vector<uint8_t>(count * rows * cols);
    for (size_t i = 0; i < count; ++i) {
        if (masks[i].type() != CV_8UC1 || !masks[i].isContinuous()) {
            delete buffer;
            throw std::runtime_error("expected continuous CV_8UC1 masks");
        }
        if (static_cast<size_t>(masks[i].rows) != rows || static_cast<size_t>(masks[i].cols) != cols) {
            delete buffer;
            throw std::runtime_error("batched masks must all be the same size");
        }
        std::memcpy(buffer->data() + i * rows * cols, masks[i].data, rows * cols);
    }
    nb::capsule owner(buffer, [](void* p) noexcept { delete static_cast<std::vector<uint8_t>*>(p); });
    return nb::ndarray<nb::numpy, uint8_t, nb::shape<-1, -1, -1>>(buffer->data(), {count, rows, cols}, owner);
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
        .def_static("from_quaternion", &poseFromQuaternion, "x"_a, "y"_a, "z"_a, "qw"_a, "qx"_a, "qy"_a, "qz"_a,
                    "Build a DronePose from a position and an ENU/FLU orientation quaternion, scalar "
                    "first: (qw, qx, qy, qz). Note ROS's geometry_msgs/Quaternion orders its fields "
                    "x, y, z, w. The quaternion need not be normalized. The result is interchangeable "
                    "with a roll/pitch/yaw DronePose and renders identically.")
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
            "gates occlude it. Same grid as render(), so it lines up with keypoints "
            "and bounding_box. At min_visible_corners=0 gates that project nowhere are "
            "dropped, so a returned detection always has something in its mask.")
        .def_ro("mask_bounding_box", &GateDetection::maskBoundingBox,
                "Bounding box of `mask`. Follows the fisheye-curved silhouette, so unlike "
                "`bounding_box` it stays correct when the edges bow outside the corners and "
                "when no corner is in the frustum.")
        .def("__repr__", [](const GateDetection& d) {
            return "GateDetection(gate='" + d.gate + "', keypoints=" + std::to_string(d.keypoints.size()) + ")";
        });

    nb::class_<GateEdges>(m, "GateEdges")
        .def_ro("gate", &GateEdges::gate)
        .def_prop_ro(
            "polylines",
            [](const GateEdges& e) {
                std::vector<std::vector<std::array<double, 2>>> out;
                out.reserve(e.polylines.size());
                for (const auto& polyline : e.polylines) {
                    std::vector<std::array<double, 2>> points;
                    points.reserve(polyline.size());
                    for (const auto& p : polyline) {
                        points.push_back({p.x, p.y});
                    }
                    out.push_back(std::move(points));
                }
                return out;
            },
            "Visible face edges as open polylines, each a list of (x, y) in output pixels.")
        .def("__repr__", [](const GateEdges& e) {
            return "GateEdges(gate='" + e.gate + "', polylines=" + std::to_string(e.polylines.size()) + ")";
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
            "render_batch",
            [](const GateRenderer& self, const std::vector<DronePose>& poses) {
                return matsToNdarray(self.render(poses));
            },
            "poses"_a,
            "Render many poses at once, as an (n, height, width) uint8 numpy array.\n\n"
            "Byte-for-byte what a Python loop over render() gives, parallelised over poses\n"
            "across cores. Only usable when the poses are known up front -- offline dataset\n"
            "rendering or a vectorised simulator, not a single-drone loop where the next\n"
            "pose depends on the current mask.")
        .def(
            "render",
            [](const GateRenderer& self, double x, double y, double z, double roll, double pitch, double yaw) {
                return matToNdarray(self.render(x, y, z, roll, pitch, yaw));
            },
            "x"_a, "y"_a, "z"_a, "roll"_a, "pitch"_a, "yaw"_a)
        .def(
            "render_segmented",
            [](const GateRenderer& self, const DronePose& pose) {
                const GateRenderer::Segmentation seg = self.renderSegmented(pose);
                return nb::make_tuple(matToNdarray(seg.coverage), matToNdarray(seg.instances));
            },
            "pose"_a,
            "Semantic coverage and instance labels for a DronePose, as a "
            "(coverage, instances) pair of (height, width) uint8 arrays.\n\n"
            "`coverage` is exactly what render() returns -- soft, because area "
            "resampling blends a thin frame into a partial value. `instances` is "
            "0 for background and otherwise the gate's 1-based index into "
            "gate_names, with the nearer gate owning any overlap; it is always "
            "resampled nearest-neighbour, because averaging label 3 and label 7 "
            "would produce label 5, a gate that is not there.")
        .def(
            "render_segmented",
            [](const GateRenderer& self, double x, double y, double z, double roll, double pitch, double yaw) {
                const GateRenderer::Segmentation seg = self.renderSegmented(x, y, z, roll, pitch, yaw);
                return nb::make_tuple(matToNdarray(seg.coverage), matToNdarray(seg.instances));
            },
            "x"_a, "y"_a, "z"_a, "roll"_a, "pitch"_a, "yaw"_a)
        .def_prop_ro("gate_names", &GateRenderer::gateNames,
                     "Gate names in label order: `instances == i + 1` is `gate_names[i]`.")
        .def(
            "render_detections",
            [](const GateRenderer& self, const DronePose& pose, int minVisibleCorners) {
                return self.renderDetections(pose, minVisibleCorners);
            },
            "pose"_a, "min_visible_corners"_a = 0,
            "Detect per-gate keypoints/bounding boxes for a DronePose.")
        .def(
            "render_detections",
            [](const GateRenderer& self, double x, double y, double z, double roll, double pitch, double yaw,
               int minVisibleCorners) { return self.renderDetections(x, y, z, roll, pitch, yaw, minVisibleCorners); },
            "x"_a, "y"_a, "z"_a, "roll"_a, "pitch"_a, "yaw"_a, "min_visible_corners"_a = 0)
        .def(
            "render_face_edges",
            [](const GateRenderer& self, const DronePose& pose) { return self.renderFaceEdges(pose); }, "pose"_a,
            "Each gate's visible face edges for a DronePose, for drawing: the near ring's outer edge and "
            "aperture and the walls facing the camera, as polylines in output pixels. Stretches hidden "
            "behind a nearer gate or the gate's own frame are left out.")
        .def(
            "render_face_edges",
            [](const GateRenderer& self, double x, double y, double z, double roll, double pitch, double yaw) {
                return self.renderFaceEdges(DronePose{x, y, z, roll, pitch, yaw});
            },
            "x"_a, "y"_a, "z"_a, "roll"_a, "pitch"_a, "yaw"_a)
        .def_prop_ro("image_width", &GateRenderer::imageWidth)
        .def_prop_ro("image_height", &GateRenderer::imageHeight);
}
