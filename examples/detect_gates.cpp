// Render a gate-segmentation mask or pose/keypoint detections for a drone pose.
//
// Given a gates layout, a drone pose, and a camera calibration:
// --mode segment (default) renders a grayscale PNG (0 = background,
// 255 = gate frame) of what the drone's camera would see from that pose.
// --mode pose instead writes a JSON array of per-gate keypoint (4 inner +
// 4 outer corner) and bounding-box detections, with cross-gate occlusion
// handling.
#include <fstream>
#include <iostream>
#include <string>

#include <opencv2/imgcodecs.hpp>

#include "detect_gates/gate_renderer.hpp"
#include "detect_gates/scene.hpp"

namespace {

struct Args {
    std::string gatesConfig = "config/gates_config.yaml";
    std::string droneConfig = "config/config.yaml";
    std::string cameraConfig = "config/camera_calibration.yaml";
    std::string output;
    std::string mode = "segment";
    bool rectified = false;
};

Args parseArgs(int argc, char** argv) {
    Args args;
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        auto nextValue = [&]() -> std::string {
            if (i + 1 >= argc) {
                throw std::runtime_error("missing value for " + arg);
            }
            return argv[++i];
        };

        if (arg == "--gates-config") {
            args.gatesConfig = nextValue();
        } else if (arg == "--drone-config") {
            args.droneConfig = nextValue();
        } else if (arg == "--camera-config") {
            args.cameraConfig = nextValue();
        } else if (arg == "--output") {
            args.output = nextValue();
        } else if (arg == "--mode") {
            args.mode = nextValue();
        } else if (arg == "-r" || arg == "--rectified") {
            args.rectified = true;
        } else {
            throw std::runtime_error("unknown argument: " + arg);
        }
    }
    if (args.output.empty()) {
        throw std::runtime_error("--output is required");
    }
    if (args.mode != "segment" && args.mode != "pose") {
        throw std::runtime_error("--mode must be 'segment' or 'pose'");
    }
    return args;
}

void writeDetectionsJson(std::ostream& out, const std::vector<detect_gates::GateDetection>& detections) {
    out << "[\n";
    for (size_t i = 0; i < detections.size(); ++i) {
        const auto& det = detections[i];
        out << "  {\"gate\": \"" << det.gate << "\", \"bounding_box\": {"
            << "\"x1\": " << det.boundingBox.x1 << ", \"y1\": " << det.boundingBox.y1
            << ", \"x2\": " << det.boundingBox.x2 << ", \"y2\": " << det.boundingBox.y2 << "}, "
            << "\"keypoints\": [";
        for (size_t k = 0; k < det.keypoints.size(); ++k) {
            const auto& kp = det.keypoints[k];
            out << "{\"name\": \"" << kp.name << "\", \"x\": " << kp.x << ", \"y\": " << kp.y
                << ", \"visible\": " << (kp.visible ? "true" : "false") << "}"
                << (k + 1 < det.keypoints.size() ? ", " : "");
        }
        out << "]}" << (i + 1 < detections.size() ? ",\n" : "\n");
    }
    out << "]\n";
}

}  // namespace

int main(int argc, char** argv) {
    using namespace detect_gates;

    Args args;
    try {
        args = parseArgs(argc, argv);
    } catch (const std::exception& e) {
        std::cerr << "error: " << e.what() << "\n";
        return 1;
    }

    const DronePose dronePos{19.0, 2.0, 0.155, 0.0, 0.0, 3.13};
    const GateRenderer renderer(args.gatesConfig, args.droneConfig, args.cameraConfig, args.rectified);

    if (args.mode == "segment") {
        const cv::Mat mask = renderer.render(dronePos);
        if (!cv::imwrite(args.output, mask)) {
            std::cerr << "error: failed to write " << args.output << "\n";
            return 1;
        }
    } else {
        const std::vector<GateDetection> detections = renderer.renderDetections(dronePos);
        std::ofstream out(args.output);
        if (!out) {
            std::cerr << "error: failed to write " << args.output << "\n";
            return 1;
        }
        writeDetectionsJson(out, detections);
    }

    return 0;
}
