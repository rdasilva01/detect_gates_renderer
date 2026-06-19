// Overlay pose-mode detections on top of the segment-mode mask, for the same
// pose, as a visual cross-check that the two independently-implemented code
// paths (renderSegmentation vs. detectGates) agree: keypoints should sit
// right on the mask's frame edges, and bounding boxes should hug each gate's
// silhouette.
#include <iostream>
#include <string>

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include "detect_gates/gate_renderer.hpp"
#include "detect_gates/scene.hpp"

namespace {

// "top_left_inner" -> "tl_i": first letter of each word in the side
// ("top_left" -> "tl"), plus the first letter of the ring ("inner"/"outer"),
// short enough to label a point without cluttering the image.
std::string abbreviate(const std::string& name) {
    const size_t lastUnderscore = name.rfind('_');
    const std::string side = name.substr(0, lastUnderscore);    // "top_left"
    const std::string ring = name.substr(lastUnderscore + 1);   // "inner" or "outer"

    std::string abbrev;
    abbrev += side.front();
    abbrev += side[side.find('_') + 1];
    return abbrev + "_" + ring.substr(0, 1);
}

struct Args {
    std::string gatesConfig = "config/gates_config.yaml";
    std::string droneConfig = "config/config.yaml";
    std::string cameraConfig = "config/camera_calibration.yaml";
    std::string output;
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
        } else if (arg == "-r" || arg == "--rectified") {
            args.rectified = true;
        } else {
            throw std::runtime_error("unknown argument: " + arg);
        }
    }
    if (args.output.empty()) {
        throw std::runtime_error("--output is required");
    }
    return args;
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

    const cv::Mat mask = renderer.render(dronePos);
    cv::Mat overlay;
    cv::cvtColor(mask, overlay, cv::COLOR_GRAY2BGR);

    const std::vector<GateDetection> detections = renderer.renderDetections(dronePos);
    for (const auto& det : detections) {
        const cv::Point topLeft(static_cast<int>(det.boundingBox.x1), static_cast<int>(det.boundingBox.y1));
        const cv::Point bottomRight(static_cast<int>(det.boundingBox.x2), static_cast<int>(det.boundingBox.y2));
        cv::rectangle(overlay, topLeft, bottomRight, cv::Scalar(0, 255, 255), 1);
        cv::putText(overlay, det.gate, topLeft + cv::Point(0, -4), cv::FONT_HERSHEY_SIMPLEX, 0.4,
                    cv::Scalar(0, 255, 255), 1);

        for (const auto& kp : det.keypoints) {
            if (!kp.inFrustum) {
                // Outside the camera's field of view: (x, y) is not a
                // meaningful pixel location, so there's nothing to draw.
                continue;
            }
            const cv::Point center(static_cast<int>(kp.x), static_cast<int>(kp.y));
            const cv::Scalar color = kp.visible ? cv::Scalar(0, 255, 0) : cv::Scalar(0, 0, 255);
            cv::circle(overlay, center, 3, color, cv::FILLED);
            cv::putText(overlay, abbreviate(kp.name), center + cv::Point(4, 4), cv::FONT_HERSHEY_SIMPLEX, 0.3, color,
                        1);
        }
    }

    if (!cv::imwrite(args.output, overlay)) {
        std::cerr << "error: failed to write " << args.output << "\n";
        return 1;
    }

    return 0;
}
