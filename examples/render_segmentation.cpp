// Render a gate-segmentation image as seen from a given drone pose.
//
// Given a gates layout, a drone pose, and a camera calibration, renders a
// grayscale PNG (0 = background, 255 = gate frame) of what the drone's
// camera would see from that pose. Non-interactive port of
// map_error_net's scripts/generate_segmentation.py (default mode only).
#include <iostream>
#include <string>

#include <opencv2/imgcodecs.hpp>

#include "segment_gate/gate_renderer.hpp"
#include "segment_gate/scene.hpp"

namespace {

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
    using namespace segment_gate;

    Args args;
    try {
        args = parseArgs(argc, argv);
    } catch (const std::exception& e) {
        std::cerr << "error: " << e.what() << "\n";
        return 1;
    }

    const DronePose dronePos = loadDroneConfig(args.droneConfig).first;
    const GateRenderer renderer(args.gatesConfig, args.droneConfig, args.cameraConfig, args.rectified);

    const cv::Mat mask = renderer.render(dronePos);

    if (!cv::imwrite(args.output, mask)) {
        std::cerr << "error: failed to write " << args.output << "\n";
        return 1;
    }

    return 0;
}
