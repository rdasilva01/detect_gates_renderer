cmake_minimum_required(VERSION 3.16)
project(segment_gate VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release)
endif()

option(SEGMENT_GATE_BUILD_EXAMPLES "Build the example executables" ON)

find_package(OpenCV REQUIRED COMPONENTS core imgproc calib3d imgcodecs)
find_package(Eigen3 REQUIRED)
find_package(yaml-cpp REQUIRED)

add_library(segment_gate
    src/transforms.cpp
    src/gates.cpp
    src/projection.cpp
    src/render.cpp
    src/scene.cpp
    src/gate_renderer.cpp
)
add_library(segment_gate::segment_gate ALIAS segment_gate)

target_include_directories(segment_gate PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)

target_link_libraries(segment_gate PUBLIC
    ${OpenCV_LIBS}
    Eigen3::Eigen
    yaml-cpp
)

target_compile_features(segment_gate PUBLIC cxx_std_17)

include(GNUInstallDirs)
install(TARGETS segment_gate
    EXPORT segment_gateTargets
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)
install(DIRECTORY include/ DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
install(EXPORT segment_gateTargets
    FILE segment_gateTargets.cmake
    NAMESPACE segment_gate::
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/segment_gate
)

include(CMakePackageConfigHelpers)
configure_package_config_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/cmake/segment_gateConfig.cmake.in
    ${CMAKE_CURRENT_BINARY_DIR}/segment_gateConfig.cmake
    INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/segment_gate
)
write_basic_package_version_file(
    ${CMAKE_CURRENT_BINARY_DIR}/segment_gateConfigVersion.cmake
    VERSION ${PROJECT_VERSION}
    COMPATIBILITY SameMajorVersion
)
install(FILES
    ${CMAKE_CURRENT_BINARY_DIR}/segment_gateConfig.cmake
    ${CMAKE_CURRENT_BINARY_DIR}/segment_gateConfigVersion.cmake
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/segment_gate
)

if(SEGMENT_GATE_BUILD_EXAMPLES)
    add_subdirectory(examples)
endif()
