cmake_minimum_required(VERSION 3.16)
project(detect_gates_renderer VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release)
endif()

# Lets this static library be linked into shared objects (e.g. the Python
# extension module), and is harmless for executables/static-only consumers.
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

option(DETECT_GATES_RENDERER_BUILD_EXAMPLES "Build the example executables" ON)
# Auto-enabled when building via `pip install` (scikit-build-core sets SKBUILD,
# but to a version string like "2", not a literal ON/OFF that option() understands).
if(SKBUILD)
    set(DETECT_GATES_RENDERER_BUILD_PYTHON_DEFAULT ON)
else()
    set(DETECT_GATES_RENDERER_BUILD_PYTHON_DEFAULT OFF)
endif()
option(DETECT_GATES_RENDERER_BUILD_PYTHON "Build the Python bindings" ${DETECT_GATES_RENDERER_BUILD_PYTHON_DEFAULT})

find_package(OpenCV REQUIRED COMPONENTS core imgproc calib3d imgcodecs)
find_package(Eigen3 REQUIRED)
find_package(yaml-cpp REQUIRED)

add_library(detect_gates_renderer
    src/transforms.cpp
    src/gates.cpp
    src/projection.cpp
    src/render.cpp
    src/scene.cpp
    src/gate_renderer.cpp
)
add_library(detect_gates_renderer::detect_gates_renderer ALIAS detect_gates_renderer)

target_include_directories(detect_gates_renderer PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)

target_link_libraries(detect_gates_renderer PUBLIC
    ${OpenCV_LIBS}
    Eigen3::Eigen
    yaml-cpp
)

target_compile_features(detect_gates_renderer PUBLIC cxx_std_17)

include(GNUInstallDirs)
install(TARGETS detect_gates_renderer
    EXPORT detect_gates_rendererTargets
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)
install(DIRECTORY include/ DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
install(EXPORT detect_gates_rendererTargets
    FILE detect_gates_rendererTargets.cmake
    NAMESPACE detect_gates_renderer::
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/detect_gates_renderer
)

include(CMakePackageConfigHelpers)
configure_package_config_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/cmake/detect_gates_rendererConfig.cmake.in
    ${CMAKE_CURRENT_BINARY_DIR}/detect_gates_rendererConfig.cmake
    INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/detect_gates_renderer
)
write_basic_package_version_file(
    ${CMAKE_CURRENT_BINARY_DIR}/detect_gates_rendererConfigVersion.cmake
    VERSION ${PROJECT_VERSION}
    COMPATIBILITY SameMajorVersion
)
install(FILES
    ${CMAKE_CURRENT_BINARY_DIR}/detect_gates_rendererConfig.cmake
    ${CMAKE_CURRENT_BINARY_DIR}/detect_gates_rendererConfigVersion.cmake
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/detect_gates_renderer
)

if(DETECT_GATES_RENDERER_BUILD_EXAMPLES)
    add_subdirectory(examples)
endif()

if(DETECT_GATES_RENDERER_BUILD_PYTHON)
    add_subdirectory(python)
endif()
