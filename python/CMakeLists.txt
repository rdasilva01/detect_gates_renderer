find_package(Python 3.8 COMPONENTS Interpreter Development.Module REQUIRED)

execute_process(
    COMMAND "${Python_EXECUTABLE}" -c "import nanobind; print(nanobind.cmake_dir())"
    OUTPUT_VARIABLE NB_DIR
    OUTPUT_STRIP_TRAILING_WHITESPACE
)
list(APPEND CMAKE_PREFIX_PATH "${NB_DIR}")
find_package(nanobind CONFIG REQUIRED)

nanobind_add_module(_segment_gate_renderer src/bindings.cpp)
target_link_libraries(_segment_gate_renderer PRIVATE segment_gate_renderer::segment_gate_renderer)

install(TARGETS _segment_gate_renderer LIBRARY DESTINATION segment_gate_renderer)
