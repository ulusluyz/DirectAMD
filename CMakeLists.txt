cmake_minimum_required(VERSION 3.20)
project(DirectAMD VERSION 0.1.0 LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Compiler warnings
if(MSVC)
    add_compile_options(/W4)
else()
    add_compile_options(-Wall -Wextra -Wpedantic)
    if(ENABLE_WERROR)
        add_compile_options(-Werror)
    endif()
endif()

# Find DRM headers
find_path(DRM_INCLUDE_DIR drm/amdgpu_drm.h HINTS /usr/include /usr/local/include)
if(NOT DRM_INCLUDE_DIR)
    message(FATAL_ERROR "Could not find Linux DRM kernel headers (drm/amdgpu_drm.h).")
endif()

# Include directories
include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${DRM_INCLUDE_DIR}
)

# Core Runtime Library
add_library(directamd_runtime STATIC
    src/runtime/error.cpp
    src/math/gemv.cpp
    src/math/gemm.cpp
    src/math/softmax.cpp
    src/math/rmsnorm.cpp
    src/math/rope.cpp
    src/math/silu.cpp
    src/math/vector_mul.cpp
    src/device/device_discovery.cpp
    src/memory/va_manager.cpp
    src/memory/buffer_object.cpp
    src/isa/rdna_encoder.cpp
    src/isa/polaris_encoder.cpp
    src/isa/polaris_softmax.cpp
    src/isa/polaris_rmsnorm.cpp
    src/isa/polaris_rope.cpp
    src/isa/polaris_silu.cpp
    src/command/pm4_builder.cpp
    src/command/command_submission.cpp
)

target_include_directories(directamd_runtime PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
    ${DRM_INCLUDE_DIR}
)

# Examples
add_subdirectory(examples)

# Tests
enable_testing()
add_subdirectory(tests)

# Persistent DirectAMD tensor backend used by the local Qwen2 inference CLI.
set_target_properties(directamd_runtime PROPERTIES POSITION_INDEPENDENT_CODE ON)
add_library(directamd_llm_backend SHARED src/llm/gpu_bridge.cpp)
target_link_libraries(directamd_llm_backend PRIVATE directamd_runtime)
target_compile_options(directamd_llm_backend PRIVATE -O2)
set_target_properties(directamd_llm_backend PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/llm")
find_package(Python3 COMPONENTS Interpreter REQUIRED)
configure_file(llm/directamd-llm.in "${CMAKE_BINARY_DIR}/directamd-llm" @ONLY)
file(CHMOD "${CMAKE_BINARY_DIR}/directamd-llm"
    PERMISSIONS OWNER_READ OWNER_WRITE OWNER_EXECUTE GROUP_READ GROUP_EXECUTE WORLD_READ WORLD_EXECUTE)
add_test(NAME test_llm_host
    COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/llm/test_host.py")

# Bounded-memory, CPU-only local GGUF conversion.
add_library(directamd_gguf_decode SHARED src/llm/gguf_decode.cpp)
target_compile_options(directamd_gguf_decode PRIVATE -O2 -ffp-contract=off)
set_target_properties(directamd_gguf_decode PROPERTIES LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/llm")
configure_file(llm/directamd-convert.in "${CMAKE_BINARY_DIR}/directamd-convert" @ONLY)
file(CHMOD "${CMAKE_BINARY_DIR}/directamd-convert"
    PERMISSIONS OWNER_READ OWNER_WRITE OWNER_EXECUTE GROUP_READ GROUP_EXECUTE WORLD_READ WORLD_EXECUTE)
add_test(NAME test_gguf_converter
    COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/llm/test_convert.py" "$<TARGET_FILE:directamd_gguf_decode>")
