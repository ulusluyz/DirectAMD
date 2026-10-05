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
    src/device/device_discovery.cpp
    src/memory/va_manager.cpp
    src/memory/buffer_object.cpp
    src/isa/rdna_encoder.cpp
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
