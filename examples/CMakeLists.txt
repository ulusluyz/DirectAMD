add_executable(vector_add_example vector_add/main.cpp)
target_link_libraries(vector_add_example PRIVATE directamd_runtime)

add_executable(gemv_example gemv/main.cpp)
target_link_libraries(gemv_example PRIVATE directamd_runtime)

add_executable(gemm_example gemm/main.cpp)
target_link_libraries(gemm_example PRIVATE directamd_runtime)

add_executable(softmax_example softmax/main.cpp)
target_link_libraries(softmax_example PRIVATE directamd_runtime)

add_executable(rmsnorm_example rmsnorm/main.cpp)
target_link_libraries(rmsnorm_example PRIVATE directamd_runtime)

add_executable(rope_example rope/main.cpp)
target_link_libraries(rope_example PRIVATE directamd_runtime)

add_executable(silu_example silu/main.cpp)
target_link_libraries(silu_example PRIVATE directamd_runtime)

add_executable(vector_mul_example vector_mul/main.cpp)
target_link_libraries(vector_mul_example PRIVATE directamd_runtime)
