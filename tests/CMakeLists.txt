add_executable(test_device test_device.cpp)
target_link_libraries(test_device PRIVATE directamd_runtime)
add_test(NAME test_device COMMAND test_device)

add_executable(test_memory test_memory.cpp)
target_link_libraries(test_memory PRIVATE directamd_runtime)
add_test(NAME test_memory COMMAND test_memory)

add_executable(test_isa test_isa.cpp)
target_link_libraries(test_isa PRIVATE directamd_runtime)
add_test(NAME test_isa COMMAND test_isa)

add_executable(test_command test_command.cpp)
target_link_libraries(test_command PRIVATE directamd_runtime)
add_test(NAME test_command COMMAND test_command)

add_executable(test_vector_add test_vector_add.cpp)
target_link_libraries(test_vector_add PRIVATE directamd_runtime)
add_test(NAME test_vector_add COMMAND test_vector_add)
