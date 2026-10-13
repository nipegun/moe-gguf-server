# dependencies

find_package(Threads REQUIRED)

# third-party

# ...

# flags

mgs_add_compile_flags()

# mgs and its direct dependencies

add_subdirectory(mtmd)
add_subdirectory(${PROJECT_SOURCE_DIR}/../frontend ${CMAKE_BINARY_DIR}/frontend)
add_subdirectory(server)
