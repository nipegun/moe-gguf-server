cmake_minimum_required(VERSION 3.14...3.28)
project(moe-gguf-server C CXX)

# Build the engine, the server and the web interface from the backend tree.
add_subdirectory("${CMAKE_CURRENT_SOURCE_DIR}/../backend" "${CMAKE_CURRENT_BINARY_DIR}/backend")
