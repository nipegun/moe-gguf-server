cmake_minimum_required(VERSION 3.14...3.28) # for add_link_options and implicit target directories.
project("mgs" C CXX)
include(CheckIncludeFileCXX)

#set(CMAKE_WARN_DEPRECATED YES)
set(CMAKE_WARN_UNUSED_CLI YES)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if (NOT XCODE AND NOT MSVC AND NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
  set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release" "MinSizeRel" "RelWithDebInfo")
endif()

message("CMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}")

# Add path to modules
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake/")

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

if (CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR OR CMAKE_PROJECT_NAME STREQUAL "moe-gguf-server")
  set(MGS_STANDALONE ON)
else()
  set(MGS_STANDALONE OFF)
endif()

option(MGS_USE_SYSTEM_GGML "Use system libggml" OFF)

option(MGS_WASM_MEM64 "mgs: use 64-bit memory in WASM builds" ON)

if (EMSCRIPTEN)
  set(BUILD_SHARED_LIBS_DEFAULT OFF)

  # Use 64-bit memory to support backend_get_memory queries
  # TODO: analyze performance impact, see https://spidermonkey.dev/blog/2025/01/15/is-memory64-actually-worth-using
  if (MGS_WASM_MEM64)
   add_compile_options("-sMEMORY64=1")
   add_link_options("-sMEMORY64=1")
  endif()
  add_link_options("-sALLOW_MEMORY_GROWTH=1")

  option(MGS_WASM_SINGLE_FILE "mgs: embed WASM inside the generated mgs.js" OFF)
  option(MGS_BUILD_HTML "mgs: build HTML file" ON)
  if (MGS_BUILD_HTML)
    set(CMAKE_EXECUTABLE_SUFFIX ".html")
  endif()
else()
  if (MINGW)
    set(BUILD_SHARED_LIBS_DEFAULT OFF)
  else()
    set(BUILD_SHARED_LIBS_DEFAULT ON)
  endif()
endif()

option(BUILD_SHARED_LIBS "build shared libraries" ${BUILD_SHARED_LIBS_DEFAULT})

if (WIN32)
  add_compile_definitions(_CRT_SECURE_NO_WARNINGS)
endif()

if (MSVC)
  add_compile_options("$<$<COMPILE_LANGUAGE:C>:/utf-8>")
  add_compile_options("$<$<COMPILE_LANGUAGE:CXX>:/utf-8>")
  add_compile_options("$<$<COMPILE_LANGUAGE:C>:/bigobj>")
  add_compile_options("$<$<COMPILE_LANGUAGE:CXX>:/bigobj>")
endif()

if (MGS_STANDALONE)
  # enable parallel builds for msbuild
  list(APPEND CMAKE_VS_GLOBALS UseMultiToolTask=true)
  list(APPEND CMAKE_VS_GLOBALS EnforceProcessCountAcrossBuilds=true)
endif()

if (CMAKE_SYSTEM_NAME STREQUAL "iOS")
  set(MGS_TOOLS_INSTALL_DEFAULT OFF)
else()
  set(MGS_TOOLS_INSTALL_DEFAULT ${MGS_STANDALONE})
endif()

# subprocess spawning isn't a supported/sandbox-friendly operation on mobile OSes or in WASM
if (CMAKE_SYSTEM_NAME STREQUAL "iOS" OR CMAKE_SYSTEM_NAME STREQUAL "Android" OR ANDROID
    OR CMAKE_SYSTEM_NAME STREQUAL "Emscripten" OR EMSCRIPTEN)
  set(MGS_SUBPROCESS_DEFAULT OFF)
else()
  set(MGS_SUBPROCESS_DEFAULT ON)
endif()

#
# option list
#

# debug
option(MGS_ALL_WARNINGS           "mgs: enable all compiler warnings"                   ON)
option(MGS_ALL_WARNINGS_3RD_PARTY "mgs: enable all compiler warnings in 3rd party libs" OFF)

# build
option(MGS_FATAL_WARNINGS "mgs: enable -Werror flag" OFF)

# sanitizers
option(MGS_SANITIZE_THREAD    "mgs: enable thread sanitizer"    OFF)
option(MGS_SANITIZE_ADDRESS   "mgs: enable address sanitizer"   OFF)
option(MGS_SANITIZE_UNDEFINED "mgs: enable undefined sanitizer" OFF)

option(MGS_BUILD_UI        "mgs: build the embedded Web UI for server"                                       ON)

option(MGS_TOOLS_INSTALL "mgs: install tools" ${MGS_TOOLS_INSTALL_DEFAULT})

# 3rd party libs
option(MGS_OPENSSL         "mgs: use openssl to support HTTPS" ON)
option(MGS_BUILD_BORINGSSL "mgs: download and build BoringSSL to support HTTPS instead of the system OpenSSL" OFF)
option(MGS_BUILD_LIBRESSL  "mgs: download and build LibreSSL to support HTTPS instead of the system OpenSSL" OFF)
option(MGS_SUBPROCESS      "mgs-common: use subprocess, required by server tools and server router mode" ${MGS_SUBPROCESS_DEFAULT})
option(MGS_LLGUIDANCE      "mgs-common: include LLGuidance library for structured output in common utils" OFF)


# Required for relocatable CMake package
include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/build-info.cmake)
include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/common.cmake)

message(STATUS "mgs version: ${MGS_VERSION}")

# override ggml options
set(GGML_ALL_WARNINGS   ${MGS_ALL_WARNINGS})
set(GGML_FATAL_WARNINGS ${MGS_FATAL_WARNINGS})

# change the default for these ggml options
if (NOT DEFINED GGML_TINYBLAS)
  set(GGML_TINYBLAS_DEFAULT ON)
endif()

if (NOT DEFINED GGML_CUDA_GRAPHS)
  set(GGML_CUDA_GRAPHS_DEFAULT ON)
endif()

#
# 3rd-party
#

if (MGS_USE_SYSTEM_GGML)
  message(STATUS "Using system-provided libggml, skipping ggml build")
  find_package(ggml REQUIRED)
  add_library(ggml ALIAS ggml::ggml)
endif()

if (NOT TARGET ggml AND NOT MGS_USE_SYSTEM_GGML)
  add_subdirectory(ggml)
  # ... otherwise assume ggml is added by a parent CMakeLists.txt
endif()

#
# build the library
#

add_subdirectory(src)

#
# server
#

add_subdirectory(common)
add_subdirectory(vendor/cpp-httplib)
add_subdirectory(tools)

#
# install
#

include(GNUInstallDirs)
include(CMakePackageConfigHelpers)

set(MGS_INCLUDE_INSTALL_DIR ${CMAKE_INSTALL_INCLUDEDIR} CACHE PATH "Location of header  files")
set(MGS_LIB_INSTALL_DIR     ${CMAKE_INSTALL_LIBDIR}     CACHE PATH "Location of library files")
set(MGS_BIN_INSTALL_DIR     ${CMAKE_INSTALL_BINDIR}     CACHE PATH "Location of binary  files")

set(MGS_PUBLIC_HEADERS
  ${CMAKE_CURRENT_SOURCE_DIR}/include/mgs.h
  ${CMAKE_CURRENT_SOURCE_DIR}/include/mgs-cpp.h)

set_target_properties(mgs-engine
  PROPERTIES
    PUBLIC_HEADER "${MGS_PUBLIC_HEADERS}")

install(TARGETS mgs-engine LIBRARY PUBLIC_HEADER)

install(TARGETS mgs-common LIBRARY)

configure_package_config_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/cmake/mgs-config.cmake.in
    ${CMAKE_CURRENT_BINARY_DIR}/mgs-config.cmake
  INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/mgs
  PATH_VARS MGS_INCLUDE_INSTALL_DIR
       MGS_LIB_INSTALL_DIR
       MGS_BIN_INSTALL_DIR )

write_basic_package_version_file(
    ${CMAKE_CURRENT_BINARY_DIR}/mgs-version.cmake
  VERSION ${MGS_VERSION}
  COMPATIBILITY SameMajorVersion)

install(FILES ${CMAKE_CURRENT_BINARY_DIR}/mgs-config.cmake
       ${CMAKE_CURRENT_BINARY_DIR}/mgs-version.cmake
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/mgs)

configure_file(cmake/mgs.pc.in
    "${CMAKE_CURRENT_BINARY_DIR}/mgs.pc"
    @ONLY)

install(FILES "${CMAKE_CURRENT_BINARY_DIR}/mgs.pc"
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/pkgconfig)
