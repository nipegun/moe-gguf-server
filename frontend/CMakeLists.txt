set(TARGET mgs-ui)

set(MGS_UI_GZIP ON CACHE BOOL "Apply gzip compress to assets to save bandwidth")

# The interface version is always MGS_VERSION (cmake/build-info.cmake), which
# deploy/scripts/ui-assets.cmake reads at build time; check it here to fail early.
if("${MGS_VERSION}" STREQUAL "" OR NOT MGS_VERSION MATCHES "^[A-Za-z0-9._-]+$")
  message(FATAL_ERROR "UI: invalid MGS_VERSION='${MGS_VERSION}' - must match ^[A-Za-z0-9._-]+$")
endif()

set(vUiCpp "${CMAKE_CURRENT_BINARY_DIR}/ui.cpp")
set(vUiH   "${CMAKE_CURRENT_BINARY_DIR}/ui.h")

if(CMAKE_CROSSCOMPILING)
  find_program(HOST_CXX_COMPILER NAMES g++ clang++ NO_CMAKE_FIND_ROOT_PATH)
  if(NOT HOST_CXX_COMPILER)
    message(FATAL_ERROR "UI: no host C++ compiler (g++/clang++) found to build mgs-ui-embed; set -DHOST_CXX_COMPILER=<path>")
  endif()
  message(STATUS "UI: building mgs-ui-embed with host compiler ${HOST_CXX_COMPILER}")

  if(CMAKE_HOST_WIN32)
    set(vUiEmbedExecutable "${CMAKE_CURRENT_BINARY_DIR}/mgs-ui-embed-host.exe")
  else()
    set(vUiEmbedExecutable "${CMAKE_CURRENT_BINARY_DIR}/mgs-ui-embed-host")
  endif()

  add_custom_command(
    OUTPUT  "${vUiEmbedExecutable}"
    COMMAND "${HOST_CXX_COMPILER}" -O2 -std=c++17
        -o "${vUiEmbedExecutable}" "${CMAKE_CURRENT_SOURCE_DIR}/embed.cpp"
    DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/embed.cpp"
    COMMENT "Building mgs-ui-embed (host)"
    VERBATIM
  )

  # phony target to tie it into the dependency graph
  add_custom_target(mgs-ui-embed DEPENDS "${vUiEmbedExecutable}")
else()
  # exclude mgs-ui-embed from sanitizer flags,
  # it's a build-time-only tool, no need to instrument it
  # this is to fix TSan "memory layout is incompatible" error on CI
  get_directory_property(lUiDirectoryCompileOptions COMPILE_OPTIONS)
  get_directory_property(lUiDirectoryLinkLibraries LINK_LIBRARIES)
  set(lUiEmbedCompileOptions ${lUiDirectoryCompileOptions})
  set(lUiEmbedLinkLibraries ${lUiDirectoryLinkLibraries})
  list(FILTER lUiEmbedCompileOptions EXCLUDE REGEX ".*-fsanitize=.*")
  list(FILTER lUiEmbedLinkLibraries EXCLUDE REGEX ".*-fsanitize=.*")
  set_directory_properties(PROPERTIES
    COMPILE_OPTIONS "${lUiEmbedCompileOptions}"
    LINK_LIBRARIES  "${lUiEmbedLinkLibraries}")

  add_executable(mgs-ui-embed embed.cpp)
  target_compile_features(mgs-ui-embed PRIVATE cxx_std_17)
  set_target_properties(mgs-ui-embed PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}"
  )
  set(vUiEmbedExecutable "$<TARGET_FILE:mgs-ui-embed>")

  # restore so the mgs-ui library below keeps sanitizer instrumentation
  set_directory_properties(PROPERTIES
    COMPILE_OPTIONS "${lUiDirectoryCompileOptions}"
    LINK_LIBRARIES  "${lUiDirectoryLinkLibraries}")
endif()

# Run the provisioning script every build so source changes in frontend/ are
# always picked up. The script uses copy_if_different for ui.cpp/ui.h, so the
# library only recompiles when contents actually change.
add_custom_target(mgs-ui-assets ALL
  BYPRODUCTS ${vUiCpp} ${vUiH}
  COMMAND ${CMAKE_COMMAND}
    "-DUI_SOURCE_DIR=${CMAKE_CURRENT_SOURCE_DIR}"
    "-DUI_BINARY_DIR=${CMAKE_CURRENT_BINARY_DIR}"
    "-DMGS_SOURCE_DIR=${PROJECT_SOURCE_DIR}"
    "-DBUILD_UI=${MGS_BUILD_UI}"
    "-DMGS_UI_EMBED=${vUiEmbedExecutable}"
    "-DMGS_UI_GZIP=${MGS_UI_GZIP}"
    -P "${PROJECT_SOURCE_DIR}/../deploy/scripts/ui-assets.cmake"
  COMMENT "Provisioning UI assets"
  VERBATIM
)

add_dependencies(mgs-ui-assets mgs-ui-embed)

set_source_files_properties(${vUiCpp} ${vUiH} PROPERTIES GENERATED TRUE)

add_library(${TARGET} STATIC ${vUiCpp} ${vUiH})
target_compile_features(${TARGET} PRIVATE cxx_std_17)
add_dependencies(${TARGET} mgs-ui-assets)

if (BUILD_SHARED_LIBS)
  set_target_properties(${TARGET} PROPERTIES POSITION_INDEPENDENT_CODE ON)
endif()

target_include_directories(${TARGET} PUBLIC
  ${CMAKE_CURRENT_BINARY_DIR}
)
