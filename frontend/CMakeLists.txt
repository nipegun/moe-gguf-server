set(TARGET mgs-ui)

set(MGS_UI_GZIP ON CACHE BOOL "Apply gzip compress to assets to save bandwidth")

# Resolve the interface version: explicit env var > project version > unset
if(DEFINED ENV{UI_VERSION})
  set(UI_VERSION "$ENV{UI_VERSION}")
elseif(DEFINED MGS_VERSION)
  set(UI_VERSION "${MGS_VERSION}")
else()
  set(UI_VERSION "")
endif()

if(NOT "${UI_VERSION}" STREQUAL "" AND NOT UI_VERSION MATCHES "^[A-Za-z0-9._-]+$")
  message(FATAL_ERROR "UI: invalid UI_VERSION='${UI_VERSION}' - must match ^[A-Za-z0-9._-]+$")
endif()

set(UI_CPP "${CMAKE_CURRENT_BINARY_DIR}/ui.cpp")
set(UI_H   "${CMAKE_CURRENT_BINARY_DIR}/ui.h")

if(CMAKE_CROSSCOMPILING)
  find_program(HOST_CXX_COMPILER NAMES g++ clang++ NO_CMAKE_FIND_ROOT_PATH)
  if(NOT HOST_CXX_COMPILER)
    message(FATAL_ERROR "UI: no host C++ compiler (g++/clang++) found to build mgs-ui-embed; set -DHOST_CXX_COMPILER=<path>")
  endif()
  message(STATUS "UI: building mgs-ui-embed with host compiler ${HOST_CXX_COMPILER}")

  if(CMAKE_HOST_WIN32)
    set(MGS_UI_EMBED_EXE "${CMAKE_CURRENT_BINARY_DIR}/mgs-ui-embed-host.exe")
  else()
    set(MGS_UI_EMBED_EXE "${CMAKE_CURRENT_BINARY_DIR}/mgs-ui-embed-host")
  endif()

  add_custom_command(
    OUTPUT  "${MGS_UI_EMBED_EXE}"
    COMMAND "${HOST_CXX_COMPILER}" -O2 -std=c++17
        -o "${MGS_UI_EMBED_EXE}" "${CMAKE_CURRENT_SOURCE_DIR}/embed.cpp"
    DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/embed.cpp"
    COMMENT "Building mgs-ui-embed (host)"
    VERBATIM
  )

  # phony target to tie it into the dependency graph
  add_custom_target(mgs-ui-embed DEPENDS "${MGS_UI_EMBED_EXE}")
else()
  # exclude mgs-ui-embed from sanitizer flags,
  # it's a build-time-only tool, no need to instrument it
  # this is to fix TSan "memory layout is incompatible" error on CI
  get_directory_property(_mgs_ui_dir_co COMPILE_OPTIONS)
  get_directory_property(_mgs_ui_dir_ll LINK_LIBRARIES)
  set(_mgs_ui_embed_co ${_mgs_ui_dir_co})
  set(_mgs_ui_embed_ll ${_mgs_ui_dir_ll})
  list(FILTER _mgs_ui_embed_co EXCLUDE REGEX ".*-fsanitize=.*")
  list(FILTER _mgs_ui_embed_ll EXCLUDE REGEX ".*-fsanitize=.*")
  set_directory_properties(PROPERTIES
    COMPILE_OPTIONS "${_mgs_ui_embed_co}"
    LINK_LIBRARIES  "${_mgs_ui_embed_ll}")

  add_executable(mgs-ui-embed embed.cpp)
  target_compile_features(mgs-ui-embed PRIVATE cxx_std_17)
  set_target_properties(mgs-ui-embed PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}"
  )
  set(MGS_UI_EMBED_EXE "$<TARGET_FILE:mgs-ui-embed>")

  # restore so the mgs-ui library below keeps sanitizer instrumentation
  set_directory_properties(PROPERTIES
    COMPILE_OPTIONS "${_mgs_ui_dir_co}"
    LINK_LIBRARIES  "${_mgs_ui_dir_ll}")
endif()

# Run the provisioning script every build so source changes in frontend/ are
# always picked up. The script uses copy_if_different for ui.cpp/ui.h, so the
# library only recompiles when contents actually change.
add_custom_target(mgs-ui-assets ALL
  BYPRODUCTS ${UI_CPP} ${UI_H}
  COMMAND ${CMAKE_COMMAND}
    "-DUI_SOURCE_DIR=${CMAKE_CURRENT_SOURCE_DIR}"
    "-DUI_BINARY_DIR=${CMAKE_CURRENT_BINARY_DIR}"
    "-DMGS_SOURCE_DIR=${PROJECT_SOURCE_DIR}"
    "-DUI_VERSION=${UI_VERSION}"
    "-DBUILD_UI=${MGS_BUILD_UI}"
    "-DMGS_UI_EMBED=${MGS_UI_EMBED_EXE}"
    "-DMGS_UI_GZIP=${MGS_UI_GZIP}"
    -P "${PROJECT_SOURCE_DIR}/../deploy/scripts/ui-assets.cmake"
  COMMENT "Provisioning UI assets"
  VERBATIM
)

add_dependencies(mgs-ui-assets mgs-ui-embed)

set_source_files_properties(${UI_CPP} ${UI_H} PROPERTIES GENERATED TRUE)

add_library(${TARGET} STATIC ${UI_CPP} ${UI_H})
target_compile_features(${TARGET} PRIVATE cxx_std_17)
add_dependencies(${TARGET} mgs-ui-assets)

if (BUILD_SHARED_LIBS)
  set_target_properties(${TARGET} PROPERTIES POSITION_INDEPENDENT_CODE ON)
endif()

target_include_directories(${TARGET} PUBLIC
  ${CMAKE_CURRENT_BINARY_DIR}
)
