# Provision UI assets and generate ui.cpp/ui.h.
#
# With BUILD_UI=ON the interface is built with npm; with BUILD_UI=OFF the
# server is built without an embedded interface.

cmake_minimum_required(VERSION 3.18)

set(UI_SOURCE_DIR     "" CACHE STRING "UI source directory (to run npm build)")
set(UI_BINARY_DIR     "" CACHE STRING "UI binary directory (to store generated files)")
set(MGS_SOURCE_DIR    "" CACHE STRING "Backend source root (to resolve the version from cmake/build-info.cmake)")
set(UI_VERSION        "" CACHE STRING "Interface version (empty = resolve from cmake/build-info.cmake)")
set(BUILD_UI          "" CACHE STRING "Build UI via npm (ON/OFF)")
set(MGS_UI_EMBED      "" CACHE STRING "Path to mgs-ui-embed helper")
set(MGS_UI_GZIP       "" CACHE STRING "Apply gzip compress to assets to save bandwidth")

set(DIST_DIR     "${UI_BINARY_DIR}/dist")
set(WORK_DIR     "${UI_BINARY_DIR}/ui-src")
set(STAMP_FILE   "${UI_BINARY_DIR}/.ui-stamp")
set(UI_CPP       "${UI_BINARY_DIR}/ui.cpp")
set(UI_H         "${UI_BINARY_DIR}/ui.h")

function(npm_build_should_skip out_var)
  set(${out_var} FALSE PARENT_SCOPE)

  if(NOT EXISTS "${DIST_DIR}/index.html")
    return()
  endif()

  if(EXISTS "${STAMP_FILE}")
    return()
  endif()

  # A version change must rewrite build.json even when no interface source changed
  if(NOT EXISTS "${DIST_DIR}/build.json")
    return()
  endif()
  file(READ "${DIST_DIR}/build.json" build_json)
  if(NOT build_json STREQUAL "{\"version\":\"${UI_VERSION}\"}")
    return()
  endif()

  if(NOT EXISTS "${UI_SOURCE_DIR}/sources.cmake")
    return()
  endif()
  include("${UI_SOURCE_DIR}/sources.cmake")

  set(globs "")
  foreach(g ${UI_SOURCE_GLOBS})
    list(APPEND globs "${UI_SOURCE_DIR}/${g}")
  endforeach()
  file(GLOB_RECURSE sources ${globs})
  foreach(f ${UI_SOURCE_FILES})
    list(APPEND sources "${UI_SOURCE_DIR}/${f}")
  endforeach()

  file(TIMESTAMP "${DIST_DIR}/index.html" out_ts)

  foreach(s ${sources})
    if(NOT EXISTS "${s}")
      continue()
    endif()
    file(TIMESTAMP "${s}" s_ts)
    if(s_ts STRGREATER out_ts)
      return()
    endif()
  endforeach()

  set(${out_var} TRUE PARENT_SCOPE)
endfunction()

function(stage_sources)
  if(EXISTS "${WORK_DIR}")
    file(GLOB staged RELATIVE "${WORK_DIR}" "${WORK_DIR}/*")
    list(REMOVE_ITEM staged "node_modules")
    foreach(entry ${staged})
      file(REMOVE_RECURSE "${WORK_DIR}/${entry}")
    endforeach()
  endif()

  file(COPY "${UI_SOURCE_DIR}/"
    DESTINATION "${WORK_DIR}"
    NO_SOURCE_PERMISSIONS
    PATTERN "node_modules" EXCLUDE
  )
endfunction()

function(npm_build out_var)
  set(${out_var} FALSE PARENT_SCOPE)

  if(NOT EXISTS "${UI_SOURCE_DIR}/package.json")
    message(STATUS "UI: ${UI_SOURCE_DIR}/package.json not found, skipping npm")
    return()
  endif()

  npm_build_should_skip(skip)
  if(skip)
    message(STATUS "UI: npm output up-to-date, skipping build")
    set(${out_var} TRUE PARENT_SCOPE)
    return()
  endif()

  if(CMAKE_HOST_WIN32)
    find_program(NPM_EXECUTABLE NAMES npm.cmd npm.bat npm)
  else()
    find_program(NPM_EXECUTABLE npm)
  endif()
  if(NOT NPM_EXECUTABLE)
    message(STATUS "UI: npm not found, skipping npm build")
    return()
  endif()

  stage_sources()

  # Use our own marker because npm can write its internal lock before an
  # interrupted installation has populated every package completely.
  set(NPM_MARKER "${WORK_DIR}/node_modules/.mgs-ui-install-stamp")
  set(need_install FALSE)
  if(NOT EXISTS "${NPM_MARKER}")
    set(need_install TRUE)
  else()
    file(TIMESTAMP "${WORK_DIR}/package-lock.json" lock_ts)
    file(TIMESTAMP "${NPM_MARKER}" marker_ts)
    if(lock_ts STRGREATER marker_ts)
      set(need_install TRUE)
    endif()
  endif()

  if(need_install)
    message(STATUS "UI: running npm ci")
    execute_process(
      COMMAND ${NPM_EXECUTABLE} ci
      WORKING_DIRECTORY "${WORK_DIR}"
      RESULT_VARIABLE rc
      ERROR_VARIABLE  err
    )
    if(NOT rc EQUAL 0)
      message(STATUS "UI: npm ci failed (${rc})")
      message(STATUS "  stderr: ${err}")
      return()
    endif()
    file(WRITE "${NPM_MARKER}" "installed\n")
  endif()

  file(MAKE_DIRECTORY "${DIST_DIR}")

  message(STATUS "UI: running npm run build, output -> ${DIST_DIR}")
  execute_process(
    COMMAND ${CMAKE_COMMAND} -E env "MGS_UI_OUT_DIR=${DIST_DIR}" "MGS_UI_VERSION=${UI_VERSION}"
        ${NPM_EXECUTABLE} run build
    WORKING_DIRECTORY "${WORK_DIR}"
    RESULT_VARIABLE rc
    ERROR_VARIABLE  err
  )
  if(NOT rc EQUAL 0)
    message(STATUS "UI: npm run build failed (${rc})")
    message(STATUS "  stderr: ${err}")
    return()
  endif()

  if(NOT EXISTS "${DIST_DIR}/index.html")
    message(STATUS "UI: npm build finished but assets missing in ${DIST_DIR}")
    return()
  endif()

  message(STATUS "UI: npm build succeeded")
  file(REMOVE "${STAMP_FILE}")
  set(${out_var} TRUE PARENT_SCOPE)
endfunction()

function(resolve_version out_var)
  if(NOT "${UI_VERSION}" STREQUAL "")
    set(${out_var} "${UI_VERSION}" PARENT_SCOPE)
    return()
  endif()

  if(EXISTS "${MGS_SOURCE_DIR}/cmake/build-info.cmake")
    include("${MGS_SOURCE_DIR}/cmake/build-info.cmake")
    if(NOT "${MGS_VERSION}" STREQUAL "")
      set(${out_var} "${MGS_VERSION}" PARENT_SCOPE)
      return()
    endif()
  endif()

  set(${out_var} "" PARENT_SCOPE)
endfunction()

function(emit_files dist_dir)
  # If gzip is requested, compress every asset into a parallel _gzip/ tree
  # the structure stays the same; for ex: /abc/def --> /_gzip/abc/def
  # embed.cpp will check for _gzip and will pick it up
  if(MGS_UI_GZIP AND EXISTS "${dist_dir}/index.html")
    find_program(GZIP_EXECUTABLE gzip)
    if(NOT GZIP_EXECUTABLE)
      message(WARNING "UI: MGS_UI_GZIP requested but gzip not found, embedding uncompressed")
    else()
      set(gzip_dir "${dist_dir}/_gzip")
      file(REMOVE_RECURSE "${gzip_dir}")
      file(GLOB_RECURSE all_files RELATIVE "${dist_dir}" "${dist_dir}/*")
      foreach(f ${all_files})
        get_filename_component(dst_dir "${gzip_dir}/${f}" DIRECTORY)
        file(MAKE_DIRECTORY "${dst_dir}")
        execute_process(
          COMMAND "${GZIP_EXECUTABLE}" -c "${dist_dir}/${f}"
          OUTPUT_FILE "${gzip_dir}/${f}"
          RESULT_VARIABLE gz_rc
        )
        if(NOT gz_rc EQUAL 0)
          message(FATAL_ERROR "UI: gzip failed for ${f}")
        endif()
      endforeach()
      message(STATUS "UI: gzip compression applied (${gzip_dir})")
    endif()
  endif()

  set(args "${UI_CPP}" "${UI_H}")
  if(EXISTS "${dist_dir}/index.html")
    list(APPEND args "${dist_dir}")
  endif()

  execute_process(
    COMMAND "${MGS_UI_EMBED}" ${args}
    RESULT_VARIABLE rc
  )
  if(NOT rc EQUAL 0)
    message(FATAL_ERROR "UI: mgs-ui-embed failed (${rc})")
  endif()
endfunction()

# ---------------------------------------------------------------------------
# 1. Without BUILD_UI: build the server without an embedded interface
# ---------------------------------------------------------------------------
if(NOT BUILD_UI)
  file(REMOVE_RECURSE "${DIST_DIR}")
  file(MAKE_DIRECTORY "${DIST_DIR}")
  emit_files("${DIST_DIR}")
  return()
endif()

# ---------------------------------------------------------------------------
# 2. With BUILD_UI: npm build
# ---------------------------------------------------------------------------
set(provisioned FALSE)

if(BUILD_UI)
  # Resolve the version from cmake/build-info.cmake if not explicitly set
  resolve_version(UI_VERSION)
  npm_build(NPM_OK)
  if(NPM_OK)
    set(provisioned TRUE)
  else()
    message(FATAL_ERROR "UI: local interface build failed; refusing to embed outdated assets")
  endif()
endif()

# ---------------------------------------------------------------------------
# 3. Fallback: warn about stale or missing assets, then emit whatever we have
# ---------------------------------------------------------------------------
if(NOT provisioned)
  if(EXISTS "${DIST_DIR}/index.html")
    message(WARNING "UI: provisioning failed; embedding stale assets from ${DIST_DIR}")
  else()
    message(WARNING "UI: no assets available - building without an embedded UI.")
  endif()
endif()

emit_files("${DIST_DIR}")
