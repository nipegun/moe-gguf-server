# Provision UI assets and generate ui.cpp/ui.h.
#
# With BUILD_UI=ON the interface is built with npm; with BUILD_UI=OFF the
# server is built without an embedded interface. The interface version is
# always MGS_VERSION, read from cmake/build-info.cmake.

cmake_minimum_required(VERSION 3.18)

# Parameters received with -D from frontend/CMakeLists.txt.
set(UI_SOURCE_DIR     "" CACHE STRING "UI source directory (to run npm build)")
set(UI_BINARY_DIR     "" CACHE STRING "UI binary directory (to store generated files)")
set(MGS_SOURCE_DIR    "" CACHE STRING "Backend source root (to read MGS_VERSION from cmake/build-info.cmake)")
set(BUILD_UI          "" CACHE STRING "Build UI via npm (ON/OFF)")
set(MGS_UI_EMBED      "" CACHE STRING "Path to mgs-ui-embed helper")
set(MGS_UI_GZIP       "" CACHE STRING "Apply gzip compress to assets to save bandwidth")

set(vDistDirectory  "${UI_BINARY_DIR}/dist")
set(vWorkDirectory  "${UI_BINARY_DIR}/ui-src")
set(vStampFile      "${UI_BINARY_DIR}/.ui-stamp")
set(vUiCpp          "${UI_BINARY_DIR}/ui.cpp")
set(vUiH            "${UI_BINARY_DIR}/ui.h")

function(fNpmBuildShouldSkip pOutputVariable)
  set(${pOutputVariable} FALSE PARENT_SCOPE)

  if(NOT EXISTS "${vDistDirectory}/index.html")
    return()
  endif()

  if(EXISTS "${vStampFile}")
    return()
  endif()

  # A version change must rewrite build.json even when no interface source changed
  if(NOT EXISTS "${vDistDirectory}/build.json")
    return()
  endif()
  file(READ "${vDistDirectory}/build.json" vBuildJson)
  if(NOT vBuildJson STREQUAL "{\"version\":\"${vUiVersion}\"}")
    return()
  endif()

  if(NOT EXISTS "${UI_SOURCE_DIR}/sources.cmake")
    return()
  endif()
  # sources.cmake defines UI_SOURCE_GLOBS and UI_SOURCE_FILES.
  include("${UI_SOURCE_DIR}/sources.cmake")

  set(lGlobs "")
  foreach(vGlob ${UI_SOURCE_GLOBS})
    list(APPEND lGlobs "${UI_SOURCE_DIR}/${vGlob}")
  endforeach()
  file(GLOB_RECURSE lSources ${lGlobs})
  foreach(vFile ${UI_SOURCE_FILES})
    list(APPEND lSources "${UI_SOURCE_DIR}/${vFile}")
  endforeach()

  file(TIMESTAMP "${vDistDirectory}/index.html" vOutputTimestamp)

  foreach(vSource ${lSources})
    if(NOT EXISTS "${vSource}")
      continue()
    endif()
    file(TIMESTAMP "${vSource}" vSourceTimestamp)
    if(vSourceTimestamp STRGREATER vOutputTimestamp)
      return()
    endif()
  endforeach()

  set(${pOutputVariable} TRUE PARENT_SCOPE)
endfunction()

function(fStageSources)
  if(EXISTS "${vWorkDirectory}")
    file(GLOB lStaged RELATIVE "${vWorkDirectory}" "${vWorkDirectory}/*")
    list(REMOVE_ITEM lStaged "node_modules")
    foreach(vEntry ${lStaged})
      file(REMOVE_RECURSE "${vWorkDirectory}/${vEntry}")
    endforeach()
  endif()

  file(COPY "${UI_SOURCE_DIR}/"
    DESTINATION "${vWorkDirectory}"
    NO_SOURCE_PERMISSIONS
    PATTERN "node_modules" EXCLUDE
  )
endfunction()

function(fNpmBuild pOutputVariable)
  set(${pOutputVariable} FALSE PARENT_SCOPE)

  if(NOT EXISTS "${UI_SOURCE_DIR}/package.json")
    message(STATUS "UI: ${UI_SOURCE_DIR}/package.json not found, skipping npm")
    return()
  endif()

  fNpmBuildShouldSkip(vSkip)
  if(vSkip)
    message(STATUS "UI: npm output up-to-date, skipping build")
    set(${pOutputVariable} TRUE PARENT_SCOPE)
    return()
  endif()

  if(CMAKE_HOST_WIN32)
    find_program(vNpmExecutable NAMES npm.cmd npm.bat npm)
  else()
    find_program(vNpmExecutable npm)
  endif()
  if(NOT vNpmExecutable)
    message(STATUS "UI: npm not found, skipping npm build")
    return()
  endif()

  fStageSources()

  # Use our own marker because npm can write its internal lock before an
  # interrupted installation has populated every package completely.
  set(vNpmMarker "${vWorkDirectory}/node_modules/.mgs-ui-install-stamp")
  set(vNeedInstall FALSE)
  if(NOT EXISTS "${vNpmMarker}")
    set(vNeedInstall TRUE)
  else()
    file(TIMESTAMP "${vWorkDirectory}/package-lock.json" vLockTimestamp)
    file(TIMESTAMP "${vNpmMarker}" vMarkerTimestamp)
    if(vLockTimestamp STRGREATER vMarkerTimestamp)
      set(vNeedInstall TRUE)
    endif()
  endif()

  if(vNeedInstall)
    message(STATUS "UI: running npm ci")
    execute_process(
      COMMAND ${vNpmExecutable} ci
      WORKING_DIRECTORY "${vWorkDirectory}"
      RESULT_VARIABLE vResult
      ERROR_VARIABLE  vError
    )
    if(NOT vResult EQUAL 0)
      message(STATUS "UI: npm ci failed (${vResult})")
      message(STATUS "  stderr: ${vError}")
      return()
    endif()
    file(WRITE "${vNpmMarker}" "installed\n")
  endif()

  file(MAKE_DIRECTORY "${vDistDirectory}")

  # MGS_UI_OUT_DIR and MGS_UI_VERSION are read by the vite plugins of the interface.
  message(STATUS "UI: running npm run build, output -> ${vDistDirectory}")
  execute_process(
    COMMAND ${CMAKE_COMMAND} -E env "MGS_UI_OUT_DIR=${vDistDirectory}" "MGS_UI_VERSION=${vUiVersion}"
        ${vNpmExecutable} run build
    WORKING_DIRECTORY "${vWorkDirectory}"
    RESULT_VARIABLE vResult
    ERROR_VARIABLE  vError
  )
  if(NOT vResult EQUAL 0)
    message(STATUS "UI: npm run build failed (${vResult})")
    message(STATUS "  stderr: ${vError}")
    return()
  endif()

  if(NOT EXISTS "${vDistDirectory}/index.html")
    message(STATUS "UI: npm build finished but assets missing in ${vDistDirectory}")
    return()
  endif()

  message(STATUS "UI: npm build succeeded")
  file(REMOVE "${vStampFile}")
  set(${pOutputVariable} TRUE PARENT_SCOPE)
endfunction()

function(fResolveVersion pOutputVariable)
  # MGS_VERSION in cmake/build-info.cmake is the only source of the version.
  if(NOT EXISTS "${MGS_SOURCE_DIR}/cmake/build-info.cmake")
    message(FATAL_ERROR "UI: ${MGS_SOURCE_DIR}/cmake/build-info.cmake not found; it defines MGS_VERSION")
  endif()
  include("${MGS_SOURCE_DIR}/cmake/build-info.cmake")
  if("${MGS_VERSION}" STREQUAL "" OR NOT MGS_VERSION MATCHES "^[A-Za-z0-9._-]+$")
    message(FATAL_ERROR "UI: invalid MGS_VERSION='${MGS_VERSION}' - must match ^[A-Za-z0-9._-]+$")
  endif()
  set(${pOutputVariable} "${MGS_VERSION}" PARENT_SCOPE)
endfunction()

function(fEmitFiles pDistDirectory)
  # If gzip is requested, compress every asset into a parallel _gzip/ tree
  # the structure stays the same; for ex: /abc/def --> /_gzip/abc/def
  # embed.cpp will check for _gzip and will pick it up
  if(MGS_UI_GZIP AND EXISTS "${pDistDirectory}/index.html")
    find_program(vGzipExecutable gzip)
    if(NOT vGzipExecutable)
      message(WARNING "UI: MGS_UI_GZIP requested but gzip not found, embedding uncompressed")
    else()
      set(vGzipDirectory "${pDistDirectory}/_gzip")
      file(REMOVE_RECURSE "${vGzipDirectory}")
      file(GLOB_RECURSE lAllFiles RELATIVE "${pDistDirectory}" "${pDistDirectory}/*")
      foreach(vFile ${lAllFiles})
        get_filename_component(vDestinationDirectory "${vGzipDirectory}/${vFile}" DIRECTORY)
        file(MAKE_DIRECTORY "${vDestinationDirectory}")
        execute_process(
          COMMAND "${vGzipExecutable}" -c "${pDistDirectory}/${vFile}"
          OUTPUT_FILE "${vGzipDirectory}/${vFile}"
          RESULT_VARIABLE vGzipResult
        )
        if(NOT vGzipResult EQUAL 0)
          message(FATAL_ERROR "UI: gzip failed for ${vFile}")
        endif()
      endforeach()
      message(STATUS "UI: gzip compression applied (${vGzipDirectory})")
    endif()
  endif()

  set(lEmbedArguments "${vUiCpp}" "${vUiH}")
  if(EXISTS "${pDistDirectory}/index.html")
    list(APPEND lEmbedArguments "${pDistDirectory}")
  endif()

  execute_process(
    COMMAND "${MGS_UI_EMBED}" ${lEmbedArguments}
    RESULT_VARIABLE vResult
  )
  if(NOT vResult EQUAL 0)
    message(FATAL_ERROR "UI: mgs-ui-embed failed (${vResult})")
  endif()
endfunction()

# ---------------------------------------------------------------------------
# 1. Without BUILD_UI: build the server without an embedded interface
# ---------------------------------------------------------------------------
if(NOT BUILD_UI)
  file(REMOVE_RECURSE "${vDistDirectory}")
  file(MAKE_DIRECTORY "${vDistDirectory}")
  fEmitFiles("${vDistDirectory}")
  return()
endif()

# ---------------------------------------------------------------------------
# 2. With BUILD_UI: npm build, versioned with MGS_VERSION
# ---------------------------------------------------------------------------
if(BUILD_UI)
  fResolveVersion(vUiVersion)
  fNpmBuild(vNpmOk)
  if(NOT vNpmOk)
    message(FATAL_ERROR "UI: local interface build failed; refusing to embed outdated assets")
  endif()
endif()

fEmitFiles("${vDistDirectory}")
