# mtmd

set(MTMD_VIDEO_HELP "enable video support in mtmd (requires ffmpeg binary in PATH)")

set(MTMD_VIDEO ON CACHE BOOL "${MTMD_VIDEO_HELP}")
# TODO: add MTMD_VIDEO_METHOD in the future to select between ffmpeg and other backends

if (MTMD_VIDEO AND NOT MGS_SUBPROCESS)
  message(STATUS "Disabling MTMD_VIDEO because MGS_SUBPROCESS is OFF")
  set(MTMD_VIDEO OFF CACHE BOOL "${MTMD_VIDEO_HELP}" FORCE)
endif()

find_package(Threads REQUIRED)

add_library(mtmd
      mtmd.cpp
      mtmd-audio.cpp
      mtmd-image.cpp
      mtmd.h
      mtmd-helper.cpp
      mtmd-helper.h
      clip.cpp
      clip.h
      clip-impl.h
      clip-model.h
      clip-graph.h
      models/models.h
      models/cogvlm.cpp
      models/conformer.cpp
      models/dotsocr.cpp
      models/exaone4_5.cpp
      models/glm4v.cpp
      models/granite-speech.cpp
      models/granite4-vision.cpp
      models/hunyuanvl.cpp
      models/internvl.cpp
      models/kimivl.cpp
      models/kimik25.cpp
      models/nemotron-v2-vl.cpp
      models/minicpmv.cpp
      models/paddleocr.cpp
      models/pixtral.cpp
      models/qwen2vl.cpp
      models/minimax-m3.cpp
      models/qwen3vl.cpp
      models/mimovl.cpp
      models/qwen3a.cpp
      models/mimo-audio.cpp
      models/step3vl.cpp
      models/siglip.cpp
      models/whisper-enc.cpp
      models/mobilenetv5.cpp
      models/youtuvl.cpp
      models/yasa2.cpp
      models/parakeet.cpp
      )

set_target_properties(mtmd PROPERTIES
  VERSION ${MGS_VERSION}
  SOVERSION 0
  MACHO_CURRENT_VERSION 0 # keep macOS linker from seeing oversized version number
)

target_link_libraries     (mtmd PUBLIC ggml mgs-engine)
target_link_libraries     (mtmd PRIVATE Threads::Threads)
target_include_directories(mtmd PUBLIC  .)
target_include_directories(mtmd PRIVATE ../..)
target_include_directories(mtmd PRIVATE ../../vendor)
target_compile_features   (mtmd PRIVATE cxx_std_17)

if (MTMD_VIDEO)
  target_compile_definitions(mtmd PRIVATE MTMD_VIDEO)
endif()

if (BUILD_SHARED_LIBS)
  set_target_properties     (mtmd PROPERTIES POSITION_INDEPENDENT_CODE ON)
  target_compile_definitions(mtmd PRIVATE MGS_BUILD)
  target_compile_definitions(mtmd PUBLIC  MGS_SHARED)
endif()

set(MTMD_PUBLIC_HEADERS
  ${CMAKE_CURRENT_SOURCE_DIR}/mtmd.h
  ${CMAKE_CURRENT_SOURCE_DIR}/mtmd-helper.h
  )

set_target_properties(mtmd
  PROPERTIES
  PUBLIC_HEADER "${MTMD_PUBLIC_HEADERS}")

install(TARGETS mtmd LIBRARY PUBLIC_HEADER)

if (NOT MSVC)
  # for stb_image.h and miniaudio.h
  target_compile_options(mtmd PRIVATE -Wno-cast-qual)
endif()

if (ANDROID)
  # miniaudio.h defines ma_android_sdk_version() without a prior prototype
  target_compile_options(mtmd PRIVATE -Wno-missing-prototypes)
endif()

if (TARGET BUILD_INFO)
  add_dependencies(mtmd        BUILD_INFO)
  add_dependencies(mtmd-helper BUILD_INFO)
endif()

# if mtmd is linked against mgs-common, we throw an error
if (TARGET mtmd)
  get_target_property(libs mtmd LINK_LIBRARIES)
  if (libs AND "mgs-common" IN_LIST libs)
    message(FATAL_ERROR "mtmd is designed to be a public library.\n"
              "It must not link against mgs-common")
  endif()
endif()
