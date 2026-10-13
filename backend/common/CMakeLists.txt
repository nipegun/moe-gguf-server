find_package(Threads REQUIRED)

mgs_add_compile_flags()

#
# mgs-common-base
#

# Build info header

set(TEMPLATE_FILE "${CMAKE_CURRENT_SOURCE_DIR}/build-info.cpp.in")
set(OUTPUT_FILE   "${CMAKE_CURRENT_BINARY_DIR}/build-info.cpp")

configure_file(${TEMPLATE_FILE} ${OUTPUT_FILE})

set(TARGET mgs-common-base)
add_library(${TARGET} STATIC ${OUTPUT_FILE})

target_include_directories(${TARGET} PUBLIC .)

if (BUILD_SHARED_LIBS)
  set_target_properties(${TARGET} PROPERTIES POSITION_INDEPENDENT_CODE ON)
endif()

#
# mgs-common
#

set(TARGET mgs-common)

add_library(${TARGET}
  arg.cpp
  arg.h
  base64.hpp
  chat-auto-parser-generator.cpp
  chat-auto-parser-helpers.cpp
  chat-auto-parser.h
  chat-diff-analyzer.cpp
  chat-peg-parser.cpp
  chat-peg-parser.h
  chat.cpp
  chat.h
  common.cpp
  common.h
  console.cpp
  console.h
  debug.cpp
  debug.h
  download.cpp
  download.h
  fit.cpp
  fit.h
  hf-cache.cpp
  hf-cache.h
  http.h
  imatrix-loader.cpp
  imatrix-loader.h
  json-schema-to-grammar.cpp
  llguidance.cpp
  log.cpp
  log.h
  ngram-cache.cpp
  ngram-cache.h
  ngram-map.cpp
  ngram-map.h
  ngram-mod.cpp
  ngram-mod.h
  peg-parser.cpp
  peg-parser.h
  preset.cpp
  preset.h
  reasoning-budget.cpp
  reasoning-budget.h
  sampling.cpp
  sampling.h
  speculative.cpp
  speculative.h
  subproc.cpp
  subproc.h
  trie.cpp
  trie.h
  unicode.cpp
  unicode.h
  jinja/lexer.cpp
  jinja/lexer.h
  jinja/parser.cpp
  jinja/parser.h
  jinja/runtime.cpp
  jinja/runtime.h
  jinja/value.cpp
  jinja/value.h
  jinja/string.cpp
  jinja/string.h
  jinja/caps.cpp
  jinja/caps.h
  )

set_target_properties(${TARGET} PROPERTIES
  VERSION ${MGS_VERSION}
  SOVERSION 0
  MACHO_CURRENT_VERSION 0 # keep macOS linker from seeing oversized version number
)

target_include_directories(${TARGET} PUBLIC . ../vendor)
target_compile_features   (${TARGET} PUBLIC cxx_std_17)

if (MGS_SUBPROCESS)
  target_compile_definitions(${TARGET} PUBLIC MGS_SUBPROCESS)
endif()

if (BUILD_SHARED_LIBS)
  set_target_properties(${TARGET} PROPERTIES POSITION_INDEPENDENT_CODE ON)

  # TODO: make fine-grained exports in the future
  set_target_properties(${TARGET} PROPERTIES WINDOWS_EXPORT_ALL_SYMBOLS ON)
endif()

target_link_libraries(${TARGET} PUBLIC  mgs-common-base)
target_link_libraries(${TARGET} PRIVATE cpp-httplib)

if (MGS_LLGUIDANCE)
  include(ExternalProject)
  set(LLGUIDANCE_SRC ${CMAKE_BINARY_DIR}/llguidance/source)
  set(LLGUIDANCE_PATH ${LLGUIDANCE_SRC}/target/release)
  set(LLGUIDANCE_LIB_NAME "${CMAKE_STATIC_LIBRARY_PREFIX}llguidance${CMAKE_STATIC_LIBRARY_SUFFIX}")

  ExternalProject_Add(llguidance_ext
    GIT_REPOSITORY https://github.com/guidance-ai/llguidance
    # v1.0.1:
    GIT_TAG d795912fedc7d393de740177ea9ea761e7905774
    PREFIX ${CMAKE_BINARY_DIR}/llguidance
    SOURCE_DIR ${LLGUIDANCE_SRC}
    BUILD_IN_SOURCE TRUE
    CONFIGURE_COMMAND ""
    BUILD_COMMAND cargo build --release --package llguidance
    INSTALL_COMMAND ""
    BUILD_BYPRODUCTS ${LLGUIDANCE_PATH}/${LLGUIDANCE_LIB_NAME} ${LLGUIDANCE_PATH}/llguidance.h
    UPDATE_COMMAND ""
  )
  target_compile_definitions(${TARGET} PUBLIC MGS_USE_LLGUIDANCE)

  add_library(llguidance STATIC IMPORTED)
  set_target_properties(llguidance PROPERTIES IMPORTED_LOCATION ${LLGUIDANCE_PATH}/${LLGUIDANCE_LIB_NAME})
  add_dependencies(llguidance llguidance_ext)

  target_include_directories(${TARGET} PRIVATE ${LLGUIDANCE_PATH})
  target_link_libraries(${TARGET} PRIVATE llguidance)
  if (WIN32)
    target_link_libraries(${TARGET} PRIVATE ws2_32 userenv ntdll bcrypt)
  endif()
endif()

target_link_libraries(${TARGET} PUBLIC mgs-engine Threads::Threads)
