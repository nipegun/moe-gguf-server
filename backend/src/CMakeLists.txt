mgs_add_compile_flags()

#
# libraries
#

# mgs-engine

# MoE models and the MTP heads/shared blocks they require.
set(MGS_MODELS_SOURCES
  models/deepseek41.cpp
  models/delta-net-base.cpp
  models/flint.cpp
  models/glm-dsa.cpp
  models/glm5-next.cpp
  models/kimi-k3.cpp
  models/mamba-base.cpp
  models/mimo2.cpp
  models/minimax-m3.cpp
  models/nemotron-h-moe.cpp
  models/qwen35.cpp
  models/qwen35moe.cpp
  models/qwen4exp.cpp
)

add_library(mgs-engine
      ../include/mgs.h
      mgs.cpp
      mgs-adapter.cpp
      mgs-arch.cpp
      mgs-batch.cpp
      mgs-context.cpp
      mgs-cparams.cpp
      mgs-expert-cache.cpp
      mgs-grammar.cpp
      mgs-graph.cpp
      mgs-hparams.cpp
      mgs-impl.cpp
      mgs-io.cpp
      mgs-kv-cache.cpp
      mgs-kv-cache-iswa.cpp
      mgs-kv-cache-dsa.cpp
      mgs-kv-cache-msa.cpp
      mgs-kv-cache-dsv41.cpp
      mgs-memory.cpp
      mgs-memory-hybrid.cpp
      mgs-memory-hybrid-idx.cpp
      mgs-memory-recurrent.cpp
      mgs-mmap.cpp
      mgs-model-loader.cpp
      mgs-model-saver.cpp
      mgs-model.cpp
      mgs-quant.cpp
      mgs-sampler.cpp
      mgs-vocab.cpp
      unicode-data.cpp
      unicode.cpp
      unicode.h
      ${MGS_MODELS_SOURCES}
      )

set_target_properties(mgs-engine PROPERTIES
  VERSION ${MGS_VERSION}
  SOVERSION 0
  MACHO_CURRENT_VERSION 0 # keep macOS linker from seeing oversized version number
)

target_include_directories(mgs-engine PRIVATE .)
target_include_directories(mgs-engine PUBLIC ../include)
target_compile_features   (mgs-engine PRIVATE cxx_std_17) # don't bump

target_link_libraries(mgs-engine PUBLIC ggml)

if (BUILD_SHARED_LIBS)
  set_target_properties(mgs-engine PROPERTIES POSITION_INDEPENDENT_CODE ON)
  target_compile_definitions(mgs-engine PRIVATE MGS_BUILD)
  target_compile_definitions(mgs-engine PUBLIC  MGS_SHARED)
endif()
