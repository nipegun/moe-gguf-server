mgs_add_compile_flags()

#
# libraries
#

# mgs-engine

# MoE models and the MTP heads/shared blocks they require.
set(MGS_MODELS_SOURCES
  models/afmoe.cpp
  models/arctic.cpp
  models/bailingmoe.cpp
  models/bailingmoe2.cpp
  models/cohere2moe.cpp
  models/dbrx.cpp
  models/deepseek41.cpp
  models/delta-net-base.cpp
  models/dflash.cpp
  models/dots1.cpp
  models/ernie4-5-moe.cpp
  models/exaone-moe.cpp
  models/glm-dsa.cpp
  models/glm4-moe.cpp
  models/glm5-next.cpp
  models/granite-hybrid.cpp
  models/granite-moe.cpp
  models/grok.cpp
  models/grovemoe.cpp
  models/hunyuan-moe.cpp
  models/hy-v3.cpp
  models/jamba.cpp
  models/kimi-k3.cpp
  models/kimi-linear.cpp
  models/laguna.cpp
  models/lfm2moe.cpp
  models/llada-moe.cpp
  models/mamba-base.cpp
  models/mellum.cpp
  models/mimo2.cpp
  models/minicpm.cpp
  models/minimax-m2.cpp
  models/minimax-m3.cpp
  models/mistral3.cpp
  models/nemotron-h-moe.cpp
  models/nomic-bert-moe.cpp
  models/olmoe.cpp
  models/openai-moe.cpp
  models/phimoe.cpp
  models/qwen2moe.cpp
  models/qwen35.cpp
  models/qwen35moe.cpp
  models/qwen4exp.cpp
  models/qwen3moe.cpp
  models/qwen3next.cpp
  models/qwen3vlmoe.cpp
  models/rnd1.cpp
  models/smallthinker.cpp
  models/step35.cpp
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
      mgs-memory-hybrid-iswa.cpp
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
