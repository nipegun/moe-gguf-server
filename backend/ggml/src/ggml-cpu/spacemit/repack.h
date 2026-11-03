#pragma once

#include "ggml-common.h"
#include "ggml.h"

#include <cstddef>
#include <cstdint>

namespace ggml::cpu::riscv64_spacemit {

template <typename BLOC_TYPE, int64_t INTER_SIZE, int64_t NB_COLS>
int repack(ggml_tensor *pT, const void *pData, size_t pDataSize);

} // namespace ggml::cpu::riscv64_spacemit
