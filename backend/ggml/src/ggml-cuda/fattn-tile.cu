#include "common.cuh"
#include "fattn-tile.cuh"

void ggml_cuda_flash_attn_ext_tile(ggml_backend_cuda_context &pCtx, ggml_tensor *pDst) {
  const ggml_tensor *vK = pDst->src[1];
  const ggml_tensor *vV = pDst->src[2];
  switch (vK->ne[0]) {
  case 40: {
    GGML_ASSERT(vV->ne[0] == vK->ne[0]);
    ggml_cuda_flash_attn_ext_tile_case<40, 40>(pCtx, pDst);
  } break;
  case 64: {
    GGML_ASSERT(vV->ne[0] == vK->ne[0]);
    ggml_cuda_flash_attn_ext_tile_case<64, 64>(pCtx, pDst);
  } break;
  case 72: {
    GGML_ASSERT(vV->ne[0] == vK->ne[0]);
    ggml_cuda_flash_attn_ext_tile_case<72, 72>(pCtx, pDst);
  } break;
  case 80: {
    GGML_ASSERT(vV->ne[0] == vK->ne[0]);
    ggml_cuda_flash_attn_ext_tile_case<80, 80>(pCtx, pDst);
  } break;
  case 96: {
    GGML_ASSERT(vV->ne[0] == vK->ne[0]);
    ggml_cuda_flash_attn_ext_tile_case<96, 96>(pCtx, pDst);
  } break;
  case 112: {
    GGML_ASSERT(vV->ne[0] == vK->ne[0]);
    ggml_cuda_flash_attn_ext_tile_case<112, 112>(pCtx, pDst);
  } break;
  case 128: {
    GGML_ASSERT(vV->ne[0] == vK->ne[0]);
    ggml_cuda_flash_attn_ext_tile_case<128, 128>(pCtx, pDst);
  } break;
  case 192: {
    GGML_ASSERT(vV->ne[0] == 128);
    ggml_cuda_flash_attn_ext_tile_case<192, 128>(pCtx, pDst);
  } break;
  case 256: {
    GGML_ASSERT(vV->ne[0] == vK->ne[0]);
    ggml_cuda_flash_attn_ext_tile_case<256, 256>(pCtx, pDst);
  } break;
  case 320: {
    GGML_ASSERT(vV->ne[0] == 256);
    ggml_cuda_flash_attn_ext_tile_case<320, 256>(pCtx, pDst);
  } break;
  case 512: {
    GGML_ASSERT(vV->ne[0] == vK->ne[0]);
    ggml_cuda_flash_attn_ext_tile_case<512, 512>(pCtx, pDst);
  } break;
  case 576: {
    GGML_ASSERT(vV->ne[0] == 512);
    ggml_cuda_flash_attn_ext_tile_case<576, 512>(pCtx, pDst);
  } break;
  default: {
    GGML_ABORT("Unsupported head size");
  } break;
  }
}
