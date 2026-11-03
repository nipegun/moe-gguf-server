#include "ggml-alloc.h"
#include "ggml-backend-impl.h"
#include "ggml.h"
#include "ggml-impl.h"

#include <assert.h>
#include <limits.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define MAX(a, b) ((a) > (b) ? (a) : (b))
#define MAX_FREE_BLOCKS 256

// #define GGML_ALLOCATOR_DEBUG

// #define AT_PRINTF(...) GGML_LOG_DEBUG(__VA_ARGS__)
#define AT_PRINTF(...)

// ops that return true for this function must not use restrict pointers for their backend implementations
bool ggml_op_can_inplace(enum ggml_op pOp) {
  switch (pOp) {
  case GGML_OP_FILL:
  case GGML_OP_SCALE:
  case GGML_OP_DIAG_MASK_ZERO:
  case GGML_OP_DIAG_MASK_INF:
  case GGML_OP_ADD:
  case GGML_OP_ADD_ID:
  case GGML_OP_ADD1:
  case GGML_OP_SUB:
  case GGML_OP_MUL:
  case GGML_OP_DIV:
  case GGML_OP_SQR:
  case GGML_OP_SQRT:
  case GGML_OP_LOG:
  case GGML_OP_UNARY:
  case GGML_OP_ROPE:
  case GGML_OP_ROPE_BACK:
  case GGML_OP_SILU_BACK:
  case GGML_OP_RMS_NORM:
  case GGML_OP_RMS_NORM_BACK:
  case GGML_OP_CLAMP:
  case GGML_OP_SOFT_MAX:
  case GGML_OP_SOFT_MAX_BACK:
    return true;

  default:
    return false;
  }
}

static size_t fAlignedOffset(const void *pBuffer, size_t pOffset, size_t pAlignment) {
  assert(pAlignment && !(pAlignment & (pAlignment - 1))); // power of 2
  size_t vAlign = (pAlignment - (((uintptr_t)pBuffer + pOffset) % pAlignment)) % pAlignment;
  return pOffset + vAlign;
}

// tallocr

struct ggml_tallocr ggml_tallocr_new(ggml_backend_buffer_t pBuffer) {
  void *vBase = ggml_backend_buffer_get_base(pBuffer);
  size_t vAlign = ggml_backend_buffer_get_alignment(pBuffer);

  assert(vAlign && !(vAlign & (vAlign - 1))); // power of 2

  struct ggml_tallocr vTalloc = (struct ggml_tallocr){
      /*.buffer    = */ pBuffer,
      /*.base      = */ vBase,
      /*.alignment = */ vAlign,
      /*.offset    = */ fAlignedOffset(vBase, 0, vAlign),
  };
  return vTalloc;
}

enum ggml_status ggml_tallocr_alloc(struct ggml_tallocr *pTalloc, struct ggml_tensor *pTensor) {
  size_t vSize = ggml_backend_buffer_get_alloc_size(pTalloc->buffer, pTensor);
  vSize = GGML_PAD(vSize, pTalloc->alignment);

  if (pTalloc->offset + vSize > ggml_backend_buffer_get_size(pTalloc->buffer)) {
    GGML_LOG_ERROR("%s: not enough space in the buffer to allocate %s (needed %zu, available %zu)\n", __func__,
                   pTensor->name, vSize, ggml_backend_buffer_get_size(pTalloc->buffer) - pTalloc->offset);
    GGML_ABORT("not enough space in the buffer");
  }

  void *vAddr = (char *)ggml_backend_buffer_get_base(pTalloc->buffer) + pTalloc->offset;
  pTalloc->offset += vSize;

  assert(((uintptr_t)vAddr % pTalloc->alignment) == 0);

  return ggml_backend_tensor_alloc(pTalloc->buffer, pTensor, vAddr);
}

// dynamic tensor allocator

#define GGML_VBUFFER_MAX_CHUNKS 16

// relative memory address within an allocation that can be split into multiple buffers (chunks)
struct buffer_address {
  int chunk;     // index of a backend buffer
  size_t offset; // local memory offset within the buffer
};

static const struct buffer_address GGML_BUFFER_ADDRESS_INVALID = {-1, SIZE_MAX};

static bool fGgmlBufferAddressLess(struct buffer_address pA, struct buffer_address pB) {
  return pA.chunk != pB.chunk ? pA.chunk < pB.chunk : pA.offset < pB.offset;
}

struct free_block {
  size_t offset;
  size_t size;
};

struct tallocr_chunk {
  struct free_block aFreeBlocks[MAX_FREE_BLOCKS];
  int vNFreeBlocks;
  size_t vMaxSize;
};

struct ggml_dyn_tallocr {
  size_t vAlignment;
  size_t vMaxChunkSize;
  struct tallocr_chunk *chunks[GGML_VBUFFER_MAX_CHUNKS];
  int vNChunks;

#ifdef GGML_ALLOCATOR_DEBUG
  struct {
    const struct ggml_tensor *tensor;
    struct buffer_address addr;
  } allocated_tensors[1024];
#endif
};

static void fGgmlDynTallocrInsertBlock(struct tallocr_chunk *pChunk, size_t pOffset, size_t pSize) {
  GGML_ASSERT(pChunk->vNFreeBlocks < MAX_FREE_BLOCKS && "out of free blocks");
  // insert the new block in the correct position to keep the array sorted by address (to make merging blocks faster)
  int vInsertPos = 0;
  while (vInsertPos < pChunk->vNFreeBlocks && pChunk->aFreeBlocks[vInsertPos].offset < pOffset) {
    vInsertPos++;
  }
  // shift all blocks from insert_pos onward to make room for the new block
  for (int vI = pChunk->vNFreeBlocks; vI > vInsertPos; vI--) {
    pChunk->aFreeBlocks[vI] = pChunk->aFreeBlocks[vI - 1];
  }
  // insert the new block
  pChunk->aFreeBlocks[vInsertPos].offset = pOffset;
  pChunk->aFreeBlocks[vInsertPos].size = pSize;
  pChunk->vNFreeBlocks++;
}

static void fGgmlDynTallocrRemoveBlock(struct tallocr_chunk *pChunk, int pIdx) {
  // shift all elements after idx by 1 to the left, overwriting the element at idx
  for (int vI = pIdx; vI < pChunk->vNFreeBlocks - 1; vI++) {
    pChunk->aFreeBlocks[vI] = pChunk->aFreeBlocks[vI + 1];
  }
  pChunk->vNFreeBlocks--;
}

static int fGgmlDynTallocrNewChunk(struct ggml_dyn_tallocr *pAlloc, size_t pMinSize) {
  if (pAlloc->vNChunks >= GGML_VBUFFER_MAX_CHUNKS) {
    return -1;
  }
  struct tallocr_chunk *vChunk = calloc(1, sizeof(struct tallocr_chunk));
  vChunk->vNFreeBlocks = 1;
  vChunk->aFreeBlocks[0].offset = 0;
  // available space in a chunk is limited to max_chunk_size, but can be higher if:
  // 1. a single tensor exceeds the maximum, and cannot fit any other way
  // 2. we are running out of chunks
  // backends will either manage to allocate the larger size, or report an error.
  vChunk->aFreeBlocks[0].size = MAX(pMinSize, pAlloc->vMaxChunkSize);
  if (pAlloc->vNChunks == GGML_VBUFFER_MAX_CHUNKS - 1) {
    vChunk->aFreeBlocks[0].size = SIZE_MAX / 2;
  }
  pAlloc->chunks[pAlloc->vNChunks] = vChunk;
  pAlloc->vNChunks++;
  return pAlloc->vNChunks - 1;
}

#ifdef GGML_ALLOCATOR_DEBUG
static void add_allocated_tensor(struct ggml_dyn_tallocr *alloc, struct buffer_address addr,
                                 const struct ggml_tensor *tensor) {
  for (int i = 0; i < 1024; i++) {
    if (alloc->allocated_tensors[i].tensor == NULL) {
      alloc->allocated_tensors[i].tensor = tensor;
      alloc->allocated_tensors[i].addr = addr;
      return;
    }
  }
  GGML_ABORT("out of allocated_tensors");
}
static void remove_allocated_tensor(struct ggml_dyn_tallocr *alloc, struct buffer_address addr,
                                    const struct ggml_tensor *tensor) {
  for (int i = 0; i < 1024; i++) {
    if (alloc->allocated_tensors[i].addr.chunk == addr.chunk &&
        alloc->allocated_tensors[i].addr.offset == addr.offset) {
      alloc->allocated_tensors[i].tensor = NULL;
      return;
    }
  }
  GGML_ABORT("tried to free tensor %s not found\n", tensor->name);
}
#endif

static struct buffer_address fGgmlDynTallocrAlloc(struct ggml_dyn_tallocr *pAlloc, size_t size,
                                                  const struct ggml_tensor *pTensor) {
  size = fAlignedOffset(NULL, size, pAlloc->vAlignment);

  AT_PRINTF("%s: allocating %s (%zu bytes) - ", __func__, pTensor->name, size);

  int vBestFitChunk = -1;
  int vBestFitBlock = -1;
  size_t vMaxAvail = 0;

  // find the best fitting free block besides the last block, within any chunk
  for (int vC = 0; vC < pAlloc->vNChunks; ++vC) {
    struct tallocr_chunk *chunk = pAlloc->chunks[vC];
    size_t vBestFitSize = SIZE_MAX;
    for (int vI = 0; vI < chunk->vNFreeBlocks - 1; vI++) {
      struct free_block *vBlock = &chunk->aFreeBlocks[vI];
      vMaxAvail = MAX(vMaxAvail, vBlock->size);
      if (vBlock->size >= size && vBlock->size <= vBestFitSize) {
        vBestFitChunk = vC;
        vBestFitBlock = vI;
        vBestFitSize = vBlock->size;
      }
    }
  }

  if (vBestFitBlock == -1) {
    // no suitable block found, try the last block (this may grow a chunks size)
    int64_t vBestReuse = INT64_MIN;
    for (int vC = 0; vC < pAlloc->vNChunks; ++vC) {
      struct tallocr_chunk *chunk = pAlloc->chunks[vC];
      if (chunk->vNFreeBlocks > 0) {
        struct free_block *vBlock = &chunk->aFreeBlocks[chunk->vNFreeBlocks - 1];
        vMaxAvail = MAX(vMaxAvail, vBlock->size);
        int64_t vReuseFactor = chunk->vMaxSize - vBlock->offset - size;
        // reuse_factor < 0 : amount of extra memory that needs to be allocated
        // reuse_factor = 0 : allocated free space exactly matches tensor size
        // reuse_factor > 0 : superfluous memory that will remain unused
        bool vBetterReuse = vBestReuse < 0 && vReuseFactor > vBestReuse;
        bool vBetterFit = vReuseFactor >= 0 && vReuseFactor < vBestReuse;
        if (vBlock->size >= size && (vBetterReuse || vBetterFit)) {
          vBestFitChunk = vC;
          vBestFitBlock = chunk->vNFreeBlocks - 1;
          vBestReuse = vReuseFactor;
        }
      }
    }
  }

  if (vBestFitBlock == -1) {
    // none of the existing chunks have enough space left
    vBestFitChunk = fGgmlDynTallocrNewChunk(pAlloc, size);
    vBestFitBlock = 0;
  }
  if (vBestFitChunk == -1) {
    // since the last chunk always has virtually endless memory, this should never happen
    GGML_LOG_ERROR("%s: not enough space in the buffer to allocate %zu bytes, largest block available %zu bytes\n",
                   __func__, size, vMaxAvail);
    GGML_ABORT("graph allocation: failed to reserve memory");
  }

  struct tallocr_chunk *chunk = pAlloc->chunks[vBestFitChunk];
  struct free_block *vBlock = &chunk->aFreeBlocks[vBestFitBlock];
  struct buffer_address addr = {.chunk = vBestFitChunk, .offset = vBlock->offset};
  vBlock->offset += size;
  vBlock->size -= size;
  if (vBlock->size == 0) {
    // remove block if empty
    fGgmlDynTallocrRemoveBlock(chunk, vBestFitBlock);
  }

  AT_PRINTF("block %d, offset %zu, chunk %d\n", vBestFitBlock, addr.offset, addr.chunk);

#ifdef GGML_ALLOCATOR_DEBUG
  add_allocated_tensor(pAlloc, addr, pTensor);
  size_t cur_max = addr.offset + size;
  if (cur_max > chunk->vMaxSize) {
    // sort allocated_tensors by chunk/offset
    for (int vI = 0; vI < 1024; vI++) {
      for (int j = vI + 1; j < 1024; j++) {
        if (fGgmlBufferAddressLess(pAlloc->allocated_tensors[j].addr, pAlloc->allocated_tensors[vI].addr)) {
          const struct ggml_tensor *tmp_tensor = pAlloc->allocated_tensors[vI].tensor;
          struct buffer_address tmp_addr = pAlloc->allocated_tensors[vI].addr;
          pAlloc->allocated_tensors[vI].tensor = pAlloc->allocated_tensors[j].tensor;
          pAlloc->allocated_tensors[vI].addr = pAlloc->allocated_tensors[j].addr;
          pAlloc->allocated_tensors[j].tensor = tmp_tensor;
          pAlloc->allocated_tensors[j].addr = tmp_addr;
        }
      }
    }
    GGML_LOG_DEBUG("max_size[%d] = %.2f MB: tensors: ", addr.chunk, cur_max / 1024.0 / 1024.0);
    for (int vI = 0; vI < 1024; vI++) {
      if (pAlloc->allocated_tensors[vI].tensor) {
        GGML_LOG_DEBUG("%s [%d: %zx-%zx] (%.2f MB) ", pAlloc->allocated_tensors[vI].tensor->name,
                       pAlloc->allocated_tensors[vI].addr.chunk, pAlloc->allocated_tensors[vI].addr.offset,
                       pAlloc->allocated_tensors[vI].addr.offset + ggml_nbytes(pAlloc->allocated_tensors[vI].tensor),
                       ggml_nbytes(pAlloc->allocated_tensors[vI].tensor) / 1024.0 / 1024.0);
      }
    }
    GGML_LOG_DEBUG("\n");
  }
#endif

  chunk->vMaxSize = MAX(chunk->vMaxSize, addr.offset + size);

  return addr;

  GGML_UNUSED(pTensor);
}

// this is a very naive implementation, but for our case the number of free blocks should be very small
static void fGgmlDynTallocrFreeBytes(struct ggml_dyn_tallocr *pAlloc, struct buffer_address pAddr, size_t pSize) {
  pSize = fAlignedOffset(NULL, pSize, pAlloc->vAlignment);

  struct tallocr_chunk *vChunk = pAlloc->chunks[pAddr.chunk];

  // see if we can merge with an existing block
  for (int vI = 0; vI < vChunk->vNFreeBlocks; vI++) {
    struct free_block *vBlock = &vChunk->aFreeBlocks[vI];
    // check if ptr is at the end of the block
    if (vBlock->offset + vBlock->size == pAddr.offset) {
      vBlock->size += pSize;
      // check if we can merge with the next block
      if (vI < vChunk->vNFreeBlocks - 1) {
        struct free_block *vNext = &vChunk->aFreeBlocks[vI + 1];
        if (vBlock->offset + vBlock->size == vNext->offset) {
          vBlock->size += vNext->size;
          fGgmlDynTallocrRemoveBlock(vChunk, vI + 1);
        }
      }
      return;
    }
    // check if ptr is at the beginning of the block
    if (pAddr.offset + pSize == vBlock->offset) {
      vBlock->offset = pAddr.offset;
      vBlock->size += pSize;
      // check if we can merge with the previous block
      if (vI > 0) {
        struct free_block *vPrev = &vChunk->aFreeBlocks[vI - 1];
        if (vPrev->offset + vPrev->size == vBlock->offset) {
          vPrev->size += vBlock->size;
          fGgmlDynTallocrRemoveBlock(vChunk, vI);
        }
      }
      return;
    }
  }
  // otherwise, add a new block
  fGgmlDynTallocrInsertBlock(vChunk, pAddr.offset, pSize);
}

static void fGgmlDynTallocrReset(struct ggml_dyn_tallocr *pAlloc) {
  for (int vI = 0; vI < GGML_VBUFFER_MAX_CHUNKS; vI++) {
    free(pAlloc->chunks[vI]);
    pAlloc->chunks[vI] = NULL;
  }
  pAlloc->vNChunks = 0;

#ifdef GGML_ALLOCATOR_DEBUG
  for (int vI = 0; vI < 1024; vI++) {
    pAlloc->allocated_tensors[vI].tensor = NULL;
  }
#endif
}

static struct ggml_dyn_tallocr *fGgmlDynTallocrNew(size_t pAlignment, size_t pMaxBufferSize) {
  struct ggml_dyn_tallocr *vAlloc = (struct ggml_dyn_tallocr *)malloc(sizeof(struct ggml_dyn_tallocr));

  *vAlloc = (struct ggml_dyn_tallocr){
      /*.alignment      = */ pAlignment,
      /*.max_chunk_size = */ MIN(pMaxBufferSize, SIZE_MAX / 2), // clamp to avoid overflows
      /*.chunks         = */ {NULL},
      /*.n_chunks       = */ 0,
#ifdef GGML_ALLOCATOR_DEBUG
      /*.allocated_tensors = */ {{0}},
#endif
  };

  fGgmlDynTallocrReset(vAlloc);

  return vAlloc;
}

static void fGgmlDynTallocrFree(struct ggml_dyn_tallocr *pAlloc) {
  for (int vI = 0; vI < pAlloc->vNChunks; ++vI) {
    free(pAlloc->chunks[vI]);
  }
  free(pAlloc);
}

static size_t fGgmlDynTallocrMaxSize(struct ggml_dyn_tallocr *pAlloc, int pChunk) {
  return pChunk < pAlloc->vNChunks ? pAlloc->chunks[pChunk]->vMaxSize : 0;
}

// virtual buffer with contiguous memory range, split into multiple backend buffers (chunks)

struct vbuffer {
  ggml_backend_buffer_t chunks[GGML_VBUFFER_MAX_CHUNKS];
};

static void fGgmlVbufferFree(struct vbuffer *pBuf) {
  if (pBuf == NULL) {
    return;
  }
  for (int vI = 0; vI < GGML_VBUFFER_MAX_CHUNKS; ++vI) {
    ggml_backend_buffer_free(pBuf->chunks[vI]);
  }
  free(pBuf);
}

static size_t fGgmlVbufferChunkSize(struct vbuffer *pBuf, int pChunk) {
  return pBuf->chunks[pChunk] ? ggml_backend_buffer_get_size(pBuf->chunks[pChunk]) : 0;
}

static size_t fGgmlVbufferSize(struct vbuffer *pBuf) {
  size_t vSize = 0;
  for (int vI = 0; vI < GGML_VBUFFER_MAX_CHUNKS && pBuf->chunks[vI]; ++vI) {
    vSize += ggml_backend_buffer_get_size(pBuf->chunks[vI]);
  }
  return vSize;
}

static struct vbuffer *fGgmlVbufferAlloc(ggml_backend_buffer_type_t pBuft, const struct ggml_dyn_tallocr *pTalloc,
                                         enum ggml_backend_buffer_usage pUsage) {
  struct vbuffer *vBuf = (struct vbuffer *)calloc(1, sizeof(struct vbuffer));
  if (vBuf == NULL) {
    return NULL;
  }

  for (int vN = 0; vN < pTalloc->vNChunks; vN++) {
    size_t vChunkSize = pTalloc->chunks[vN]->vMaxSize;
    vBuf->chunks[vN] = ggml_backend_buft_alloc_buffer(pBuft, vChunkSize);
    if (vBuf->chunks[vN] == NULL) {
      fGgmlVbufferFree(vBuf);
      return NULL;
    }
    ggml_backend_buffer_set_usage(vBuf->chunks[vN], pUsage);
  }
  return vBuf;
}

static void fGgmlVbufferTensorAlloc(struct vbuffer *pBuf, struct ggml_tensor *pTensor, struct buffer_address pBufAddr) {
  void *vBase = ggml_backend_buffer_get_base(pBuf->chunks[pBufAddr.chunk]);
  void *vAddr = (char *)vBase + pBufAddr.offset;
  ggml_backend_tensor_alloc(pBuf->chunks[pBufAddr.chunk], pTensor, vAddr);
}

static void fGgmlVbufferReset(struct vbuffer *pBuf) {
  for (int vI = 0; vI < GGML_VBUFFER_MAX_CHUNKS && pBuf->chunks[vI]; ++vI) {
    ggml_backend_buffer_reset(pBuf->chunks[vI]);
  }
}

/////////////////////////////////////

// graph allocator

struct hash_node {
  int vNChildren;
  int vNViews;
  int vBufferId;
  struct buffer_address addr;
  bool vAllocated;
};

struct tensor_alloc {
  int vBufferId;
  struct buffer_address addr;
  size_t vSizeMax; // 0 = pre-allocated, unused, or view
};

struct leaf_alloc {
  struct tensor_alloc vLeaf;
};

struct node_alloc {
  struct tensor_alloc vDst;
  struct tensor_alloc vSrc[GGML_MAX_SRC];
};

struct ggml_gallocr {
  ggml_backend_buffer_type_t *vBufts;    // [n_buffers]
  struct vbuffer **vBuffers;             // [n_buffers]
  struct ggml_dyn_tallocr **vBufTallocs; // [n_buffers]
  int vNBuffers;

  struct ggml_hash_set vHashSet;
  struct hash_node *vHashValues; // [hash_set.size]

  struct node_alloc *vNodeAllocs; // [n_nodes]
  int vNNodes;

  struct leaf_alloc *vLeafAllocs; // [n_leafs]
  int vNLeafs;
};

ggml_gallocr_t ggml_gallocr_new_n(ggml_backend_buffer_type_t *pBufts, int pNBufs) {
  ggml_gallocr_t vGalloc = (ggml_gallocr_t)calloc(1, sizeof(struct ggml_gallocr));
  GGML_ASSERT(vGalloc != NULL);

  vGalloc->vBufts = calloc(pNBufs, sizeof(ggml_backend_buffer_type_t));
  GGML_ASSERT(vGalloc->vBufts != NULL);

  vGalloc->vBuffers = calloc(pNBufs, sizeof(struct vbuffer *));
  GGML_ASSERT(vGalloc->vBuffers != NULL);

  vGalloc->vBufTallocs = calloc(pNBufs, sizeof(struct ggml_dyn_tallocr *));
  GGML_ASSERT(vGalloc->vBufTallocs != NULL);

  for (int vI = 0; vI < pNBufs; vI++) {
    vGalloc->vBufts[vI] = pBufts[vI];
    vGalloc->vBuffers[vI] = NULL;

    // check if the same buffer type is used multiple times and reuse the same allocator
    for (int vJ = 0; vJ < vI; vJ++) {
      if (pBufts[vI] == pBufts[vJ]) {
        vGalloc->vBufTallocs[vI] = vGalloc->vBufTallocs[vJ];
        break;
      }
    }

    if (vGalloc->vBufTallocs[vI] == NULL) {
      size_t vAlignment = ggml_backend_buft_get_alignment(pBufts[vI]);
      size_t vMaxSize = ggml_backend_buft_get_max_size(pBufts[vI]);
      vGalloc->vBufTallocs[vI] = fGgmlDynTallocrNew(vAlignment, vMaxSize);
    }
  }
  vGalloc->vNBuffers = pNBufs;

  return vGalloc;
}

ggml_gallocr_t ggml_gallocr_new(ggml_backend_buffer_type_t pBuft) { return ggml_gallocr_new_n(&pBuft, 1); }

void ggml_gallocr_free(ggml_gallocr_t pGalloc) {
  if (pGalloc == NULL) {
    return;
  }

  for (int vI = 0; vI < pGalloc->vNBuffers; vI++) {
    if (pGalloc->vBuffers != NULL) {
      // skip if already freed
      bool vFreed = false;
      for (int vJ = 0; vJ < vI; vJ++) {
        if (pGalloc->vBuffers[vJ] == pGalloc->vBuffers[vI]) {
          vFreed = true;
          break;
        }
      }
      if (!vFreed) {
        fGgmlVbufferFree(pGalloc->vBuffers[vI]);
      }
    }
    if (pGalloc->vBufTallocs != NULL) {
      // skip if already freed
      bool vFreed = false;
      for (int vJ = 0; vJ < vI; vJ++) {
        if (pGalloc->vBufTallocs[vJ] == pGalloc->vBufTallocs[vI]) {
          vFreed = true;
          break;
        }
      }
      if (!vFreed) {
        fGgmlDynTallocrFree(pGalloc->vBufTallocs[vI]);
      }
    }
  }

  ggml_hash_set_free(&pGalloc->vHashSet);
  free(pGalloc->vHashValues);
  free(pGalloc->vBufts);
  free(pGalloc->vBuffers);
  free(pGalloc->vBufTallocs);
  free(pGalloc->vNodeAllocs);
  free(pGalloc->vLeafAllocs);
  free(pGalloc);
}

typedef struct ggml_gallocr *ggml_gallocr_t;

static struct hash_node *fGgmlGallocrHashGet(ggml_gallocr_t pGalloc, struct ggml_tensor *pT) {
  size_t vI = ggml_hash_find_or_insert(&pGalloc->vHashSet, pT);
  return &pGalloc->vHashValues[vI];
}

static bool fGgmlGallocrIsOwn(ggml_gallocr_t pGalloc, struct ggml_tensor *pT) {
  return fGgmlGallocrHashGet(pGalloc, pT)->vAllocated;
}

static bool fGgmlGallocrIsAllocated(ggml_gallocr_t pGalloc, struct ggml_tensor *pT) {
  return pT->data != NULL                   // tensor data already set externally
         || pT->buffer                      // tensor on external buffer (but not yet allocated)
         || fGgmlGallocrIsOwn(pGalloc, pT); // tensor will be allocated by galloc
}

// free the extra space at the end if the new tensor is smaller
static void fGgmlGallocrFreeExtraSpace(ggml_gallocr_t pGalloc, struct ggml_tensor *pNode, struct ggml_tensor *pParent) {
  struct hash_node *vHn = fGgmlGallocrHashGet(pGalloc, pNode);
  struct hash_node *vPHn = fGgmlGallocrHashGet(pGalloc, pParent);

  size_t vParentSize = ggml_backend_buft_get_alloc_size(pGalloc->vBufts[vPHn->vBufferId], pParent);
  size_t vNodeSize = ggml_backend_buft_get_alloc_size(pGalloc->vBufts[vHn->vBufferId], pNode);

  GGML_ASSERT(vParentSize >= vNodeSize);

  // note: we want after the freeing the chunks to continue to be aligned
  struct ggml_dyn_tallocr *vPAlloc = pGalloc->vBufTallocs[vPHn->vBufferId];
  vParentSize = fAlignedOffset(NULL, vParentSize, vPAlloc->vAlignment);
  vNodeSize = fAlignedOffset(NULL, vNodeSize, vPAlloc->vAlignment);

  if (vParentSize > vNodeSize) {
    struct buffer_address vPAddr = vPHn->addr;
    vPAddr.offset += vNodeSize;
    size_t vExtraSize = vParentSize - vNodeSize;
    AT_PRINTF("freeing extra %zu bytes from parent %s for %s\n", vExtraSize, pParent->name, pNode->name);
    fGgmlDynTallocrFreeBytes(vPAlloc, vPAddr, vExtraSize);
  }
}

static void fGgmlGallocrAllocateNode(ggml_gallocr_t pGalloc, struct ggml_tensor *pNode, int pBufferId) {
  GGML_ASSERT(pBufferId >= 0);
  struct hash_node *vHn = fGgmlGallocrHashGet(pGalloc, pNode);

  if (!fGgmlGallocrIsAllocated(pGalloc, pNode) && !ggml_impl_is_view(pNode)) {
    vHn->vAllocated = true;
    assert(vHn->addr.offset == 0);

    // try to reuse a parent's buffer (inplace)
    if (ggml_op_can_inplace(pNode->op)) {
      for (int vI = 0; vI < GGML_MAX_SRC; vI++) {
        struct ggml_tensor *vParent = pNode->src[vI];
        if (vParent == NULL) {
          continue;
        }

        // if the node's data is external, then we cannot re-use it
        if (!fGgmlGallocrIsOwn(pGalloc, vParent)) {
          AT_PRINTF("not reusing parent %s for %s as %p is external\n", vParent->name, pNode->name, vParent->data);
          continue;
        }

        // outputs cannot be reused
        if (vParent->flags & GGML_TENSOR_FLAG_OUTPUT ||
            (vParent->view_src != NULL && vParent->view_src->flags & GGML_TENSOR_FLAG_OUTPUT)) {
          AT_PRINTF("not reusing parent %s for %s as it is an output\n", vParent->name, pNode->name);
          continue;
        }

        if (!ggml_are_same_layout(pNode, vParent)) {
          AT_PRINTF("not reusing parent %s for %s as layouts are different\n", vParent->name, pNode->name);
          continue;
        }

        struct hash_node *vPHn = fGgmlGallocrHashGet(pGalloc, vParent);
        if (vPHn->vNChildren == 1 && vPHn->vNViews == 0) {
          if (ggml_impl_is_view(vParent)) {
            struct ggml_tensor *vViewSrc = vParent->view_src;
            struct hash_node *vViewSrcHn = fGgmlGallocrHashGet(pGalloc, vViewSrc);
            if (vViewSrcHn->vNViews == 1 && vViewSrcHn->vNChildren == 0 && vViewSrc->data == vParent->data) {
              AT_PRINTF("reusing view parent %s (%s) for %s\n", vParent->name, vViewSrc->name, pNode->name);
              assert(vViewSrcHn->addr.chunk == vPHn->addr.chunk && vViewSrcHn->addr.offset == vPHn->addr.offset);
              vHn->vBufferId = vPHn->vBufferId;
              vHn->addr = vPHn->addr;
              vPHn->vAllocated = false; // avoid freeing the parent
              vViewSrcHn->vAllocated = false;
              fGgmlGallocrFreeExtraSpace(pGalloc, pNode, vViewSrc);
              return;
            }
          } else {
            AT_PRINTF("reusing parent %s for %s\n", vParent->name, pNode->name);
            vHn->vBufferId = vPHn->vBufferId;
            vHn->addr = vPHn->addr;
            vPHn->vAllocated = false; // avoid freeing the parent
            fGgmlGallocrFreeExtraSpace(pGalloc, pNode, vParent);
            return;
          }
        }
      }
    }
    // allocate tensor from the buffer
    struct ggml_dyn_tallocr *vAlloc = pGalloc->vBufTallocs[pBufferId];
    ggml_backend_buffer_type_t vBuft = pGalloc->vBufts[pBufferId];
    size_t vSize = ggml_backend_buft_get_alloc_size(vBuft, pNode);
    vHn->vBufferId = pBufferId;
    vHn->addr = fGgmlDynTallocrAlloc(vAlloc, vSize, pNode);
  }
}

static void fGgmlGallocrFreeNode(ggml_gallocr_t pGalloc, struct ggml_tensor *pNode) {
  // graph outputs are never freed
  if (pNode->flags & GGML_TENSOR_FLAG_OUTPUT) {
    AT_PRINTF("not freeing output %s\n", pNode->name);
    return;
  }

  struct hash_node *vHn = fGgmlGallocrHashGet(pGalloc, pNode);
  int vBufferId = vHn->vBufferId;
  struct ggml_dyn_tallocr *vAlloc = pGalloc->vBufTallocs[vBufferId];
  ggml_backend_buffer_type_t vBuft = pGalloc->vBufts[vBufferId];
  size_t size = ggml_backend_buft_get_alloc_size(vBuft, pNode);

  AT_PRINTF("%s: freeing %s at {chunk=%d, offset=%zu} (%zu bytes) - n_free_blocks = %d\n", __func__, pNode->name,
            vHn->addr.chunk, vHn->addr.offset, size, vAlloc->chunks[vHn->addr.chunk]->vNFreeBlocks);
#ifdef GGML_ALLOCATOR_DEBUG
  remove_allocated_tensor(vAlloc, vHn->addr, pNode);
#endif

  fGgmlDynTallocrFreeBytes(vAlloc, vHn->addr, size);
  vHn->vAllocated = false;
}

static int fGetNodeBufferId(const int *pNodeBufferIds, int pI) { return pNodeBufferIds ? pNodeBufferIds[pI] : 0; }

static void fGgmlGallocrAllocGraphImpl(ggml_gallocr_t pGalloc, struct ggml_cgraph *pGraph, const int *pNodeBufferIds,
                                       const int *pLeafBufferIds) {
  // clear hash tables
  ggml_hash_set_reset(&pGalloc->vHashSet);
  memset(pGalloc->vHashValues, 0, sizeof(struct hash_node) * pGalloc->vHashSet.size);

  // allocate leafs
  // these may be tensors that the application is not using in the graph, but may still want to allocate for other
  // purposes
  for (int vI = 0; vI < pGraph->n_leafs; vI++) {
    struct ggml_tensor *vLeaf = pGraph->leafs[vI];
    fGgmlGallocrAllocateNode(pGalloc, vLeaf, fGetNodeBufferId(pLeafBufferIds, vI));
  }

  // count number of children and views
  // allocate other graph inputs and leafs first to avoid overwriting them
  for (int vI = 0; vI < pGraph->n_nodes; vI++) {
    struct ggml_tensor *vNode = pGraph->nodes[vI];

    // TODO: better way to add external dependencies
    // GGML_OP_NONE does not appear normally in the graph nodes, but is used by ggml-backend to add dependencies to
    // control when some tensors are allocated and freed. in this case, the dependencies are in `src`, but the node
    // itself is never used and should not be considered a dependency
    if (ggml_impl_is_view(vNode) && vNode->op != GGML_OP_NONE) {
      struct ggml_tensor *vViewSrc = vNode->view_src;
      fGgmlGallocrHashGet(pGalloc, vViewSrc)->vNViews += 1;
    }

    if (vNode->flags & GGML_TENSOR_FLAG_INPUT) {
      fGgmlGallocrAllocateNode(pGalloc, pGraph->nodes[vI], fGetNodeBufferId(pNodeBufferIds, vI));
    }

    for (int vJ = 0; vJ < GGML_MAX_SRC; vJ++) {
      struct ggml_tensor *vSrc = vNode->src[vJ];
      if (vSrc == NULL) {
        continue;
      }

      fGgmlGallocrHashGet(pGalloc, vSrc)->vNChildren += 1;

      // allocate explicit inputs
      if (vSrc->flags & GGML_TENSOR_FLAG_INPUT) {
        fGgmlGallocrAllocateNode(pGalloc, vSrc, fGetNodeBufferId(pNodeBufferIds, vI));
      }
    }
  }

  // allocate tensors
  for (int vI = 0; vI < pGraph->n_nodes; vI++) {
    struct ggml_tensor *vNode = pGraph->nodes[vI];
    int vBufferId = fGetNodeBufferId(pNodeBufferIds, vI);

    // allocate parents (only leafs need to be allocated at this point)
    for (int vJ = 0; vJ < GGML_MAX_SRC; vJ++) {
      struct ggml_tensor *vParent = vNode->src[vJ];
      if (vParent == NULL) {
        continue;
      }
      fGgmlGallocrAllocateNode(pGalloc, vParent, vBufferId);
    }

    // allocate node
    fGgmlGallocrAllocateNode(pGalloc, vNode, vBufferId);

    AT_PRINTF("exec: %s (%s) <= ", ggml_op_desc(vNode), vNode->name);
    for (int vJ = 0; vJ < GGML_MAX_SRC; vJ++) {
      struct ggml_tensor *vParent = vNode->src[vJ];
      if (vParent == NULL) {
        continue;
      }
      AT_PRINTF("%s", vParent->name);
      if (vJ < GGML_MAX_SRC - 1 && vNode->src[vJ + 1] != NULL) {
        AT_PRINTF(", ");
      }
    }
    AT_PRINTF("\n");

    // update parents
    for (int vJ = 0; vJ < GGML_MAX_SRC; vJ++) {
      struct ggml_tensor *vParent = vNode->src[vJ];
      if (vParent == NULL) {
        continue;
      }
      struct hash_node *vPHn = fGgmlGallocrHashGet(pGalloc, vParent);
      vPHn->vNChildren -= 1;

      AT_PRINTF("parent %s: %d children, %d views, allocated: %d\n", vParent->name, vPHn->vNChildren, vPHn->vNViews,
                vPHn->vAllocated);

      if (vPHn->vNChildren == 0 && vPHn->vNViews == 0) {
        if (ggml_impl_is_view(vParent)) {
          struct ggml_tensor *vViewSrc = vParent->view_src;
          struct hash_node *vViewSrcHn = fGgmlGallocrHashGet(pGalloc, vViewSrc);
          vViewSrcHn->vNViews -= 1;
          AT_PRINTF("view_src %s: %d children, %d views\n", vViewSrc->name, vViewSrcHn->vNChildren,
                    vViewSrcHn->vNViews);
          if (vViewSrcHn->vNViews == 0 && vViewSrcHn->vNChildren == 0 && vViewSrcHn->vAllocated) {
            fGgmlGallocrFreeNode(pGalloc, vViewSrc);
          }
        } else if (vPHn->vAllocated) {
          fGgmlGallocrFreeNode(pGalloc, vParent);
        }
      }
      AT_PRINTF("\n");
    }
  }
}

static bool fGgmlGallocrReserveNImpl(ggml_gallocr_t pGalloc, struct ggml_cgraph *pGraph, const int *pNodeBufferIds,
                                     const int *pLeafBufferIds, bool pNoAlloc) {
  size_t vMinHashSize = pGraph->n_nodes + pGraph->n_leafs;
  // add 25% margin to avoid hash collisions
  vMinHashSize += vMinHashSize / 4;

  // initialize hash table
  if (pGalloc->vHashSet.size < vMinHashSize) {
    ggml_hash_set_free(&pGalloc->vHashSet);
    pGalloc->vHashSet = ggml_hash_set_new(vMinHashSize);
    GGML_ASSERT(pGalloc->vHashSet.keys != NULL);

    free(pGalloc->vHashValues);
    pGalloc->vHashValues = malloc(sizeof(struct hash_node) * pGalloc->vHashSet.size);
    GGML_ASSERT(pGalloc->vHashValues != NULL);
  }

  // reset allocators
  for (int vI = 0; vI < pGalloc->vNBuffers; vI++) {
    fGgmlDynTallocrReset(pGalloc->vBufTallocs[vI]);
  }

  // allocate in hash table
  fGgmlGallocrAllocGraphImpl(pGalloc, pGraph, pNodeBufferIds, pLeafBufferIds);

  // set the node_allocs from the hash table
  if (pGalloc->vNNodes < pGraph->n_nodes) {
    free(pGalloc->vNodeAllocs);
    pGalloc->vNodeAllocs = calloc(pGraph->n_nodes, sizeof(struct node_alloc));
    GGML_ASSERT(pGalloc->vNodeAllocs != NULL);
  }
  pGalloc->vNNodes = pGraph->n_nodes;
  for (int vI = 0; vI < pGraph->n_nodes; vI++) {
    struct ggml_tensor *vNode = pGraph->nodes[vI];
    struct node_alloc *vNodeAlloc = &pGalloc->vNodeAllocs[vI];
    if (vNode->view_src || vNode->data) {
      vNodeAlloc->vDst.vBufferId = -1;
      vNodeAlloc->vDst.addr = GGML_BUFFER_ADDRESS_INVALID;
      vNodeAlloc->vDst.vSizeMax = 0;
    } else {
      struct hash_node *vHn = fGgmlGallocrHashGet(pGalloc, vNode);
      vNodeAlloc->vDst.vBufferId = vHn->vBufferId;
      vNodeAlloc->vDst.addr = vHn->addr;
      vNodeAlloc->vDst.vSizeMax = ggml_backend_buft_get_alloc_size(pGalloc->vBufts[vHn->vBufferId], vNode);
    }
    for (int vJ = 0; vJ < GGML_MAX_SRC; vJ++) {
      struct ggml_tensor *vSrc = vNode->src[vJ];
      if (!vSrc || vSrc->view_src || vSrc->data) {
        vNodeAlloc->vSrc[vJ].vBufferId = -1;
        vNodeAlloc->vSrc[vJ].addr = GGML_BUFFER_ADDRESS_INVALID;
        vNodeAlloc->vSrc[vJ].vSizeMax = 0;
      } else {
        struct hash_node *vHn = fGgmlGallocrHashGet(pGalloc, vSrc);
        vNodeAlloc->vSrc[vJ].vBufferId = vHn->vBufferId;
        vNodeAlloc->vSrc[vJ].addr = vHn->addr;
        vNodeAlloc->vSrc[vJ].vSizeMax = ggml_backend_buft_get_alloc_size(pGalloc->vBufts[vHn->vBufferId], vSrc);
      }
    }
  }
  if (pGalloc->vNLeafs < pGraph->n_leafs) {
    free(pGalloc->vLeafAllocs);
    pGalloc->vLeafAllocs = calloc(pGraph->n_leafs, sizeof(pGalloc->vLeafAllocs[0]));
    GGML_ASSERT(pGalloc->vLeafAllocs != NULL);
  }
  pGalloc->vNLeafs = pGraph->n_leafs;
  for (int vI = 0; vI < pGraph->n_leafs; vI++) {
    struct ggml_tensor *vLeaf = pGraph->leafs[vI];
    struct hash_node *vHn = fGgmlGallocrHashGet(pGalloc, vLeaf);
    if (vLeaf->view_src || vLeaf->data) {
      pGalloc->vLeafAllocs[vI].vLeaf.vBufferId = -1;
      pGalloc->vLeafAllocs[vI].vLeaf.addr = GGML_BUFFER_ADDRESS_INVALID;
      pGalloc->vLeafAllocs[vI].vLeaf.vSizeMax = 0;
    } else {
      pGalloc->vLeafAllocs[vI].vLeaf.vBufferId = vHn->vBufferId;
      pGalloc->vLeafAllocs[vI].vLeaf.addr = vHn->addr;
      pGalloc->vLeafAllocs[vI].vLeaf.vSizeMax =
          ggml_backend_buft_get_alloc_size(pGalloc->vBufts[vHn->vBufferId], vLeaf);
    }
  }

  // reallocate buffers if needed
  for (int vI = 0; vI < pGalloc->vNBuffers; vI++) {
    // if the buffer type is used multiple times, we reuse the same buffer
    for (int vJ = 0; vJ < vI; vJ++) {
      if (pGalloc->vBufTallocs[vJ] == pGalloc->vBufTallocs[vI]) {
        pGalloc->vBuffers[vI] = pGalloc->vBuffers[vJ];
        break;
      }
    }

    // even if there are no tensors allocated in this buffer, we still need to allocate it to initialize views
    bool vRealloc = pGalloc->vBuffers[vI] == NULL;
    size_t vNewSize = 0;
    for (int vC = 0; vC < pGalloc->vBufTallocs[vI]->vNChunks; vC++) {
      size_t vCurChunkSize = pGalloc->vBuffers[vI] ? fGgmlVbufferChunkSize(pGalloc->vBuffers[vI], vC) : 0;
      size_t vNewChunkSize = fGgmlDynTallocrMaxSize(pGalloc->vBufTallocs[vI], vC);
      vNewSize += vNewChunkSize;
      if (vNewChunkSize > vCurChunkSize) {
        vRealloc = true;
      }
    }
    if (vRealloc) {
#ifndef NDEBUG
      {
        size_t vCurSize = pGalloc->vBuffers[vI] ? fGgmlVbufferSize(pGalloc->vBuffers[vI]) : 0;
        if (vCurSize > 0) {
          GGML_LOG_DEBUG("%s: reallocating %s buffer from size %.02f MiB to %.02f MiB\n", __func__,
                         ggml_backend_buft_name(pGalloc->vBufts[vI]), vCurSize / 1024.0 / 1024.0,
                         vNewSize / 1024.0 / 1024.0);
        }
      }
#endif
      fGgmlVbufferFree(pGalloc->vBuffers[vI]);
      if (pNoAlloc) {
        pGalloc->vBuffers[vI] = NULL;
      } else {
        pGalloc->vBuffers[vI] =
            fGgmlVbufferAlloc(pGalloc->vBufts[vI], pGalloc->vBufTallocs[vI], GGML_BACKEND_BUFFER_USAGE_COMPUTE);
        if (pGalloc->vBuffers[vI] == NULL) {
          GGML_LOG_ERROR("%s: failed to allocate %s buffer of size %zu\n", __func__,
                         ggml_backend_buft_name(pGalloc->vBufts[vI]), vNewSize);
          return false;
        }
      }
    }
  }

  return true;
}

void ggml_gallocr_reserve_n_size(ggml_gallocr_t pGalloc, struct ggml_cgraph *pGraph, const int *pNodeBufferIds,
                                 const int *pLeafBufferIds, size_t *pSizes) {
  GGML_ASSERT(fGgmlGallocrReserveNImpl(pGalloc, pGraph, pNodeBufferIds, pLeafBufferIds, /*no_alloc =*/true));
  for (int vI = 0; vI < pGalloc->vNBuffers; vI++) {
    pSizes[vI] = 0;
    for (int vC = 0; vC < pGalloc->vBufTallocs[vI]->vNChunks; vC++) {
      pSizes[vI] += pGalloc->vBufTallocs[vI]->chunks[vC]->vMaxSize;
    }
  }
}

bool ggml_gallocr_reserve_n(ggml_gallocr_t pGalloc, struct ggml_cgraph *pGraph, const int *pNodeBufferIds,
                            const int *pLeafBufferIds) {
  return fGgmlGallocrReserveNImpl(pGalloc, pGraph, pNodeBufferIds, pLeafBufferIds, /*no_alloc =*/false);
}

bool ggml_gallocr_reserve(ggml_gallocr_t pGalloc, struct ggml_cgraph *pGraph) {
  return ggml_gallocr_reserve_n(pGalloc, pGraph, NULL, NULL);
}

static void fGgmlGallocrInitTensor(ggml_gallocr_t pGalloc, struct ggml_tensor *pTensor,
                                   struct tensor_alloc *pTensorAlloc) {
  int vBufferId = pTensorAlloc->vBufferId;
  assert(pTensor->data || pTensor->view_src ||
         ggml_backend_buft_get_alloc_size(pGalloc->vBufts[vBufferId], pTensor) <= pTensorAlloc->vSizeMax);

  if (pTensor->view_src != NULL) {
    if (pTensor->buffer == NULL) {
      assert(pTensorAlloc->addr.offset == SIZE_MAX);
      if (pTensor->view_src->buffer == NULL) {
        // this tensor was allocated without ggml-backend
        return;
      }
      ggml_backend_view_init(pTensor);
    }
  } else {
    if (pTensor->data == NULL) {
      assert(pTensorAlloc->addr.offset != SIZE_MAX);
      assert(ggml_backend_buft_get_alloc_size(pGalloc->vBufts[vBufferId], pTensor) <= pTensorAlloc->vSizeMax);
      fGgmlVbufferTensorAlloc(pGalloc->vBuffers[vBufferId], pTensor, pTensorAlloc->addr);
    } else {
      if (pTensor->buffer == NULL) {
        // this tensor was allocated without ggml-backend
        return;
      }
    }
  }
}

static bool fGgmlGallocrNodeNeedsRealloc(ggml_gallocr_t pGalloc, struct ggml_tensor *pNode,
                                         struct tensor_alloc *pTalloc) {
  size_t vNodeSize = 0;
  if (!pNode->data && !pNode->view_src) {
    // If we previously had data but don't now then reallocate
    if (pTalloc->vBufferId < 0) {
      return false;
    }
    vNodeSize = ggml_backend_buft_get_alloc_size(pGalloc->vBufts[pTalloc->vBufferId], pNode);
  }
  return pTalloc->vSizeMax >= vNodeSize;
}

static bool fGgmlGallocrNeedsRealloc(ggml_gallocr_t pGalloc, struct ggml_cgraph *pGraph) {
  if (pGalloc->vNNodes != pGraph->n_nodes) {
#ifndef NDEBUG
    GGML_LOG_DEBUG("%s: graph has different number of nodes\n", __func__);
#endif
    return true;
  }

  if (pGalloc->vNLeafs != pGraph->n_leafs) {
#ifndef NDEBUG
    GGML_LOG_DEBUG("%s: graph has different number of leafs\n", __func__);
#endif
    return true;
  }

  for (int vI = 0; vI < pGraph->n_nodes; vI++) {
    struct ggml_tensor *vNode = pGraph->nodes[vI];
    struct node_alloc *vNodeAlloc = &pGalloc->vNodeAllocs[vI];

    if (!fGgmlGallocrNodeNeedsRealloc(pGalloc, vNode, &vNodeAlloc->vDst)) {
#ifndef NDEBUG
      GGML_LOG_DEBUG("%s: node %s is not valid\n", __func__, vNode->name);
#endif
      return true;
    }

    for (int vJ = 0; vJ < GGML_MAX_SRC; vJ++) {
      struct ggml_tensor *vSrc = vNode->src[vJ];
      if (vSrc == NULL) {
        continue;
      }
      if (!fGgmlGallocrNodeNeedsRealloc(pGalloc, vSrc, &vNodeAlloc->vSrc[vJ])) {
#ifndef NDEBUG
        GGML_LOG_DEBUG("%s: src %d (%s) of node %s is not valid\n", __func__, vJ, vSrc->name, vNode->name);
#endif
        return true;
      }
    }
  }

  return false;
}

bool ggml_gallocr_alloc_graph(ggml_gallocr_t pGalloc, struct ggml_cgraph *pGraph) {
  if (fGgmlGallocrNeedsRealloc(pGalloc, pGraph)) {
    if (pGalloc->vNBuffers == 1) {
#ifndef NDEBUG
      GGML_LOG_DEBUG("%s: reallocating buffers automatically\n", __func__);
#endif
      if (!ggml_gallocr_reserve(pGalloc, pGraph)) {
        return false;
      }
    } else {
#ifndef NDEBUG
      GGML_LOG_DEBUG("%s: cannot reallocate multi buffer graph automatically, call reserve\n", __func__);
#endif
      return false;
    }
  }

  // reset buffers
  for (int vI = 0; vI < pGalloc->vNBuffers; vI++) {
    if (pGalloc->vBuffers[vI] != NULL) {
      fGgmlVbufferReset(pGalloc->vBuffers[vI]);
    }
  }

  // allocate the graph tensors from the previous assignments
  // leafs
  for (int vI = 0; vI < pGraph->n_leafs; vI++) {
    struct ggml_tensor *vLeaf = pGraph->leafs[vI];
    struct leaf_alloc *vLeafAlloc = &pGalloc->vLeafAllocs[vI];
    fGgmlGallocrInitTensor(pGalloc, vLeaf, &vLeafAlloc->vLeaf);
  }
  // nodes
  for (int vI = 0; vI < pGraph->n_nodes; vI++) {
    struct ggml_tensor *vNode = pGraph->nodes[vI];
    struct node_alloc *vNodeAlloc = &pGalloc->vNodeAllocs[vI];
    for (int vJ = 0; vJ < GGML_MAX_SRC; vJ++) {
      struct ggml_tensor *vSrc = vNode->src[vJ];
      if (vSrc == NULL) {
        continue;
      }
      fGgmlGallocrInitTensor(pGalloc, vSrc, &vNodeAlloc->vSrc[vJ]);
    }
    fGgmlGallocrInitTensor(pGalloc, vNode, &vNodeAlloc->vDst);
  }

  return true;
}

size_t ggml_gallocr_get_buffer_size(ggml_gallocr_t pGalloc, int pBufferId) {
  GGML_ASSERT(pBufferId >= 0 && pBufferId < pGalloc->vNBuffers);

  if (pGalloc->vBuffers[pBufferId] == NULL) {
    return 0;
  }

  for (int vI = 0; vI < pBufferId; vI++) {
    if (pGalloc->vBuffers[vI] == pGalloc->vBuffers[pBufferId]) {
      // this buffer is the same as a previous one due to the same buffer type being used multiple times
      // only return the buffer size the first time it appears to avoid double counting
      return 0;
    }
  }

  return fGgmlVbufferSize(pGalloc->vBuffers[pBufferId]);
}

// utils

static void fFreeBuffers(ggml_backend_buffer_t **pBuffers, const size_t *pNBuffers) {
  for (size_t vI = 0; vI < *pNBuffers; vI++) {
    ggml_backend_buffer_free((*pBuffers)[vI]);
  }
  free(*pBuffers);
}

static bool fAllocTensorRange(struct ggml_context *pCtx, struct ggml_tensor *pFirst, struct ggml_tensor *pLast,
                              ggml_backend_buffer_type_t pBuft, size_t pSize, ggml_backend_buffer_t **pBuffers,
                              size_t *pNBuffers) {

  ggml_backend_buffer_t vBuffer = ggml_backend_buft_alloc_buffer(pBuft, pSize);
  if (vBuffer == NULL) {
    GGML_LOG_ERROR("%s: failed to allocate %s buffer of size %zu\n", __func__, ggml_backend_buft_name(pBuft), pSize);
    fFreeBuffers(pBuffers, pNBuffers);
    return false;
  }

  *pBuffers = realloc(*pBuffers, sizeof(ggml_backend_buffer_t) * (*pNBuffers + 1));
  (*pBuffers)[(*pNBuffers)++] = vBuffer;

  struct ggml_tallocr vTallocr = ggml_tallocr_new(vBuffer);

  for (struct ggml_tensor *vT = pFirst; vT != pLast; vT = ggml_get_next_tensor(pCtx, vT)) {
    enum ggml_status vStatus = GGML_STATUS_SUCCESS;
    if (vT->data == NULL) {
      if (vT->view_src == NULL) {
        vStatus = ggml_tallocr_alloc(&vTallocr, vT);
      } else if (vT->buffer == NULL) {
        vStatus = ggml_backend_view_init(vT);
      }
    } else {
      if (vT->view_src != NULL && vT->buffer == NULL) {
        // view of a pre-allocated tensor
        vStatus = ggml_backend_view_init(vT);
      }
    }
    if (vStatus != GGML_STATUS_SUCCESS) {
      GGML_LOG_ERROR("%s: failed to initialize tensor %s\n", __func__, vT->name);
      fFreeBuffers(pBuffers, pNBuffers);
      return false;
    }
  }

  return true;
}

static ggml_backend_buffer_t fGgmlBackendAllocCtxTensorsFromBuftImpl(struct ggml_context *pCtx,
                                                                     ggml_backend_buffer_type_t pBuft,
                                                                     size_t *pNbytesTotal, bool pNoAlloc) {
  GGML_ASSERT(ggml_get_no_alloc(pCtx) == true);

  size_t vAlignment = ggml_backend_buft_get_alignment(pBuft);
  size_t vMaxSize = ggml_backend_buft_get_max_size(pBuft);

  ggml_backend_buffer_t *vBuffers = NULL;
  size_t vNBuffers = 0;
  *pNbytesTotal = 0;

  size_t vCurBufSize = 0;
  struct ggml_tensor *vFirst = ggml_get_first_tensor(pCtx);
  for (struct ggml_tensor *vT = vFirst; vT != NULL; vT = ggml_get_next_tensor(pCtx, vT)) {
    size_t vThisSize = 0;
    if (vT->data == NULL && vT->view_src == NULL) {
      vThisSize = GGML_PAD(ggml_backend_buft_get_alloc_size(pBuft, vT), vAlignment);
    }

    if (vCurBufSize > 0 && (vCurBufSize + vThisSize) > vMaxSize) {
      // allocate tensors in the current buffer
      if (!pNoAlloc && !fAllocTensorRange(pCtx, vFirst, vT, pBuft, vCurBufSize, &vBuffers, &vNBuffers)) {
        return NULL;
      }
      vFirst = vT;
      *pNbytesTotal += vCurBufSize;
      vCurBufSize = vThisSize;
    } else {
      vCurBufSize += vThisSize;
    }
  }

  // allocate remaining tensors
  if (vCurBufSize > 0) {
    *pNbytesTotal += vCurBufSize;
    if (!pNoAlloc && !fAllocTensorRange(pCtx, vFirst, NULL, pBuft, vCurBufSize, &vBuffers, &vNBuffers)) {
      return NULL;
    }
  }

  if (pNoAlloc) {
    return NULL;
  }

  if (vNBuffers == 0) {
#ifndef NDEBUG
    GGML_LOG_DEBUG("%s: all tensors in the context are already allocated\n", __func__);
#endif
    GGML_ASSERT(!vBuffers);
    return NULL;
  }

  ggml_backend_buffer_t vBuffer;
  if (vNBuffers == 1) {
    vBuffer = vBuffers[0];
  } else {
    vBuffer = ggml_backend_multi_buffer_alloc_buffer(vBuffers, vNBuffers);
  }
  if (vBuffers) {
    free(vBuffers); // can be NULL if context is empty or no_alloc
  }
  return vBuffer;
}

size_t ggml_backend_alloc_ctx_tensors_from_buft_size(struct ggml_context *pCtx, ggml_backend_buffer_type_t pBuft) {
  size_t vNbytesTotal = 0;
  ggml_backend_buffer_t vBuf = fGgmlBackendAllocCtxTensorsFromBuftImpl(pCtx, pBuft, &vNbytesTotal, /*no_alloc=*/true);
  GGML_ASSERT(!vBuf);
  return vNbytesTotal;
}

ggml_backend_buffer_t ggml_backend_alloc_ctx_tensors_from_buft(struct ggml_context *pCtx,
                                                               ggml_backend_buffer_type_t pBuft) {
  size_t vNbytesTotal = 0;
  if (ggml_backend_buft_is_meta(pBuft)) {
    return ggml_backend_meta_alloc_ctx_tensors_from_buft(pCtx, pBuft);
  }
  return fGgmlBackendAllocCtxTensorsFromBuftImpl(pCtx, pBuft, &vNbytesTotal, /*no_alloc =*/false);
}

ggml_backend_buffer_t ggml_backend_alloc_ctx_tensors(struct ggml_context *pCtx, ggml_backend_t pBackend) {
  return ggml_backend_alloc_ctx_tensors_from_buft(pCtx, ggml_backend_get_default_buffer_type(pBackend));
}
