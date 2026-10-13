#pragma once

#include "mgs.h"

#include "mgs-cparams.h"

#include <array>
#include <vector>
#include <set>
#include <bitset>
#include <memory>
#include <unordered_map>

// keep this struct lightweight
struct mgs_ubatch {
  bool mEqualSeqs() const { return vBEqualSeqs != 0; }

  // typical for M-RoPE cases:
  //   0 - sequential position of the tokens/embeddings in the sequence
  //   1 - y position in the image
  //   2 - x position in the image
  //   3 - other
  bool mIsPos2d() const {
    // TODO @ngxson : we may need to check for model arch when more models use >1 positions
    return vNPos >= 3;
  }

  uint32_t vBEqualSeqs; // note: this is a boolean, but we use an int32_t for alignment
                        //       otherwise address sanitizer complains
  // TODO: whole_seqs for embeddings?

  uint32_t vNTokens;    // total tokens (n_seq_tokens * n_seqs)
  uint32_t vNSeqTokens; // tokens per sequence set
  uint32_t vNSeqs;      // sequence sets in the ubatch
  uint32_t vNSeqsUnq;   // unique sequence ids in the ubatch
  uint32_t vNPos;       // number of position inputs for each token/embedding

  // seq_id_unq: unique sequence ids in the ubatch
  // seq_idx:    indices of the unique sequence ids in the ubatch in [0, n_seqs_unq)
  //             used for extracting sequence pooled embeddings

  //                          // size               | idx | val
  mgs_token *vToken;     // [n_tokens]         | i   | id, token
  float *vEmbd;          // [n_embd, n_tokens] | i   | embd
  mgs_pos *vPos;         // [n_tokens*n_pos]   | i   | pos
  int32_t *vNSeqId;      // [n_tokens]         | i   | -
  mgs_seq_id **vSeqId;   // [n_tokens]         | s   | s0, s1, seq_id
  mgs_seq_id *vSeqIdUnq; // [n_seqs_unq]       | s   | seq_id
  int32_t *vSeqIdx;      // [MGS_MAX_SEQ]    | -   | seq_idx
  int8_t *vOutput;       // [n_tokens]         | i   | -

  struct data_t {
    std::vector<mgs_token> lToken;
    std::vector<float> lEmbd;
    std::vector<mgs_pos> lPos;
    std::vector<int32_t> lNSeqId;
    std::vector<mgs_seq_id *> lSeqId; // these point into the seq_id_data below
    std::vector<mgs_seq_id> lSeqIdUnq;
    std::vector<int32_t> lSeqIdx;
    std::vector<int8_t> lOutput;
    std::vector<int32_t> lBatchIdxs; // original batch index for each token

    std::vector<mgs_seq_id> lSeqIdData;
  };

  // the mgs_ubatch pointers above point to this data if set. otherwise - point to external non-owning data
  std::shared_ptr<data_t> vData;
};

// a helper for sanitizing, fulfilling and splitting a batch
class mgs_batch_allocr {
public:
  mgs_batch_allocr(uint32_t pNPosPerEmbd);

  // sanitize and auto-gen missing data in the input batch
  // memory is optional. if provided will be used to check for sequence continuity and to determine the positions
  bool mInit(const mgs_batch &pBatchInp, const mgs_vocab &pVocab, const mgs_memory_i *pMemory, uint32_t pNEmbd,
             uint32_t pNSeqMax, bool pOutputAll);

  const mgs_batch &mGetBatch() const;

  uint32_t mGetNTokens() const;
  uint32_t mGetNOutputs() const;
  uint32_t mGetNUsed() const;

  // the array of output indices in the order they were encountered during the ubatch splitting
  std::vector<int32_t> &mGetOutIds();

  // min/max positions of each sequence in the current ubatch
  mgs_pos mSeqPosMin(mgs_seq_id pSeqId) const;
  mgs_pos mSeqPosMax(mgs_seq_id pSeqId) const;

  // call once before splitting the batch to reset the internal state
  void mSplitReset();

  // simple split, unknown number of sequence sets of unequal lengths
  mgs_ubatch mSplitSimple(uint32_t pNUbatch);

  // make ubatches of equal-length sequences sets
  // if sequential == true, the tokens in the ubatch will have increasing sequential sequence ids
  // n_keep_tail = minimum trailing tokens of a seq that must land in the same ubatch
  mgs_ubatch mSplitEqual(uint32_t pNUbatch, bool pSequential, uint32_t pNKeepTail);

  // sequence-set-wise split - each ubatch contains a single sequence-set
  mgs_ubatch mSplitSeq(uint32_t pNUbatch);

  // a helper method for creating a well-defined ubatch of tokens
  // TODO: support embeddings if needed in the future
  mgs_ubatch mUbatchReserve(uint32_t pNSeqTokens, uint32_t pNSeqs);

private:
  void mClear();

  // create the next ubatch based on the provided batch indices (idxs) and the number of sequence sets (n_seqs)
  // return mgs_ubatch.n_tokens == 0 if the entire batch was consumed
  mgs_ubatch mUbatchAdd(const std::vector<int32_t> &pIdxs, uint32_t pNSeqs, bool pEqualSeqs);

  // for debugging, start with MGS_BATCH_DEBUG=2
  void mUbatchPrint(const mgs_ubatch &pUbatch, int pDebug);

  mgs_batch vBatch;

  // only for debugging purposes
  const mgs_vocab *vVocab;

  // TODO: this is more of a temporary solution until we have a better way to handle multiple positions per token/embd
  const uint32_t cNPosPerEmbd;

  uint32_t vNEmbd;
  uint32_t vNSeqMax;
  uint32_t vNOutputs;

  std::array<mgs_seq_id, 1> aSeqId0 = {{0}}; // default sequence id

  std::vector<mgs_pos> lPos;
  std::vector<int32_t> lNSeqId;
  std::vector<mgs_seq_id *> lSeqId;
  std::vector<mgs_seq_id> lSeqIdUnq;
  std::vector<int32_t> lSeqIdx;
  std::vector<int8_t> lOutput;

  using pos_set_t = std::set<mgs_pos>;
  using seq_cpl_t = std::vector<bool>;

  // helper flag to quickly determine if there are any coupled sequences in the batch
  bool vHasCpl = false;

  std::vector<pos_set_t> lSeqPos; // seq_pos[s]: the set of positions in sequence s
  std::vector<seq_cpl_t> lSeqCpl; // seq_cpl[s0][s1]: if sequence s0 is coupled to sequence s1

  using idx_vec_t = std::vector<int32_t>;
  using seq_set_t = std::bitset<MGS_MAX_SEQ>;

  std::vector<seq_set_t> lSeqSet; // seq_set[i]: the sequence set of token i

  std::unordered_map<seq_set_t, idx_vec_t> dSeqSetMap; // the indices at which the sequence set appears

  // batch indices of the output
  std::vector<int32_t> lOutIds;

  uint32_t vNUsed;

  // used[i] indicates if token i has already been used in a previous ubatch
  std::vector<bool> lUsed;

  int vDebug;
};
