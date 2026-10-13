#include "common.h"
#include "download.h"
#include "log.h"
#include "mgs.h"
#include "mtmd.h"
#include "mtmd-helper.h"
#include "chat.h"
#include "base64.hpp"

#include "server-common.h"

#include <chrono>
#include <condition_variable>
#include <random>
#include <sstream>
#include <fstream>
#include <limits>
#include <mutex>

json fFormatErrorResponse(const std::string &pMessage, const enum error_type pType) {
  std::string vTypeStr;
  int vCode = 500;
  switch (pType) {
  case cErrorTypeInvalidRequest:
    vTypeStr = "invalid_request_error";
    vCode = 400;
    break;
  case cErrorTypeAuthentication:
    vTypeStr = "authentication_error";
    vCode = 401;
    break;
  case cErrorTypeNotFound:
    vTypeStr = "not_found_error";
    vCode = 404;
    break;
  case cErrorTypeServer:
    vTypeStr = "server_error";
    vCode = 500;
    break;
  case cErrorTypePermission:
    vTypeStr = "permission_error";
    vCode = 403;
    break;
  case cErrorTypeNotSupported:
    vTypeStr = "not_supported_error";
    vCode = 501;
    break;
  case cErrorTypeUnavailable:
    vTypeStr = "unavailable_error";
    vCode = 503;
    break;
  case cErrorTypeExceedContextSize:
    vTypeStr = "exceed_context_size_error";
    vCode = 400;
    break;
  }
  return json{
      {"code", vCode},
      {"message", pMessage},
      {"type", vTypeStr},
  };
}

//
// random string / id
//

std::string fRandomString() {
  static const std::string cStr("0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz");

  std::random_device vRd;
  std::mt19937 vGenerator(vRd());

  std::string vResult(32, ' ');

  for (int vI = 0; vI < 32; ++vI) {
    vResult[vI] = cStr[vGenerator() % cStr.size()];
  }

  return vResult;
}

std::string fGenChatcmplid() { return "chatcmpl-" + fRandomString(); }

std::string fGenToolCallId() { return fRandomString(); }

const char *fGetMediaMarker() {
  static const std::string cMarker = []() {
    // allow user to pin a reproducible marker via env var
    const char *vEnv = getenv("MGS_MEDIA_MARKER");
    if (vEnv && vEnv[0] != '\0') {
      return std::string(vEnv);
    }
    return std::string("<__media_") + fRandomString() + "__>";
  }();
  return cMarker.c_str();
}

//
// lora utils
//

bool fLoraAllAlora(const std::vector<common_adapter_lora_info> &pLoras) {
  bool vFoundAlora = false;
  for (const auto &vLora : pLoras) {
    if (vLora.vScale != 0) {
      if (mgs_adapter_get_alora_n_invocation_tokens(vLora.vPtr) == 0) {
        return false;
      }
      vFoundAlora = true;
    }
  }
  return vFoundAlora;
}

bool fLoraShouldClearCache(const std::vector<common_adapter_lora_info> &pCurrent,
                           const std::vector<common_adapter_lora_info> &pNext) {

  // This should always be called after determining that the two sets are
  // _not_ equal. This assert is therefore some slightly wasted work and
  // should be safe to remove as long as this method is called correctly.
  GGML_ASSERT(!fAreLoraEqual(pCurrent, pNext));

  return (!(fLoraGetEnabledIds(pCurrent).empty() || fLoraAllAlora(pCurrent)) || !fLoraAllAlora(pNext));
}

std::map<int, float> fParseLoraRequest(const json &pData) {
  std::map<int, float> dLora;

  // set value
  for (const auto &vEntry : pData) {
    int vId = fJsonValue(vEntry, "id", -1);
    float vScale = fJsonValue(vEntry, "scale", 0.0f);
    dLora[vId] = vScale;
  }

  return dLora;
}

bool fAreLoraEqual(const std::vector<common_adapter_lora_info> &pL1, const std::vector<common_adapter_lora_info> &pL2) {
  if (pL1.size() != pL2.size()) {
    return false;
  }
  for (size_t vI = 0; vI < pL1.size(); ++vI) {
    // we don't check lora.path to reduce the time complexity
    if (pL1[vI].vScale != pL2[vI].vScale || pL1[vI].vPtr != pL2[vI].vPtr) {
      return false;
    }
  }
  return true;
}

std::vector<size_t> fLoraGetEnabledIds(const std::vector<common_adapter_lora_info> &pLoras) {
  std::vector<size_t> lEnabledIds;
  for (size_t vI = 0; vI < pLoras.size(); ++vI) {
    if (pLoras[vI].vScale > 0) {
      lEnabledIds.push_back(vI);
    }
  }
  return lEnabledIds;
}

//
// base64 utils (TODO: use the base64::decode from base64.hpp)
//

static const std::string cBase64Chars = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
                                        "abcdefghijklmnopqrstuvwxyz"
                                        "0123456789+/";

static inline bool fIsBase64(uint8_t pC) { return (isalnum(pC) || (pC == '+') || (pC == '/')); }

static inline vRawBuffer fBase64Decode(const std::string &pEncodedString) {
  int vI = 0;
  int vJ = 0;
  int vIn = 0;

  int vInLen = pEncodedString.size();

  uint8_t aCharArray4[4];
  uint8_t aCharArray3[3];

  vRawBuffer vRet;

  while (vInLen-- && (pEncodedString[vIn] != '=') && fIsBase64(pEncodedString[vIn])) {
    aCharArray4[vI++] = pEncodedString[vIn];
    vIn++;
    if (vI == 4) {
      for (vI = 0; vI < 4; vI++) {
        aCharArray4[vI] = cBase64Chars.find(aCharArray4[vI]);
      }

      aCharArray3[0] = ((aCharArray4[0]) << 2) + ((aCharArray4[1] & 0x30) >> 4);
      aCharArray3[1] = ((aCharArray4[1] & 0xf) << 4) + ((aCharArray4[2] & 0x3c) >> 2);
      aCharArray3[2] = ((aCharArray4[2] & 0x3) << 6) + aCharArray4[3];

      for (vI = 0; (vI < 3); vI++) {
        vRet.push_back(aCharArray3[vI]);
      }

      vI = 0;
    }
  }

  if (vI) {
    for (vJ = vI; vJ < 4; vJ++) {
      aCharArray4[vJ] = 0;
    }

    for (vJ = 0; vJ < 4; vJ++) {
      aCharArray4[vJ] = cBase64Chars.find(aCharArray4[vJ]);
    }

    aCharArray3[0] = ((aCharArray4[0]) << 2) + ((aCharArray4[1] & 0x30) >> 4);
    aCharArray3[1] = ((aCharArray4[1] & 0xf) << 4) + ((aCharArray4[2] & 0x3c) >> 2);
    aCharArray3[2] = ((aCharArray4[2] & 0x3) << 6) + aCharArray4[3];

    for (vJ = 0; vJ < vI - 1; vJ++) {
      vRet.push_back(aCharArray3[vJ]);
    }
  }

  return vRet;
}

//
// server_tokens implementation
//

server_tokens::server_tokens(mtmd::input_chunks &pMtmdChunks, bool pHasMtmd) : vHasMtmd(pHasMtmd) {
  for (size_t vI = 0; vI < pMtmdChunks.size(); ++vI) {
    mPushBack(pMtmdChunks[vI]);
  }
}

server_tokens::server_tokens(const mgs_tokens &pTokens, bool pHasMtmd) : vHasMtmd(pHasMtmd), vTokens(pTokens) {}

mgs_pos server_tokens::mPosNext(int64_t pNTokens) const {
  if (!vHasMtmd) {
    if (pNTokens < 0) {
      return vTokens.size();
    }

    return pNTokens;
  }

  if (pNTokens < 0) {
    mgs_pos vRes = vTokens.size();

    for (auto vIt = dMapIdxToMedia.begin(); vIt != dMapIdxToMedia.end(); ++vIt) {
      const auto &vChunk = vIt->second;
      vRes += mtmd_input_chunk_get_n_pos(vChunk.get()) - mtmd_input_chunk_get_n_tokens(vChunk.get());
    }

    return vRes;
  }

  int64_t vIdx = 0;
  mgs_pos vPos = 0;

  GGML_ASSERT(pNTokens <= (int64_t)vTokens.size());

  while (vIdx < pNTokens) {
    const auto cMediaIt = dMapIdxToMedia.find(vIdx);
    if (cMediaIt != dMapIdxToMedia.end()) {
      const auto &vChunk = cMediaIt->second;
      const mgs_pos cNPos = mtmd_input_chunk_get_n_pos(vChunk.get());
      const size_t cNTok = mtmd_input_chunk_get_n_tokens(vChunk.get());

      vPos += cNPos;
      vIdx += cNTok;
    } else {
      vPos++;
      vIdx++;
    }
  }

  return vPos;
}

size_t server_tokens::mSizeUpToPos(mgs_pos pMaxPos) const {
  if (!vHasMtmd) {
    return std::min((size_t)pMaxPos, vTokens.size());
  }

  size_t vIdx = 0;
  mgs_pos vPos = 0;

  while (vIdx < vTokens.size()) {
    const auto cMediaIt = dMapIdxToMedia.find(vIdx);
    if (cMediaIt != dMapIdxToMedia.end()) {
      const auto &vChunk = cMediaIt->second;
      const mgs_pos cNPos = mtmd_input_chunk_get_n_pos(vChunk.get());
      const size_t cNTok = mtmd_input_chunk_get_n_tokens(vChunk.get());

      vPos += cNPos;
      vIdx += cNTok;
    } else {
      vPos++;
      vIdx++;
    }

    if (vPos >= pMaxPos) {
      break;
    }
  }

  return vIdx;
}

std::string server_tokens::mStr() const {
  std::ostringstream vOss;
  vOss << "tokens: ";
  for (size_t vIdx = 0; vIdx < vTokens.size(); ++vIdx) {
    mgs_token vT = vTokens[vIdx];
    vOss << "idx:" << vIdx << " ";
    if (vT == MGS_TOKEN_NULL) {
      vOss << "<embd> ";
    } else {
      vOss << vT << " ";
    }
  }
  vOss << "\n";
  vOss << "image idx: ";
  for (const auto &vIt : dMapIdxToMedia) {
    vOss << vIt.first << ", ";
  }
  return vOss.str();
}

const mtmd::input_chunk_ptr &server_tokens::mFindChunk(size_t pIdx) const {
  auto vIt = dMapIdxToMedia.find(pIdx);
  if (vIt != dMapIdxToMedia.end()) {
    return vIt->second;
  }
  throw std::runtime_error("Chunk not found");
}

std::pair<const mtmd::input_chunk_ptr *, size_t> server_tokens::mFindNextMediaChunk(size_t pIdx) const {
  auto vIt = dMapIdxToMedia.upper_bound(pIdx);
  if (vIt != dMapIdxToMedia.end()) {
    return {&vIt->second, vIt->first};
  }
  return {nullptr, 0};
}

void server_tokens::mPushBack(mgs_token pTok) {
  if (pTok == MGS_TOKEN_NULL) {
    throw std::runtime_error("Invalid token");
  }
  vTokens.emplace_back(pTok);
}

void server_tokens::mPushBack(const mtmd_input_chunk *pChunk) {
  auto vType = mtmd_input_chunk_get_type(pChunk);
  if (vType == MTMD_INPUT_CHUNK_TYPE_IMAGE || vType == MTMD_INPUT_CHUNK_TYPE_AUDIO) {
    GGML_ASSERT(vHasMtmd);
    const size_t cNTokens = mtmd_input_chunk_get_n_tokens(pChunk);
    size_t vStartIdx = vTokens.size();
    for (size_t vI = 0; vI < cNTokens; ++vI) {
      vTokens.emplace_back(MGS_TOKEN_NULL);
    }
    mtmd::input_chunk_ptr vNewChunk(mtmd_input_chunk_copy(pChunk));
    dMapIdxToMedia[vStartIdx] = std::move(vNewChunk);
  } else if (vType == MTMD_INPUT_CHUNK_TYPE_TEXT) {
    size_t vNTokens;
    const auto *vTextTokens = mtmd_input_chunk_get_tokens_text(pChunk, &vNTokens);
    for (size_t vI = 0; vI < vNTokens; ++vI) {
      mPushBack(vTextTokens[vI]);
    }
  } else {
    GGML_ABORT("Invalid chunk type");
  }
}

void server_tokens::mPushBack(server_tokens &pTokens) {
  size_t vStartIdx = size();
  for (size_t vI = 0; vI < pTokens.size(); vI++) {
    mPushBack(pTokens[vI]);
  }
  if (pTokens.vHasMtmd) {
    // Assert if we are copying MTMD chunks to a server_tokens that does not have mtmd.
    // We could also just check, but this will prevent silently dropping MTMD data.
    GGML_ASSERT(vHasMtmd);
    for (auto vIt = pTokens.dMapIdxToMedia.begin(); vIt != pTokens.dMapIdxToMedia.end();) {
      auto *vChunk = pTokens.dMapIdxToMedia[vIt->first].get();
      mtmd::input_chunk_ptr vNewChunk(mtmd_input_chunk_copy(vChunk));
      dMapIdxToMedia[vStartIdx + vIt->first] = std::move(vNewChunk);
    }
  }
}

void server_tokens::mInsert(const mgs_tokens &pInpTokens) {
  vTokens.insert(vTokens.end(), pInpTokens.begin(), pInpTokens.end());
}

const mgs_tokens &server_tokens::mGetTokens() const {
  GGML_ASSERT(!vHasMtmd);
  return vTokens;
}

mgs_tokens server_tokens::mGetTextTokens() const {
  mgs_tokens vRes;
  vRes.reserve(vTokens.size());
  for (mgs_token vT : vTokens) {
    if (vT != MGS_TOKEN_NULL) {
      vRes.push_back(vT);
    }
  }
  return vRes;
}

void server_tokens::mSetToken(mgs_pos pPos, mgs_token pId) {
  GGML_ASSERT(!vHasMtmd); // only allow this if mtmd is disabled
  vTokens[pPos] = pId;
}

void server_tokens::mKeepFirst(size_t pN) {
  GGML_ASSERT(pN <= vTokens.size());
  if (vHasMtmd) {
    if (pN == vTokens.size()) {
      return; // nothing to do
    }
    // we throw an error if we try to remove a token in the middle of an image
    // for ex. with input of 5 text tokens and 2 images:
    //    [0] [1] [2] [3] [4] [img0] [img0] [img0] [img1] [img1]
    // n  1   2   3   4   5   6      7      8      9      10
    // allowed to resize      ^                    ^
    // disallowed to resize          ^      ^             ^
    if (pN > 0) {
      // make sure we never remove tokens in the middle of an image
      // note that the case where we keep a full image at the end is allowed:
      //   tokens[n - 1] == MGS_TOKEN_NULL && tokens[n] != MGS_TOKEN_NULL
      if (vTokens[pN - 1] == MGS_TOKEN_NULL && vTokens[pN] == MGS_TOKEN_NULL) {
        mFindChunk(pN - 1); // will throw an error if the token is not begin-of-chunk
      }
    }
    // remove all image chunks that are not used anymore
    for (auto vIt = dMapIdxToMedia.begin(); vIt != dMapIdxToMedia.end();) {
      size_t vIdx = vIt->first;
      if (vIdx >= pN) {
        vIt = dMapIdxToMedia.erase(vIt);
      } else {
        ++vIt;
      }
    }
  }
  vTokens.resize(pN);
}

std::string server_tokens::mDetokenize(const mgs_context *pCtx, bool pSpecial) const {
  mgs_tokens vTextTokens;
  vTextTokens.reserve(vTokens.size());
  for (const auto &vT : vTokens) {
    if (vT != MGS_TOKEN_NULL) {
      vTextTokens.push_back(vT);
    }
  }
  return fCommonDetokenize(pCtx, vTextTokens, pSpecial);
}

size_t server_tokens::mGetCommonPrefix(const server_tokens &pB) const {
  const size_t cMaxIdx = std::min(vTokens.size(), pB.vTokens.size());

  if (!vHasMtmd) {
    for (size_t vI = 0; vI < cMaxIdx; ++vI) {
      if (vTokens[vI] == pB.vTokens[vI]) {
        continue;
      }

      return vI;
    }

    return cMaxIdx;
  }

  for (size_t vI = 0; vI < cMaxIdx; ++vI) {
    const mgs_token cAi = vTokens[vI];
    const mgs_token cBi = pB.vTokens[vI];

    if (cAi == MGS_TOKEN_NULL && cBi == MGS_TOKEN_NULL) {
      const auto &vAChunk = mFindChunk(vI);
      const auto &vBChunk = pB.mFindChunk(vI);

      GGML_ASSERT(vAChunk && vBChunk);

      const std::string cIdAi = mtmd_input_chunk_get_id(vAChunk.get());
      const std::string cIdBi = mtmd_input_chunk_get_id(vBChunk.get());

      const size_t cNTokA = mtmd_input_chunk_get_n_tokens(vAChunk.get());
      const size_t cNTokB = mtmd_input_chunk_get_n_tokens(vBChunk.get());

      if (cIdAi == cIdBi && cNTokA == cNTokB) {
        GGML_ASSERT(cNTokA > 0 && "Invalid media chunk"); // should never happen
        vI += cNTokA - 1;                                 // will be +1 by the for loop
        continue;
      }

      return vI;
    }

    if (cAi == cBi) {
      continue;
    }

    return vI;
  }

  return cMaxIdx; // all tokens are equal
}

common_chat_msg_spans server_tokens::mFindMessageSpans(const common_chat_msg_delimiters &pDelims) const {
  std::map<size_t, size_t> dSkips;
  for (const auto &vIt : dMapIdxToMedia) {
    dSkips[vIt.first] = mtmd_input_chunk_get_n_tokens(vIt.second.get());
  }
  return pDelims.mSplit(vTokens, dSkips);
}

bool server_tokens::mValidate(const struct mgs_context *pCtx) const {
  const mgs_model *vModel = mgs_get_model(pCtx);
  const mgs_vocab *vVocab = mgs_model_get_vocab(vModel);
  const int32_t cNVocab = mgs_vocab_n_tokens(vVocab);

  for (size_t vI = 0; vI < vTokens.size(); ++vI) {
    const auto &vT = vTokens[vI];
    if (vT == MGS_TOKEN_NULL) {
      try {
        const auto &vChunk = mFindChunk(vI);
        size_t vNTokens = mtmd_input_chunk_get_n_tokens(vChunk.get());
        vI += vNTokens - 1; // will be +1 by the for loop
      } catch (const std::exception &vE) {
        return false;
      }
    } else if (vT < 0 || vT >= cNVocab) {
      return false;
    }
  }
  return true;
}

server_tokens server_tokens::mClone() const {
  server_tokens vRes;
  vRes.vHasMtmd = vHasMtmd;
  vRes.vTokens = vTokens;
  for (auto vIt = dMapIdxToMedia.begin(); vIt != dMapIdxToMedia.end(); ++vIt) {
    size_t vIdx = vIt->first;
    const mtmd::input_chunk_ptr &vChunk = vIt->second;
    vRes.dMapIdxToMedia[vIdx] = mtmd::input_chunk_ptr(mtmd_input_chunk_copy(vChunk.get()));
  }
  return vRes;
}

//
// tokenizer and input processing utils
//

bool fJsonIsArrayOfNumbers(const json &pData) {
  if (pData.is_array()) {
    for (const auto &vE : pData) {
      if (!vE.is_number_integer()) {
        return false;
      }
    }
    return true;
  }
  return false;
}

bool fJsonIsArrayOfMixedNumbersStrings(const json &pData) {
  bool vSeenString = false;
  bool vSeenNumber = false;
  if (pData.is_array()) {
    for (const auto &vE : pData) {
      vSeenString |= vE.is_string();
      vSeenNumber |= vE.is_number_integer();
      if (vSeenNumber && vSeenString) {
        return true;
      }
    }
  }
  return false;
}

bool fJsonIsArrayAndContainsNumbers(const json &pData) {
  if (pData.is_array()) {
    for (const auto &vE : pData) {
      if (vE.is_number_integer()) {
        return true;
      }
    }
    return false;
  }
  return false;
}

json fJsonGetNestedValues(const std::vector<std::string> &pPaths, const json &pJs) {
  json vResult = json::object();

  for (const std::string &vPath : pPaths) {
    json vCurrent = pJs;
    const auto cKeys = fStringSplit<std::string>(vPath, /*separator*/ '/');
    bool vValidPath = true;
    for (const std::string &vK : cKeys) {
      if (vValidPath && vCurrent.is_object() && vCurrent.contains(vK)) {
        vCurrent = vCurrent[vK];
      } else {
        vValidPath = false;
      }
    }
    if (vValidPath) {
      vResult[vPath] = vCurrent;
    }
  }
  return vResult;
}

mgs_tokens fTokenizeMixed(const mgs_vocab *pVocab, const json &pJsonPrompt, bool pAddSpecial, bool pParseSpecial) {
  // If `pAddSpecial` is true, we only add BOS, when json_prompt is a string,
  // or the first element of the json_prompt array is a string.
  mgs_tokens vPromptTokens;

  if (pJsonPrompt.is_array()) {
    bool vFirst = true;
    for (const auto &vP : pJsonPrompt) {
      if (vP.is_string()) {
        auto vS = vP.template get<std::string>();

        mgs_tokens vP;
        if (vFirst) {
          vP = fCommonTokenize(pVocab, vS, pAddSpecial, pParseSpecial);
          vFirst = false;
        } else {
          vP = fCommonTokenize(pVocab, vS, false, pParseSpecial);
        }

        vPromptTokens.insert(vPromptTokens.end(), vP.begin(), vP.end());
      } else {
        if (vFirst) {
          vFirst = false;
        }

        vPromptTokens.push_back(vP.template get<mgs_token>());
      }
    }
  } else {
    auto vS = pJsonPrompt.template get<std::string>();
    vPromptTokens = fCommonTokenize(pVocab, vS, pAddSpecial, pParseSpecial);
  }

  return vPromptTokens;
}

size_t fValidateUtf8(const std::string &pText) {
  size_t vLen = pText.size();
  if (vLen == 0)
    return 0;

  // Check the last few bytes to see if a multi-byte character is cut off
  for (size_t vI = 1; vI <= 4 && vI <= vLen; ++vI) {
    unsigned char vC = pText[vLen - vI];
    // Check for start of a multi-byte sequence from the end
    if ((vC & 0xE0) == 0xC0) {
      // 2-byte character start: 110xxxxx
      // Needs at least 2 bytes
      if (vI < 2)
        return vLen - vI;
    } else if ((vC & 0xF0) == 0xE0) {
      // 3-byte character start: 1110xxxx
      // Needs at least 3 bytes
      if (vI < 3)
        return vLen - vI;
    } else if ((vC & 0xF8) == 0xF0) {
      // 4-byte character start: 11110xxx
      // Needs at least 4 bytes
      if (vI < 4)
        return vLen - vI;
    }
  }

  // If no cut-off multi-byte character is found, return full length
  return vLen;
}

server_tokens fProcessMtmdPrompt(mtmd_context *pMctx, const std::string &pPrompt, const std::vector<vRawBuffer> &pFiles,
                                 bool pIsPlaceholder) {
  // decoded images and audio stay in memory until the prompt is tokenized, so their total size is bounded per request
  // (each item and each video are bounded by the decoding limits of mtmd-helper)
  static constexpr size_t cMaxDecodedMediaBytes = static_cast<size_t>(2) << 30; // 2 GiB
  size_t vDecodedBytes = 0;
  // these will be freed upon going out of scope
  mtmd::bitmaps vBitmaps;
  std::vector<mtmd_helper::video_ptr> lVideos;
  for (auto &vFile : pFiles) {
    auto vOut = mtmd_helper_bitmap_init_from_buf(pMctx, vFile.data(), vFile.size(), pIsPlaceholder);
    if (!vOut.bitmap) {
      throw std::runtime_error(
          "Failed to load image, audio or video file: unsupported format, or it exceeds the decoding limits");
    }
    vBitmaps.entries.emplace_back(vOut.bitmap);
    if (vOut.video_ctx) {
      lVideos.emplace_back(vOut.video_ctx);
    }
    vDecodedBytes += mtmd_bitmap_get_n_bytes(vOut.bitmap);
    if (vDecodedBytes > cMaxDecodedMediaBytes) {
      throw std::invalid_argument("The decoded images and audio of this request exceed 2 GiB; send fewer or smaller "
                                  "media files");
    }
  }
  // process prompt
  std::vector<server_tokens> lInputs;
  // multimodal
  mtmd_input_text vInpTxt = {
      pPrompt.data(),
      pPrompt.size(),
      /* add_special */ true,
      /* parse_special */ true,
  };
  mtmd::input_chunks vChunks(mtmd_input_chunks_init());
  auto vBitmapsCPtr = vBitmaps.c_ptr();
  int32_t vTokenized = mtmd_tokenize(pMctx, vChunks.ptr.get(), &vInpTxt, vBitmapsCPtr.data(), vBitmapsCPtr.size());
  if (vTokenized != 0) {
    throw std::runtime_error("Failed to tokenize prompt");
  }
  auto vResult = server_tokens(vChunks, true);
  return vResult;
}

/**
 * break the input "prompt" object into multiple prompt if needed, then tokenize them
 * use fTokenizeInputPrompts() if the input could be an array.
 * this supports these cases:
 * - "prompt": "string"
 * - "prompt": [12, 34, 56]
 * - "prompt": [12, 34, "string", 56, 78]
 * - "prompt": { "prompt_string": "string", "multimodal_data": [ "base64" ] }
 */
static server_tokens fTokenizeInputSubprompt(const mgs_vocab *pVocab, mtmd_context *pMctx, const json &pJsonPrompt,
                                             bool pAddSpecial, bool pParseSpecial) {
  constexpr char aJsonStringPromptKey[] = "prompt_string";
  constexpr char aJsonMtmdDataKey[] = "multimodal_data";
  const bool cHasMtmd = pMctx != nullptr;
  if (pJsonPrompt.is_string() || fJsonIsArrayOfMixedNumbersStrings(pJsonPrompt)) {
    // string or mixed
    mgs_tokens vTmp = fTokenizeMixed(pVocab, pJsonPrompt, pAddSpecial, pParseSpecial);
    return server_tokens(vTmp, false);
  } else if (fJsonIsArrayOfNumbers(pJsonPrompt)) {
    // array of tokens
    mgs_tokens vTmp = pJsonPrompt.get<mgs_tokens>();
    return server_tokens(vTmp, false);
  } else if (pJsonPrompt.contains(aJsonStringPromptKey)) {
    // JSON object with prompt key.
    if (pJsonPrompt.contains(aJsonMtmdDataKey)) {
      if (!cHasMtmd)
        throw std::runtime_error("Multimodal data provided, but model does not support multimodal requests.");

      // JSON object with prompt and multimodal key.
      std::vector<vRawBuffer> lFiles;
      for (const auto &vEntry : pJsonPrompt.at(aJsonMtmdDataKey)) {
        lFiles.push_back(fBase64Decode(vEntry));
      }
      return fProcessMtmdPrompt(pMctx, pJsonPrompt.at(aJsonStringPromptKey), lFiles);
    } else {
      // Not multimodal, but contains a subobject.
      mgs_tokens vTmp = fTokenizeMixed(pVocab, pJsonPrompt.at(aJsonStringPromptKey), pAddSpecial, pParseSpecial);
      return server_tokens(vTmp, false);
    }
  } else {
    throw std::runtime_error("\"prompt\" elements must be a string, a list of tokens, a JSON object containing a "
                             "prompt string, or a list of mixed strings & tokens.");
  }
}

std::vector<server_tokens> fTokenizeInputPrompts(const mgs_vocab *pVocab, mtmd_context *pMctx, const json &pJsonPrompt,
                                                 bool pAddSpecial, bool pParseSpecial) {
  std::vector<server_tokens> lResult;
  if (pJsonPrompt.is_array() && !fJsonIsArrayAndContainsNumbers(pJsonPrompt)) {
    lResult.reserve(pJsonPrompt.size());
    for (const auto &vP : pJsonPrompt) {
      lResult.push_back(fTokenizeInputSubprompt(pVocab, pMctx, vP, pAddSpecial, pParseSpecial));
    }
  } else {
    lResult.push_back(fTokenizeInputSubprompt(pVocab, pMctx, pJsonPrompt, pAddSpecial, pParseSpecial));
  }
  if (lResult.empty()) {
    throw std::runtime_error("\"prompt\" must not be empty");
  }
  return lResult;
}

//
// OAI utils
//

// used by /completions endpoint
json fOaicompatCompletionParamsParse(const json &pBody) {
  json vMgsParams;

  if (!pBody.contains("prompt")) {
    throw std::runtime_error("\"prompt\" is required");
  }

  // Handle "stop" field
  if (pBody.contains("stop") && pBody.at("stop").is_string()) {
    vMgsParams["stop"] = json::array({pBody.at("stop").get<std::string>()});
  } else {
    vMgsParams["stop"] = fJsonValue(pBody, "stop", json::array());
  }

  // Handle "echo" field
  if (fJsonValue(pBody, "echo", false)) {
    throw std::runtime_error("Only no echo is supported");
  }

  // Params supported by OAI but unsupported by mgs
  static const std::vector<std::string> lUnsupportedParams{"best_of", "suffix"};
  for (const auto &vParam : lUnsupportedParams) {
    if (pBody.contains(vParam)) {
      throw std::runtime_error("Unsupported param: " + vParam);
    }
  }

  // Copy remaining properties to mgs_params
  for (const auto &vItem : pBody.items()) {
    // Exception: if "n_predict" is present, we overwrite the value specified earlier by "max_tokens"
    if (!vMgsParams.contains(vItem.key()) || vItem.key() == "n_predict") {
      vMgsParams[vItem.key()] = vItem.value();
    }
  }

  return vMgsParams;
}

// limits for remote media downloads, which run synchronously on HTTP worker threads
static constexpr long cMediaDownloadMaxBytes = 1024 * 1024 * 10; // 10 MiB
static constexpr long cMediaDownloadConnectTimeout = 10;        // seconds, name resolution and each connection
static constexpr long cMediaDownloadIoTimeout = 10;             // seconds, each read or write
static constexpr long cMediaDownloadTotalTimeout = 30;          // seconds, the whole download including redirects
static constexpr int cMediaDownloadMaxRedirects = 5;
static constexpr int cMediaDownloadMaxConcurrent = 8; // downloads in flight at once across all requests

static std::mutex vMediaDownloadMutex;
static std::condition_variable vMediaDownloadCv;
static int vMediaDownloadActive = 0;

// holds one of the cMediaDownloadMaxConcurrent download slots, waiting at most the total download timeout for it
struct media_download_slot {
  media_download_slot() {
    std::unique_lock<std::mutex> vLock(vMediaDownloadMutex);
    if (!vMediaDownloadCv.wait_for(vLock, std::chrono::seconds(cMediaDownloadTotalTimeout),
                                   [] { return vMediaDownloadActive < cMediaDownloadMaxConcurrent; })) {
      throw std::runtime_error("Too many remote media downloads in progress, try again later");
    }
    vMediaDownloadActive++;
  }
  ~media_download_slot() {
    {
      std::lock_guard<std::mutex> vLock(vMediaDownloadMutex);
      vMediaDownloadActive--;
    }
    vMediaDownloadCv.notify_one();
  }
  media_download_slot(const media_download_slot &) = delete;
  media_download_slot &operator=(const media_download_slot &) = delete;
};

// url can be
// - http(s):// for remote files
// - file:// for local files (only allowed if media_path is set)
// - data: for base64 encoded data with uri scheme (e.g. data:image/png;base64,...)
// - raw base64 encoded data
static void fHandleMedia(std::vector<vRawBuffer> &pOutFiles, const std::string &pUrl, const std::string &pMediaPath,
                         bool pAcceptBase64Uri, bool pAllowPrivateUrls) {
  if (!pMediaPath.empty()) {
    // should already be enforced by arg.cpp, but checking just in case
    GGML_ASSERT(pMediaPath.back() == DIRECTORY_SEPARATOR);
  }

  if (fStringStartsWith(pUrl, "http")) {
    // download remote media; unless --media-allow-private-urls is set, only public addresses are reached, each host
    // is resolved once and the connection is pinned to the checked address, and redirects are revalidated hop by hop
    common_remote_params vParams;
    vParams.vMaxSize = cMediaDownloadMaxBytes;
    vParams.vTimeout = cMediaDownloadIoTimeout;
    vParams.vConnectTimeout = cMediaDownloadConnectTimeout;
    vParams.vTotalTimeout = cMediaDownloadTotalTimeout;
    vParams.vMaxRedirects = cMediaDownloadMaxRedirects;
    vParams.vPublicHostsOnly = !pAllowPrivateUrls;
    // credentials embedded in the URL are not written to the log
    std::string vShownUrl = pUrl;
    const size_t cAuthorityStart = vShownUrl.find("://");
    if (cAuthorityStart != std::string::npos) {
      const size_t cAuthorityEnd = vShownUrl.find('/', cAuthorityStart + 3);
      const size_t cAt = vShownUrl.rfind('@', cAuthorityEnd == std::string::npos ? std::string::npos : cAuthorityEnd);
      if (cAt != std::string::npos && cAt > cAuthorityStart) {
        vShownUrl.replace(cAuthorityStart + 3, cAt - cAuthorityStart - 3, "****:****");
      }
    }
    SRV_INF("downloading media from '%s'\n", vShownUrl.c_str());
    std::pair<long, std::vector<char>> vRes;
    try {
      media_download_slot vSlot;
      vRes = fCommonRemoteGetContent(pUrl, vParams);
    } catch (const common_remote_destination_error &vError) {
      throw std::invalid_argument(std::string("Failed to download media: ") + vError.what() +
                                  " (remote media from loopback, private, link-local and other non-public addresses "
                                  "is refused unless the server is started with --media-allow-private-urls)");
    } catch (const std::exception &vError) {
      throw std::runtime_error(std::string("Failed to download media: ") + vError.what());
    }
    if (200 <= vRes.first && vRes.first < 300) {
      SRV_INF("downloaded %zu bytes\n", vRes.second.size());
      vRawBuffer vData;
      vData.insert(vData.end(), vRes.second.begin(), vRes.second.end());
      pOutFiles.push_back(vData);
    } else {
      throw std::runtime_error("Failed to download media: HTTP status " + std::to_string(vRes.first));
    }

  } else if (fStringStartsWith(pUrl, "file://")) {
    if (pMediaPath.empty()) {
      throw std::invalid_argument("file:// URLs are not allowed unless --media-path is specified");
    }
    // load local image file
    std::string vFilePath = pUrl.substr(7); // remove "file://"
    vRawBuffer vData;
    if (!fFsValidateFilename(vFilePath, true)) {
      throw std::invalid_argument("file path is not allowed: " + vFilePath);
    }
    SRV_INF("loading image from local file '%s'\n", (pMediaPath + vFilePath).c_str());
    std::ifstream vFile(pMediaPath + vFilePath, std::ios::binary);
    if (!vFile) {
      throw std::invalid_argument("file does not exist or cannot be opened: " + vFilePath);
    }
    vData.assign((std::istreambuf_iterator<char>(vFile)), std::istreambuf_iterator<char>());
    pOutFiles.push_back(vData);

  } else if (pAcceptBase64Uri && fStringStartsWith(pUrl, "data:")) {
    // try to decode base64 image
    std::vector<std::string> lParts = fStringSplit<std::string>(pUrl, /*separator*/ ',');
    if (lParts.size() != 2) {
      throw std::runtime_error("Invalid uri-encoded base64 value");
    } else if (!fStringStartsWith(lParts[0], "data:image/")) {
      throw std::runtime_error("Invalid uri format: " + lParts[0]);
    } else if (!fStringEndsWith(lParts[0], "base64")) {
      throw std::runtime_error("uri must be base64 encoded");
    } else {
      auto vBase64Data = lParts[1];
      auto vDecodedData = fBase64Decode(vBase64Data);
      pOutFiles.push_back(vDecodedData);
    }

  } else {
    // try as raw base64 string
    auto vDecodedData = fBase64Decode(pUrl);
    if (vDecodedData.empty()) {
      throw std::runtime_error("Invalid base64 value");
    }
    pOutFiles.push_back(vDecodedData);
  }
}

// used by /chat/completions endpoint
json fOaicompatChatParamsParse(json &pBody, /* openai api json semantics */
                               const server_chat_params &pOpt, std::vector<vRawBuffer> &pOutFiles) {
  json vMgsParams;

  auto vTools = fJsonValue(pBody, "tools", json());
  auto vHasTools = vTools.is_array() && !vTools.empty();
  auto vStream = fJsonValue(pBody, "stream", false);
  auto vToolChoice = fJsonValue(pBody, "tool_choice", std::string("auto"));

  // Handle "stop" field
  if (pBody.contains("stop") && pBody.at("stop").is_string()) {
    vMgsParams["stop"] = json::array({pBody.at("stop").get<std::string>()});
  } else {
    vMgsParams["stop"] = fJsonValue(pBody, "stop", json::array());
  }

  auto vJsonSchema = fJsonValue(pBody, "json_schema", json());
  auto vGrammar = fJsonValue(pBody, "grammar", std::string());
  if (!vJsonSchema.is_null() && !vGrammar.empty()) {
    throw std::runtime_error("Cannot use both json_schema and grammar");
  }

  // Handle "response_format" field
  if (pBody.contains("response_format")) {
    json vResponseFormat = fJsonValue(pBody, "response_format", json::object());
    std::string vResponseType = fJsonValue(vResponseFormat, "type", std::string());
    if (vResponseType == "json_object") {
      if (vResponseFormat.contains("schema") || vJsonSchema.empty()) {
        vJsonSchema = fJsonValue(vResponseFormat, "schema", json::object());
      }
    } else if (vResponseType == "json_schema") {
      auto vSchemaWrapper = fJsonValue(vResponseFormat, "json_schema", json::object());
      vJsonSchema = fJsonValue(vSchemaWrapper, "schema", json::object());
    } else if (!vResponseType.empty() && vResponseType != "text") {
      throw std::invalid_argument("response_format type must be one of \"text\" or \"json_object\", but got: " +
                                  vResponseType);
    }
  }

  // get input files
  if (!pBody.contains("messages")) {
    throw std::invalid_argument("'messages' is required");
  }
  json &vMessages = pBody.at("messages");
  if (!vMessages.is_array()) {
    throw std::invalid_argument("Expected 'messages' to be an array");
  }
  for (auto &vMsg : vMessages) {
    std::string vRole = fJsonValue(vMsg, "role", std::string());
    if (vRole != "assistant" && !vMsg.contains("content")) {
      throw std::invalid_argument("All non-assistant messages must contain 'content'");
    }
    if (vRole == "assistant") {
      if (!vMsg.contains("content") && !vMsg.contains("tool_calls")) {
        throw std::invalid_argument("Assistant message must contain either 'content' or 'tool_calls'!");
      }
      if (!vMsg.contains("content")) {
        continue; // avoid errors with no content
      }
    }
    json &vContent = vMsg.at("content");
    if (vContent.is_string() || vContent.is_null()) {
      continue;
    }

    if (!vContent.is_array()) {
      throw std::invalid_argument("Expected 'content' to be a string or an array");
    }

    for (auto &vP : vContent) {
      std::string vType = fJsonValue(vP, "type", std::string());
      if (vType == "image_url") {
        if (!pOpt.vAllowImage) {
          throw std::runtime_error(
              "image input is not supported - hint: if this is unexpected, you may need to provide the mmproj");
        }

        json vImageUrl = fJsonValue(vP, "image_url", json::object());
        std::string vUrl = fJsonValue(vImageUrl, "url", std::string());
        fHandleMedia(pOutFiles, vUrl, pOpt.vMediaPath, true, pOpt.vMediaAllowPrivateUrls);

        vP["type"] = "media_marker";
        vP["text"] = fGetMediaMarker();
        vP.erase("image_url");

      } else if (vType == "input_audio") {
        if (!pOpt.vAllowAudio) {
          throw std::runtime_error(
              "audio input is not supported - hint: if this is unexpected, you may need to provide the mmproj");
        }

        // note: don't need to validate "format", it's redundant
        json vInputAudio = fJsonValue(vP, "input_audio", json::object());
        std::string vUrl = fJsonValue(vInputAudio, "data", fJsonValue(vInputAudio, "url", std::string()));
        fHandleMedia(pOutFiles, vUrl, pOpt.vMediaPath, false, pOpt.vMediaAllowPrivateUrls);

        vP["type"] = "media_marker";
        vP["text"] = fGetMediaMarker();
        vP.erase("input_audio");

      } else if (vType == "input_video") {
        if (!pOpt.vAllowVideo) {
          throw std::runtime_error(
              "video input is not supported - hint: if this is unexpected, you may need to provide the mmproj");
        }

        json vInputVideo = fJsonValue(vP, "input_video", json::object());
        std::string vUrl = fJsonValue(vInputVideo, "data", fJsonValue(vInputVideo, "url", std::string()));
        fHandleMedia(pOutFiles, vUrl, pOpt.vMediaPath, false, pOpt.vMediaAllowPrivateUrls);

        vP["type"] = "media_marker";
        vP["text"] = fGetMediaMarker();
        vP.erase("input_video");

      } else if (vType != "text") {
        throw std::invalid_argument("unsupported content[].type");
      }
    }
  }

  auto vCaps = fCommonChatTemplatesGetCaps(pOpt.vTmpls.get());

  common_chat_templates_inputs vInputs;
  vInputs.lMessages = fCommonChatMsgsParseOaicompat(vMessages);
  vInputs.lTools = fCommonChatToolsParseOaicompat(vTools);
  vInputs.vToolChoice = fCommonChatToolChoiceParseOaicompat(vToolChoice);
  vInputs.vJsonSchema = vJsonSchema.is_null() ? "" : vJsonSchema.dump();
  vInputs.vGrammar = vGrammar;
  vInputs.vParallelToolCalls = fJsonValue(pBody, "parallel_tool_calls", vCaps["supports_parallel_tool_calls"]);
  vInputs.vAddGenerationPrompt = fJsonValue(pBody, "add_generation_prompt", true);
  vInputs.vContinueFinalMessage = pBody.contains("continue_final_message")
                                      ? fCommonChatContinuationParse(pBody.at("continue_final_message"))
                                      : cCommonChatContinuationNone;
  if (vInputs.vContinueFinalMessage == cCommonChatContinuationNone && pOpt.vPrefillAssistant &&
      !vInputs.lMessages.empty() && vInputs.lMessages.back().vRole == "assistant") {
    if (vInputs.lMessages.size() >= 2 && vInputs.lMessages[vInputs.lMessages.size() - 2].vRole == "assistant") {
      throw std::invalid_argument("Cannot have 2 or more assistant messages at the end of the list.");
    }
    vInputs.vContinueFinalMessage = cCommonChatContinuationAuto;
    vInputs.vAddGenerationPrompt = false;
  }
  if (vInputs.vContinueFinalMessage != cCommonChatContinuationNone && vInputs.vAddGenerationPrompt) {
    throw std::invalid_argument("Cannot set both add_generation_prompt and continue_final_message to true.");
  }
  vInputs.vReasoningFormat = pOpt.vReasoningFormat;
  if (pBody.contains("reasoning_format")) {
    vInputs.vReasoningFormat = fCommonReasoningFormatFromName(pBody.at("reasoning_format").get<std::string>());
  }
  vInputs.vEnableThinking = pOpt.vEnableThinking;
  if (!vInputs.lTools.empty() && vInputs.vToolChoice != cCommonChatToolChoiceNone) {
    if (pBody.contains("grammar")) {
      throw std::invalid_argument("Cannot use custom grammar constraints with tools.");
    }
    vMgsParams["parse_tool_calls"] = true;
  }

  // merge the template args provided from command line with the args provided in the user request
  auto vChatTemplateKwargsObject = fJsonValue(pBody, "chat_template_kwargs", json::object());
  vInputs.dChatTemplateKwargs = pOpt.dChatTemplateKwargs;
  for (const auto &vItem : vChatTemplateKwargsObject.items()) {
    vInputs.dChatTemplateKwargs[vItem.key()] = vItem.value().dump();
  }

  // parse the "enable_thinking" kwarg to override the default value
  auto vEnableThinkingKwarg = fJsonValue(vInputs.dChatTemplateKwargs, "enable_thinking", std::string(""));
  if (vEnableThinkingKwarg == "true") {
    vInputs.vEnableThinking = true;
  } else if (vEnableThinkingKwarg == "false") {
    vInputs.vEnableThinking = false;
  } else if (!vEnableThinkingKwarg.empty() && vEnableThinkingKwarg[0] == '"') {
    throw std::invalid_argument("invalid type for \"enable_thinking\" (expected boolean, got string)");
  }

  // Parse also the OAI "reasoning_effort": "none" specific value
  if (pBody.contains("reasoning_effort")) {
    auto vReasoningEffort = fJsonValue(pBody, "reasoning_effort", std::string(""));
    if (vReasoningEffort == "none") {
      vInputs.vEnableThinking = false;
    } else if (vInputs.dChatTemplateKwargs.find("reasoning_effort") == vInputs.dChatTemplateKwargs.end()) {
      // other values are model-specific: hand them to the chat template (e.g. DeepSeek-V4.1 takes low,
      // medium, high, max or an integer in [1, 100])
      vInputs.dChatTemplateKwargs["reasoning_effort"] = pBody.at("reasoning_effort").dump();
    }
  }

  vInputs.vForcePureContent = pOpt.vForcePureContent;

  // Apply chat template to the list of messages
  auto vChatParams = fCommonChatTemplatesApply(pOpt.vTmpls.get(), vInputs);

  vMgsParams["chat_format"] = static_cast<int>(vChatParams.vFormat);
  vMgsParams["prompt"] = vChatParams.vPrompt;
  if (!vChatParams.vGrammar.empty()) {
    vMgsParams["grammar"] = vChatParams.vGrammar;
    vMgsParams["grammar_type"] = std::string("tool_calls");
  }
  vMgsParams["grammar_lazy"] = vChatParams.vGrammarLazy;
  auto vGrammarTriggers = json::array();
  for (const auto &vTrigger : vChatParams.lGrammarTriggers) {
    server_grammar_trigger vCt(vTrigger);
    vGrammarTriggers.push_back(vCt.to_json());
  }
  vMgsParams["grammar_triggers"] = vGrammarTriggers;
  vMgsParams["preserved_tokens"] = vChatParams.lPreservedTokens;
  vMgsParams["generation_prompt"] = vChatParams.vGenerationPrompt;
  for (const auto &vStop : vChatParams.lAdditionalStops) {
    vMgsParams["stop"].push_back(vStop);
  }
  if (!vChatParams.vParser.empty()) {
    vMgsParams["chat_parser"] = vChatParams.vParser;
  }

  vMgsParams["message_delimiters"] = vChatParams.vMessageDelimiters.to_json();

  // Reasoning budget: pass parameters through to sampling layer
  {
    int vReasoningBudget =
        fJsonValue(pBody, "reasoning_budget_tokens", -1);
    if (vReasoningBudget == -1) {
      vReasoningBudget = pOpt.vReasoningBudget;
    }

    if (!vChatParams.lThinkingEndTags.empty()) {
      vMgsParams["reasoning_budget_tokens"] = vReasoningBudget;
      vMgsParams["reasoning_budget_start_tag"] = vChatParams.vThinkingStartTag;
      vMgsParams["reasoning_budget_end_tags"] = vChatParams.lThinkingEndTags;
      vMgsParams["reasoning_budget_message"] =
          fJsonValue(pBody, "reasoning_budget_message", pOpt.vReasoningBudgetMessage);
      vMgsParams["reasoning_control"] = fJsonValue(pBody, "reasoning_control", false);
    }
  }

  // Handle "logprobs" field
  // TODO: The response format of this option is not yet OAI-compatible, but seems like no one really using it; We may
  // need to fix it in the future
  if (fJsonValue(pBody, "logprobs", false)) {
    if (vHasTools && vStream) {
      throw std::invalid_argument("logprobs is not supported with tools + stream");
    }
    vMgsParams["n_probs"] = fJsonValue(pBody, "top_logprobs", 20);
  } else if (pBody.contains("top_logprobs") && !pBody.at("top_logprobs").is_null()) {
    throw std::invalid_argument("top_logprobs requires logprobs to be set to true");
  }

  // Copy remaining properties to mgs_params
  // This allows user to use mgs-specific params like "mirostat", ... via OAI endpoint.
  // See "mLaunchSlotWithTask()" for a complete list of params supported by mgs
  for (const auto &vItem : pBody.items()) {
    // Exception: if "n_predict" is present, we overwrite the value specified earlier by "max_tokens"
    if (!vMgsParams.contains(vItem.key()) || vItem.key() == "n_predict") {
      vMgsParams[vItem.key()] = vItem.value();
    }
  }

  return vMgsParams;
}

json fFormatEmbeddingsResponseOaicompat(const json &pRequest, const std::string &pModelName, const json &pEmbeddings,
                                        bool pUseBase64) {
  json vData = json::array();
  int32_t vNTokens = 0;
  int vI = 0;
  for (const auto &vElem : pEmbeddings) {
    json vEmbeddingObj;

    if (pUseBase64) {
      const auto &vVec = fJsonValue(vElem, "embedding", json::array()).get<std::vector<float>>();
      const char *vDataPtr = reinterpret_cast<const char *>(vVec.data());
      size_t vDataSize = vVec.size() * sizeof(float);
      vEmbeddingObj = {{"embedding", base64::encode(vDataPtr, vDataSize)},
                       {"index", vI++},
                       {"object", "embedding"},
                       {"encoding_format", "base64"}};
    } else {
      vEmbeddingObj = {
          {"embedding", fJsonValue(vElem, "embedding", json::array())}, {"index", vI++}, {"object", "embedding"}};
    }
    vData.push_back(vEmbeddingObj);

    vNTokens += fJsonValue(vElem, "tokens_evaluated", 0);
  }

  json vRes = json{{"model", fJsonValue(pRequest, "model", pModelName)},
                   {"object", "list"},
                   {"usage", json{{"prompt_tokens", vNTokens}, {"total_tokens", vNTokens}}},
                   {"data", vData}};

  return vRes;
}

json fFormatResponseRerank(const json &pRequest, const std::string &pModelName, const json &pRanks, bool pIsTeiFormat,
                           std::vector<std::string> &pTexts, int pTopN) {
  int32_t vNTokens = 0;
  bool vReturnText = pIsTeiFormat && fJsonValue(pRequest, "return_text", false);
  std::vector<json> lElements; // Temporary vector to hold unsorted elements
  std::string vScoreLabel = pIsTeiFormat ? "score" : "relevance_score";
  for (const auto &vRank : pRanks) {
    int vIndex = fJsonValue(vRank, "index", 0);
    json vElem = json{
        {"index", vIndex},
        {vScoreLabel, fJsonValue(vRank, "score", 0.0)},
    };
    vNTokens += fJsonValue(vRank, "tokens_evaluated", 0);
    if (vReturnText) {
      vElem["text"] = std::move(pTexts[vIndex]);
    }
    lElements.push_back(vElem);
  }

  std::sort(lElements.begin(), lElements.end(), [vScoreLabel](const json &pA, const json &pB) {
    return fJsonValue(pA, vScoreLabel, 0.0) > fJsonValue(pB, vScoreLabel, 0.0);
  });

  lElements.resize(std::min(pTopN, (int)lElements.size()));
  json vResults = lElements;

  if (pIsTeiFormat)
    return vResults;

  json vRes = json{{"model", fJsonValue(pRequest, "model", pModelName)},
                   {"object", "list"},
                   {"usage", json{{"prompt_tokens", vNTokens}, {"total_tokens", vNTokens}}},
                   {"results", vResults}};

  return vRes;
}

//
// other utils
//

std::vector<mgs_token_data> fGetTokenProbabilities(mgs_context *pCtx, int pIdx, size_t pNTop) {
  std::vector<mgs_token_data> lCur;

  const auto *vLogits = mgs_get_logits_ith(pCtx, pIdx);
  const mgs_token *vSampledIds = mgs_get_sampled_candidates_ith(pCtx, pIdx);

  const int cNLogits = mgs_get_sampled_logits_count_ith(pCtx, pIdx);

  lCur.resize(cNLogits);
  if (vSampledIds) {
    for (int vI = 0; vI < cNLogits; vI++) {
      lCur[vI] = mgs_token_data{vSampledIds[vI], vLogits[vI], 0.0f};
    }
  } else {
    for (mgs_token vTokenId = 0; vTokenId < cNLogits; vTokenId++) {
      lCur[vTokenId] = mgs_token_data{vTokenId, vLogits[vTokenId], 0.0f};
    }
  }

  // sort tokens by logits (partial: only the leading `pNTop` need ordering)
  if (pNTop > lCur.size()) {
    pNTop = lCur.size();
  }
  if (pNTop > 0) {
    std::partial_sort(lCur.begin(), lCur.begin() + pNTop, lCur.end(),
                      [](const mgs_token_data &pA, const mgs_token_data &pB) { return pA.logit > pB.logit; });
  }

  // apply softmax
  float vMaxL = -std::numeric_limits<float>::infinity();
  if (pNTop > 0) {
    vMaxL = lCur[0].logit; // partial_sort guarantees the absolute maximum is at index 0
  } else {
    for (const auto &vT : lCur) {
      vMaxL = std::max(vMaxL, vT.logit);
    }
  }
  float vCumSum = 0.0f;
  for (auto &vT : lCur) {
    float vP = expf(vT.logit - vMaxL);
    vT.p = vP;
    vCumSum += vP;
  }
  for (auto &vT : lCur) {
    vT.p /= vCumSum;
  }

  return lCur;
}

std::string fSafeJsonToStr(const json &pData) { return pData.dump(-1, ' ', false, json::error_handler_t::replace); }

// TODO: reuse mgs_detokenize
template <class Iter> static std::string fTokensToStr(const mgs_vocab *pCtx, Iter pBegin, Iter pEnd) {
  std::string vRet;
  for (; pBegin != pEnd; ++pBegin) {
    vRet += fCommonTokenToPiece(pCtx, *pBegin);
  }

  return vRet;
}

std::string fTokensToStr(mgs_context *pCtx, const mgs_tokens &pTokens) {
  auto vModel = mgs_get_model(pCtx);
  return fTokensToStr(mgs_model_get_vocab(vModel), pTokens.begin(), pTokens.end());
}

std::string fTokensToStr(const mgs_vocab *pVocab, const mgs_tokens &pTokens) {
  return fTokensToStr(pVocab, pTokens.begin(), pTokens.end());
}

// format incomplete utf-8 multibyte character for output
std::string fTokensToOutputFormattedString(const mgs_context *pCtx, const mgs_token pToken) {
  std::string vOut = pToken == MGS_TOKEN_NULL ? "" : fCommonTokenToPiece(pCtx, pToken);

  // if the size is 1 and first bit is 1, meaning it's a partial character
  //   (size > 1 meaning it's already a known token)
  if (vOut.size() == 1 && (vOut[0] & 0x80) == 0x80) {
    std::stringstream vSs;
    vSs << std::hex << (vOut[0] & 0xff);
    std::string vRes(vSs.str());
    vOut = "byte: \\x" + vRes;
  }

  return vOut;
}

// format server-sent event (SSE), return the formatted string to send
// note: if data is a json array, it will be sent as multiple events, one per item
std::string fFormatOaiSse(const json &pData) {
  std::ostringstream vSs;
  auto vSendSingle = [&vSs](const json &pData) {
    vSs << "data: " << fSafeJsonToStr(pData)
        << "\n\n"; // required by RFC 8895 - A message is terminated by a blank line (two line terminators in a row).
  };

  if (pData.is_array()) {
    for (const auto &vItem : pData) {
      vSendSingle(vItem);
    }
  } else {
    vSendSingle(pData);
  }

  return vSs.str();
}

std::string fFormatOaiRespSse(const json &pData) {
  std::ostringstream vSs;
  auto vSendSingle = [&vSs](const json &pEventObj) {
    vSs << "event: " << pEventObj.at("event").get<std::string>() << "\n";
    vSs << "data: " << fSafeJsonToStr(pEventObj.at("data")) << "\n\n";
  };

  if (pData.is_array()) {
    for (const auto &vItem : pData) {
      vSendSingle(vItem);
    }
  } else {
    vSendSingle(pData);
  }

  return vSs.str();
}

std::string fFormatAnthropicSse(const json &pData) {
  std::ostringstream vSs;

  auto vSendEvent = [&vSs](const json &pEventObj) {
    if (pEventObj.contains("event") && pEventObj.contains("data")) {
      vSs << "event: " << pEventObj.at("event").get<std::string>() << "\n";
      vSs << "data: " << fSafeJsonToStr(pEventObj.at("data")) << "\n\n";
    } else {
      vSs << "data: " << fSafeJsonToStr(pEventObj) << "\n\n";
    }
  };

  if (pData.is_array()) {
    for (const auto &vEvent : pData) {
      vSendEvent(vEvent);
    }
  } else {
    vSendEvent(pData);
  }

  return vSs.str();
}

bool fIsValidUtf8(const std::string &pStr) {
  const unsigned char *vBytes = reinterpret_cast<const unsigned char *>(pStr.data());
  const unsigned char *vEnd = vBytes + pStr.length();

  while (vBytes < vEnd) {
    if (*vBytes <= 0x7F) {
      // 1-byte sequence (0xxxxxxx)
      vBytes++;
    } else if ((*vBytes & 0xE0) == 0xC0) {
      // 2-byte sequence (110xxxxx 10xxxxxx)
      if (vEnd - vBytes < 2 || (vBytes[1] & 0xC0) != 0x80)
        return false;
      vBytes += 2;
    } else if ((*vBytes & 0xF0) == 0xE0) {
      // 3-byte sequence (1110xxxx 10xxxxxx 10xxxxxx)
      if (vEnd - vBytes < 3 || (vBytes[1] & 0xC0) != 0x80 || (vBytes[2] & 0xC0) != 0x80)
        return false;
      vBytes += 3;
    } else if ((*vBytes & 0xF8) == 0xF0) {
      // 4-byte sequence (11110xxx 10xxxxxx 10xxxxxx 10xxxxxx)
      if (vEnd - vBytes < 4 || (vBytes[1] & 0xC0) != 0x80 || (vBytes[2] & 0xC0) != 0x80 || (vBytes[3] & 0xC0) != 0x80)
        return false;
      vBytes += 4;
    } else {
      // Invalid UTF-8 lead byte
      return false;
    }
  }

  return true;
}

mgs_tokens fFormatPromptInfill(const mgs_vocab *pVocab, const json &pInputPrefix, const json &pInputSuffix,
                               const json &pInputExtra, const int pNBatch, const int pNPredict, const int pNCtx,
                               const bool pSpmInfill, const mgs_tokens &pTokensPrompt) {
  // TODO: optimize this block by reducing memory allocations and movement

  // use FIM repo-level pattern:
  // ref: https://arxiv.org/pdf/2409.12186
  //
  // [FIM_REP]myproject
  // [FIM_SEP]filename0
  // extra chunk 0
  // [FIM_SEP]filename1
  // extra chunk 1
  // ...
  // [FIM_SEP]filename
  // [FIM_PRE]prefix[FIM_SUF]suffix[FIM_MID]prompt
  //
  mgs_tokens vExtraTokens;
  vExtraTokens.reserve(pNCtx);

  auto vTokensPrefix = fTokenizeMixed(pVocab, pInputPrefix, false, false);
  auto vTokensSuffix = fTokenizeMixed(pVocab, pInputSuffix, false, false);

  if (mgs_vocab_fim_rep(pVocab) != MGS_TOKEN_NULL) {
    // TODO: make project name an input
    static const auto cKFimRepo = fCommonTokenize(pVocab, "myproject\n", false, false);

    vExtraTokens.push_back(mgs_vocab_fim_rep(pVocab));
    vExtraTokens.insert(vExtraTokens.end(), cKFimRepo.begin(), cKFimRepo.end());
  }
  for (const auto &vChunk : pInputExtra) {
    // { "text": string, "filename": string }
    const std::string cText = fJsonValue(vChunk, "text", std::string());
    const std::string cFilename = fJsonValue(vChunk, "filename", std::string("tmp"));

    if (mgs_vocab_fim_sep(pVocab) != MGS_TOKEN_NULL) {
      const auto cKFimFile = fCommonTokenize(pVocab, cFilename + "\n", false, false);

      vExtraTokens.insert(vExtraTokens.end(), mgs_vocab_fim_sep(pVocab));
      vExtraTokens.insert(vExtraTokens.end(), cKFimFile.begin(), cKFimFile.end());
    } else {
      // chunk separator in binary form to avoid confusing the AI
      static const char aKChunkPrefixStr[] = {0x0a, 0x0a, 0x2d, 0x2d, 0x2d, 0x20, 0x73, 0x6e, 0x69, 0x70,
                                              0x70, 0x65, 0x74, 0x20, 0x2d, 0x2d, 0x2d, 0x0a, 0x0a, 0x00};
      static const auto cKChunkPrefixTokens = fCommonTokenize(pVocab, aKChunkPrefixStr, false, false);

      vExtraTokens.insert(vExtraTokens.end(), cKChunkPrefixTokens.begin(), cKChunkPrefixTokens.end());
    }

    const auto cChunkTokens = fCommonTokenize(pVocab, cText, false, false);
    vExtraTokens.insert(vExtraTokens.end(), cChunkTokens.begin(), cChunkTokens.end());
  }

  if (mgs_vocab_fim_sep(pVocab) != MGS_TOKEN_NULL) {
    // TODO: current filename
    static const auto cKFimFile = fCommonTokenize(pVocab, "filename\n", false, false);

    vExtraTokens.insert(vExtraTokens.end(), mgs_vocab_fim_sep(pVocab));
    vExtraTokens.insert(vExtraTokens.end(), cKFimFile.begin(), cKFimFile.end());
  }

  // for now pick FIM context to fit in a batch (ratio prefix:suffix = 3:1, TODO: configurable?)
  const int cNPrefixTake = std::min<int>(vTokensPrefix.size(), 3 * (pNBatch / 4));
  const int cNSuffixTake =
      std::min<int>(vTokensSuffix.size(), std::max<int>(0, (pNBatch / 4) - (2 + pTokensPrompt.size())));

  SRV_DBG("n_prefix_take = %d, n_suffix_take = %d, total = %d\n", cNPrefixTake, cNSuffixTake,
          (cNPrefixTake + cNSuffixTake));

  // fill the rest of the context with extra chunks
  const int cNExtraTake = std::min<int>(std::max<int>(0, pNCtx - (pNBatch)-2 * pNPredict), vExtraTokens.size());

  vTokensPrefix.erase(vTokensPrefix.begin(), vTokensPrefix.begin() + vTokensPrefix.size() - cNPrefixTake);
  vTokensSuffix.resize(cNSuffixTake);

  vTokensPrefix.insert(vTokensPrefix.begin(), mgs_vocab_fim_pre(pVocab));
  vTokensPrefix.insert(vTokensPrefix.end(), pTokensPrompt.begin(), pTokensPrompt.end());
  vTokensSuffix.insert(vTokensSuffix.begin(), mgs_vocab_fim_suf(pVocab));

  auto vEmbdInp = pSpmInfill ? vTokensSuffix : vTokensPrefix;
  auto vEmbdEnd = pSpmInfill ? vTokensPrefix : vTokensSuffix;

  if (mgs_vocab_get_add_bos(pVocab)) {
    vEmbdInp.insert(vEmbdInp.begin(), mgs_vocab_bos(pVocab));
  }

  SRV_DBG("extra: n_ctx = %d, n_extra_take = %d, n_extra = %d\n", pNCtx, cNExtraTake, (int)vExtraTokens.size());

  // put the extra context before the FIM prefix
  vEmbdInp.insert(vEmbdInp.begin(), vExtraTokens.end() - cNExtraTake, vExtraTokens.end());

  vEmbdInp.insert(vEmbdInp.end(), vEmbdEnd.begin(), vEmbdEnd.end());
  vEmbdInp.push_back(mgs_vocab_fim_mid(pVocab));

  return vEmbdInp;
}

server_tokens fFormatPromptRerank(const struct mgs_model *pModel, const struct mgs_vocab *pVocab, mtmd_context *pMctx,
                                  const std::string &pQuery, const std::string &pDoc) {
  server_tokens vResult = {};

  const char *vRerankPrompt = mgs_model_chat_template(pModel, "rerank");

  if (vRerankPrompt != nullptr) {
    std::string vPrompt = vRerankPrompt;
    fStringReplaceAll(vPrompt, "{query}", pQuery);
    fStringReplaceAll(vPrompt, "{document}", pDoc);
    server_tokens vTokens = fTokenizeInputSubprompt(pVocab, pMctx, vPrompt, false, true);
    vResult.mPushBack(vTokens);
  } else {
    // Get EOS token - use SEP token as fallback if EOS is not available
    server_tokens vQueryTokens = fTokenizeInputSubprompt(pVocab, pMctx, pQuery, false, false);
    server_tokens vDocTokens = fTokenizeInputSubprompt(pVocab, pMctx, pDoc, false, false);
    mgs_token vEosToken = mgs_vocab_eos(pVocab);
    if (vEosToken == MGS_TOKEN_NULL) {
      vEosToken = mgs_vocab_sep(pVocab);
    }

    if (mgs_vocab_get_add_bos(pVocab)) {
      vResult.mPushBack(mgs_vocab_bos(pVocab));
    }
    vResult.mPushBack(vQueryTokens);
    if (mgs_vocab_get_add_eos(pVocab)) {
      vResult.mPushBack(vEosToken);
    }
    if (mgs_vocab_get_add_sep(pVocab)) {
      vResult.mPushBack(mgs_vocab_sep(pVocab));
    }
    vResult.mPushBack(vDocTokens);
    if (mgs_vocab_get_add_eos(pVocab)) {
      vResult.mPushBack(vEosToken);
    }
  }

  return vResult;
}
