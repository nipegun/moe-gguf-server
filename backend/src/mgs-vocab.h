#pragma once

#include "mgs.h"

#include <string>
#include <vector>
#include <memory>

// pre-tokenization types
enum mgs_vocab_pre_type {
  cMgsVocabPreTypeDefault = 0,
  cMgsVocabPreTypeLlama3 = 1,
  cMgsVocabPreTypeFalcon = 4,
  cMgsVocabPreTypeMpt = 5,
  cMgsVocabPreTypeStarcoder = 6,
  cMgsVocabPreTypeGpt2 = 7,
  cMgsVocabPreTypeRefact = 8,
  cMgsVocabPreTypeCommandR = 9,
  cMgsVocabPreTypeStablelm2 = 10,
  cMgsVocabPreTypeQwen2 = 11,
  cMgsVocabPreTypeOlmo = 12,
  cMgsVocabPreTypeDbrx = 13,
  cMgsVocabPreTypeSmaug = 14,
  cMgsVocabPreTypePoro = 15,
  cMgsVocabPreTypeChatglm3 = 16,
  cMgsVocabPreTypeChatglm4 = 17,
  cMgsVocabPreTypeViking = 18,
  cMgsVocabPreTypeJais = 19,
  cMgsVocabPreTypeTekken = 20,
  cMgsVocabPreTypeSmollm = 21,
  cMgsVocabPreTypeCodeshell = 22,
  cMgsVocabPreTypeBloom = 23,
  cMgsVocabPreTypeGpt3Finnish = 24,
  cMgsVocabPreTypeExaone = 25,
  cMgsVocabPreTypeChameleon = 26,
  cMgsVocabPreTypeMinerva = 27,
  cMgsVocabPreTypeDeepseek3Llm = 28,
  cMgsVocabPreTypeGpt4o = 29,
  cMgsVocabPreTypeSuperbpe = 30,
  cMgsVocabPreTypeTrillion = 31,
  cMgsVocabPreTypeBailingmoe = 32,
  cMgsVocabPreTypeLlama4 = 33,
  cMgsVocabPreTypePixtral = 34,
  cMgsVocabPreTypeSeedCoder = 35,
  cMgsVocabPreTypeHunyuan = 36,
  cMgsVocabPreTypeKimiK2 = 37,
  cMgsVocabPreTypeHunyuanDense = 38,
  cMgsVocabPreTypeGrok2 = 39,
  cMgsVocabPreTypeGraniteDocling = 40,
  cMgsVocabPreTypeMinimaxM2 = 41,
  cMgsVocabPreTypeAfmoe = 42,
  cMgsVocabPreTypeSolarOpen = 43,
  cMgsVocabPreTypeYoutu = 44,
  cMgsVocabPreTypeExaoneMoe = 45,
  cMgsVocabPreTypeQwen35 = 46,
  cMgsVocabPreTypeTinyAya = 47,
  cMgsVocabPreTypeJoyaiLlm = 48,
  cMgsVocabPreTypeJais2 = 49,
  cMgsVocabPreTypeSarvamMoe = 51,
  cMgsVocabPreTypeMinicpm5 = 52,
  cMgsVocabPreTypeWhitespace = 53,
  cMgsVocabPreTypeGraniteEmbMulti = 54,
  cMgsVocabPreTypeMellum2 = 55,
  cMgsVocabPreTypeLaguna = 56,
};

struct LLM_KV;
struct mgs_model_loader;

struct mgs_vocab {
  struct vTokenData {
    std::string vText;
    float vScore;
    mgs_token_attr vAttr;
  };

  struct normalizer_options {
    bool vLowercase = true;
    bool vStripAccents = true;
    // TODO: clean_text, handle_chinese_chars
  };

  mgs_vocab();
  ~mgs_vocab();

  void mLoad(mgs_model_loader &pMl, const LLM_KV &pKv);

  std::string mGetTokenizerModel() const;
  std::string mGetTokenizerPre() const;

  enum mgs_vocab_type mGetType() const;
  enum mgs_vocab_pre_type mGetPreType() const;

  uint32_t mNTokens() const;
  uint32_t mNTokenTypes() const;

  std::string mTypeName() const;

  bool mIsNormal(mgs_token pId) const;
  bool mIsUnknown(mgs_token pId) const;
  bool mIsControl(mgs_token pId) const;
  bool mIsByte(mgs_token pId) const;
  bool mIsUserDefined(mgs_token pId) const;
  bool mIsUnused(mgs_token pId) const;
  bool mIsEog(mgs_token pId) const;

  uint8_t mTokenToByte(mgs_token pId) const;
  mgs_token mByteToToken(uint8_t pCh) const;

  mgs_token mTextToToken(const std::string &pText) const;

  const vTokenData &mGetTokenData(mgs_token pId) const;

  const char *mTokenGetText(mgs_token pId) const;
  float mTokenGetScore(mgs_token pId) const;
  mgs_token_attr mTokenGetAttr(mgs_token pId) const;

  mgs_token mTokenBos() const;
  mgs_token mTokenEos() const;
  mgs_token mTokenEot() const;
  mgs_token mTokenEom() const;
  mgs_token mTokenUnk() const;
  mgs_token mTokenSep() const;
  mgs_token mTokenNl() const;
  mgs_token mTokenPad() const;
  mgs_token mTokenMask() const;

  mgs_token mTokenPrefix() const;
  mgs_token mTokenMiddle() const;
  mgs_token mTokenSuffix() const;

  mgs_token mTokenFimPre() const;
  mgs_token mTokenFimSuf() const;
  mgs_token mTokenFimMid() const;
  mgs_token mTokenFimPad() const;
  mgs_token mTokenFimRep() const;
  mgs_token mTokenFimSep() const;

  bool mGetAddSpacePrefix() const;
  bool mGetAddBos() const;
  bool mGetAddEos() const;
  bool mGetAddSep() const;
  bool mGetIgnoreMerges() const;
  bool mGetCleanSpaces() const;
  bool mGetRemoveExtraWhitespaces() const;
  bool mGetEscapeWhitespaces() const;
  bool mGetTreatWhitespaceAsSuffix() const;
  const normalizer_options &mGetNormalizerOpts() const;

  const std::vector<mgs_token> &mGetSuppressTokens() const;

  int mMaxTokenLen() const;

  int mFindBpeRank(const std::string &pTokenLeft, const std::string &pTokenRight) const;
  std::vector<std::string> mGetBpeMerges() const;

  std::vector<char> mGetPrecompiledCharsmap() const;

  int32_t mTokenize(const char *pText, int32_t pTextLen, mgs_token *pTokens, int32_t pNTokensMax, bool pAddSpecial,
                    bool pParseSpecial) const;

  std::vector<mgs_token> mTokenize(const std::string &pRawText, bool pAddSpecial, bool pParseSpecial = false) const;

  // does not write null-terminator to buf
  int32_t mTokenToPiece(mgs_token pToken, char *pBuf, int32_t pLength, int32_t pLstrip, bool pSpecial) const;

  // use cached data
  const std::string &mTokenToPiece(mgs_token pToken) const;

  int32_t mDetokenize(const mgs_token *pTokens, int32_t pNTokens, char *pText, int32_t pTextLenMax, bool pRemoveSpecial,
                      bool pUnparseSpecial) const;

  std::string mDetokenize(const std::vector<mgs_token> &pTokens, bool pSpecial) const;

  void mPrintInfo() const;

private:
  struct impl;
  std::unique_ptr<impl> vPimpl;
};
