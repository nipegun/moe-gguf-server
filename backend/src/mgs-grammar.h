#pragma once

#include "mgs.h"

#include <map>
#include <regex>
#include <string>
#include <vector>

struct mgs_vocab;

// grammar element type
enum mgs_gretype {
  // end of rule definition
  cMgsGretypeEnd = 0,

  // start of alternate definition for rule
  cMgsGretypeAlt = 1,

  // non-terminal element: reference to rule
  cMgsGretypeRuleRef = 2,

  // terminal element: character (code point)
  cMgsGretypeChar = 3,

  // inverse char(s) ([^a], [^a-b] [^abc])
  cMgsGretypeCharNot = 4,

  // modifies a preceding MGS_GRETYPE_CHAR or MGS_GRETYPE_CHAR_ALT to
  // be an inclusive range ([a-z])
  cMgsGretypeCharRngUpper = 5,

  // modifies a preceding MGS_GRETYPE_CHAR or
  // MGS_GRETYPE_CHAR_RNG_UPPER to add an alternate char to match ([ab], [a-zA])
  cMgsGretypeCharAlt = 6,

  // any character (.)
  cMgsGretypeCharAny = 7,

  // terminal element: token (<[token-id]>)
  cMgsGretypeToken = 8,

  // inverse token (!<[token-id]>)
  cMgsGretypeTokenNot = 9,
};

typedef struct mgs_grammar_element {
  enum mgs_gretype vType;
  uint32_t vValue; // Unicode code point, rule ID, or token ID
} mgs_grammar_element;

struct mgs_partial_utf8 {
  uint32_t vValue; // bit value so far (unshifted)
  int vNRemain;    // num bytes remaining; -1 indicates invalid sequence
};

struct mgs_grammar_candidate {
  size_t vIndex;
  const uint32_t *vCodePoints;
  mgs_partial_utf8 vPartialUtf8;
  mgs_token vId;
};

using mgs_grammar_rule = std::vector<mgs_grammar_element>;
using mgs_grammar_stack = std::vector<const mgs_grammar_element *>;

using mgs_grammar_rules = std::vector<mgs_grammar_rule>;
using mgs_grammar_stacks = std::vector<mgs_grammar_stack>;
using mgs_grammar_candidates = std::vector<mgs_grammar_candidate>;

// TODO: remove, needed for tests atm
const mgs_grammar_rules &mgs_grammar_get_rules(const struct mgs_grammar *pGrammar);
mgs_grammar_stacks &mgs_grammar_get_stacks(struct mgs_grammar *pGrammar);

// takes a set of possible pushdown stacks on a grammar, which are required to
// be positioned at a character range (see `fMgsGrammarAdvanceStack`), and
// produces the N possible stacks if the given char is accepted at those
// positions
void mgs_grammar_accept(struct mgs_grammar *pGrammar, uint32_t pChr);

std::vector<mgs_grammar_candidate> mgs_grammar_reject_candidates_for_stack(const mgs_grammar_rules &pRules,
                                                                           const mgs_grammar_stack &pStack,
                                                                           const mgs_grammar_candidates &pCandidates);

struct mgs_grammar_parser {
  const mgs_vocab *vVocab;
  std::map<std::string, uint32_t> dSymbolIds;

  mgs_grammar_rules vRules;

  mgs_grammar_parser(const struct mgs_vocab *pVocab = nullptr) : vVocab(pVocab) {}

  mgs_grammar_stack mCRules() const;

  uint32_t mGetSymbolId(const char *pSrc, size_t pLen);
  uint32_t mGenerateSymbolId(const std::string &pBaseName);

  void mAddRule(uint32_t pRuleId, const mgs_grammar_rule &pRule);

  const char *mParseAlternates(const char *pSrc, const std::string &pRuleName, uint32_t pRuleId, bool pIsNested);

  const char *mParseSequence(const char *pSrc, const std::string &pRuleName, mgs_grammar_rule &pRule, bool pIsNested);

  const char *mParseRule(const char *pSrc);

  bool mParse(const char *pSrc);
  void mPrint(FILE *pFile);
};

struct mgs_grammar_trigger_pattern {
  std::string vPattern;
  std::regex vRegex;

  size_t mFind(const std::string &pInput) const;
};

struct mgs_grammar {
  // maintain a list of mgs_tokens and their positions in the trigger_buffer
  using token_pos = std::pair<mgs_token, std::pair<size_t, size_t>>;

  // note: allow null vocab for testing (not great)
  const mgs_vocab *vVocab;

  const mgs_grammar_rules cRules; // TODO: shared ptr
  mgs_grammar_stacks vStacks;

  // buffer for partially generated UTF-8 sequence from accepted tokens
  mgs_partial_utf8 vPartialUtf8;

  // lazy grammars wait for trigger words or tokens before constraining the sampling.
  // we still have trigger_tokens for non-lazy grammars to force printing of special trigger tokens.
  // (useful e.g. for tool_choice=required)
  bool vLazy = false;
  bool vAwaitingTrigger = false; // Initialized to true for lazy grammars only
  std::string vTriggerBuffer;    // Output buffered by lazy grammar. Will be cleared once trigger is found.
  std::vector<token_pos>
      vTriggerBufferPositions; // Tokens buffered by lazy grammar. Used to replay when a trigger is found.
  std::vector<mgs_token>
      vTriggerTokens; // Tokens that trigger a lazy grammar, or tokens to force printing of (even if special).
  std::vector<mgs_grammar_trigger_pattern>
      vTriggerPatterns; // Regular expressions that trigger a lazy grammar. Must be a full match of the entire generated
                        // string, and the grammar will be given the string from the first match group onwards.
};

//
// internal API
//

// note: needed for tests (not great)
struct mgs_grammar *mgs_grammar_init_impl(const struct mgs_vocab *pVocab, const mgs_grammar_element **pRules,
                                          size_t pNRules, size_t pStartRuleIndex);

struct mgs_grammar *mgs_grammar_init_impl(const struct mgs_vocab *pVocab, const char *pGrammarStr,
                                          const char *pGrammarRoot, bool pLazy, const char **pTriggerPatterns,
                                          size_t pNumTriggerPatterns, const mgs_token *pTriggerTokens,
                                          size_t pNumTriggerTokens);

void mgs_grammar_free_impl(struct mgs_grammar *pGrammar);

struct mgs_grammar *mgs_grammar_clone_impl(const struct mgs_grammar &pGrammar);

// TODO: move the API below as member functions of mgs_grammar
void mgs_grammar_apply_impl(const struct mgs_grammar &pGrammar, mgs_token_data_array *pCurP);

void mgs_grammar_accept_impl(struct mgs_grammar &pGrammar, mgs_token pToken);

void mgs_grammar_accept_str(struct mgs_grammar &pGrammar, const std::string &pPiece);

void mgs_grammar_accept_token(struct mgs_grammar &pGrammar, mgs_token pToken, const std::string &pPiece);
