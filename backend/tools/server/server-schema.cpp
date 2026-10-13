#include "server-schema.h"

#include "json-schema-to-grammar.h"

namespace server_schema {

//
// mgs-specific completion schema
//

std::vector<std::unique_ptr<field>> fMakeMgsCmplSchema(const common_params &pParamsBase, task_params &pParams) {
  std::vector<std::unique_ptr<field>> lFields;
  auto vAdd = [&](field *pF) { lFields.emplace_back(pF); };

  vAdd((new field_bool("verbose", pParams.vVerbose))
           ->mSetDesc("Include __verbose field in the response with additional debug information"));

  vAdd((new field_bool("timings_per_token", pParams.vTimingsPerToken))
           ->mSetDesc("Include prompt processing and text generation speed information in each response"));

  vAdd((new field_bool("stream", pParams.vStream))
           ->mSetDesc(
               "Allows receiving each predicted token in real-time instead of waiting for the completion to finish"));

  vAdd((new field_nested("stream_options"))
           ->mAddSubfield((new field_bool("include_usage", pParams.vIncludeUsage))
                              ->mSetDesc("Whether to include usage information in the stream"))
           ->mSetDesc("Additional options for streaming responses"));

  vAdd(
      (new field_bool("cache_prompt", pParams.vCachePrompt))
          ->mSetDesc("Re-use KV cache from a previous request if possible. This way the common prefix does not have to "
                     "be re-processed, only the suffix that differs between the requests"));

  vAdd((new field_bool("return_tokens", pParams.vReturnTokens))
           ->mSetDesc("Return the raw generated token ids in the `tokens` field"));

  vAdd((new field_bool("return_progress", pParams.vReturnProgress))
           ->mSetDesc("Include prompt processing progress events in stream mode"));

  vAdd(
      (new field_num("sse_ping_interval", pParams.vSsePingInterval))
          ->mSetHardLimits(-1, INT32_MAX)
          ->mSetDesc("Interval in seconds between SSE comment pings emitted while the stream stays silent, -1 disables "
                     "pings"));

  vAdd((new field_str("mgs_client"))
           ->mSetDesc("Origin of the request. The bundled web interface sends \"webui\"; its streamed OpenAI-style "
                      "requests then receive {\"mgs_queue\": {\"ahead\", \"ahead_api\", \"ahead_webui\"}} events "
                      "while they wait for a free slot (all zero when the request starts). Other clients omit it "
                      "and never receive those events")
           ->mSetHandler([&](field_eval_context &pCtx, const json &pData) {
             pCtx.vParams.vFromWebui = pData.at("mgs_client").get<std::string>() == "webui";
           }));

  vAdd((new field_num("n_predict", pParams.vNPredict))
           ->mSetHardLimits(-1, INT32_MAX)
           ->mAddAlias("max_completion_tokens")
           ->mAddAlias("max_tokens")
           ->mSetDesc("Set the maximum number of tokens to predict. When 0, no tokens will be generated but the prompt "
                      "is evaluated into the cache"));

  vAdd((new field_num("n_indent", pParams.vNIndent))
           ->mSetHardLimits(0, INT32_MAX)
           ->mSetDesc("Specify the minimum line indentation for the generated text in number of whitespace characters. "
                      "Useful for code completion tasks"));

  vAdd((new field_num("n_keep", pParams.vNKeep))
           ->mSetHardLimits(-1, INT32_MAX)
           ->mSetDesc("Specify the number of tokens from the initial prompt to retain when context size is exceeded. "
                      "Use -1 to retain all tokens from the prompt"));

  vAdd((new field_num("n_discard", pParams.vNDiscard))
           ->mSetHardLimits(0, INT32_MAX)
           ->mSetDesc("Number of tokens after n_keep that may be discarded when shifting context (0 = half context)"));

  vAdd((new field_num("n_cmpl", pParams.vNCmpl))
           ->mSetHardLimits(1, pParamsBase.vNParallel)
           ->mAddAlias("n") // alias "n" as fallback (OpenAI completions API)
           ->mSetDesc("Number of completions to generate. If the input has multiple prompts, total outputs will be N "
                      "prompts times n_cmpl"));

  vAdd((new field_num("n_cache_reuse", pParams.vNCacheReuse))
           ->mSetHardLimits(0, INT32_MAX)
           ->mSetDesc("Min chunk size to attempt reusing from the cache via KV shifting. See --cache-reuse arg"));

  // TODO: implement t_max_prompt_ms
  // vAdd((new field_num("t_max_prompt_ms", pParams.vTMaxPromptMs))

  vAdd((new field_num("t_max_predict_ms", pParams.vTMaxPredictMs))
           ->mSetHardLimits(-1, std::numeric_limits<int64_t>::max())
           ->mSetDesc("Set a time limit in milliseconds for the prediction phase. The timeout triggers if generation "
                      "exceeds this time (measured since the first token) and a newline has been generated. Useful for "
                      "FIM applications"));

  vAdd((new field_json("response_fields"))
           ->mSetDesc("A list of response fields to return. Missing fields are omitted without error. Fields with a "
                      "slash are unnested (e.g. generation_settings/n_predict moves n_predict to the root)")
           ->mSetHandler([&](field_eval_context &pCtx, const json &pData) {
             pCtx.vParams.lResponseFields = fJsonValue(pData, "response_fields", std::vector<std::string>());
           }));

  //
  // Sampling params
  //

  vAdd((new field_num("top_k", pParams.vSampling.vTopK))
           ->mSetLimits(0, INT32_MAX)
           ->mSetDesc("Limit the next token selection to the K most probable tokens (0 = disabled)"));

  vAdd(
      (new field_num("top_p", pParams.vSampling.vTopP))
          ->mSetLimits(0.0f, 1.0f)
          ->mSetDesc("Limit the next token selection to a subset of tokens with cumulative probability above threshold "
                     "P (1.0 = disabled)"));

  vAdd((new field_num("min_p", pParams.vSampling.vMinP))
           ->mSetLimits(0.0f, 1.0f)
           ->mSetDesc("The minimum probability for a token to be considered, relative to the probability of the most "
                      "likely token (0 = disabled)"));

  vAdd((new field_num("top_n_sigma", pParams.vSampling.vTopNSigma))
           ->mSetDesc("Keep tokens within n standard deviations of the top token logit (< 0 = disabled)"));

  vAdd((new field_num("xtc_probability", pParams.vSampling.vXtcProbability))
           ->mSetLimits(0.0f, 1.0f)
           ->mSetDesc("Set the chance for token removal via XTC sampler (0 = disabled)"));

  vAdd((new field_num("xtc_threshold", pParams.vSampling.vXtcThreshold))
           ->mSetLimits(0.0f, 1.0f)
           ->mSetDesc(
               "Set a minimum probability threshold for tokens to be removed via XTC sampler (> 0.5 disables XTC)"));

  vAdd((new field_num("typical_p", pParams.vSampling.vTypP))
           // ->mSetLimits(0.0f, 1.0f) // what's the valid range?
           ->mSetDesc("Enable locally typical sampling with parameter p (1.0 = disabled)"));

  vAdd((new field_num("temperature", pParams.vSampling.vTemp))
           ->mSetLimits(0.0f, std::numeric_limits<float>::infinity())
           ->mSetDesc("Adjust the randomness of the generated text (0 = greedy)"));

  vAdd((new field_num("dynatemp_range", pParams.vSampling.vDynatempRange))
           ->mSetDesc("Dynamic temperature range. The final temperature will be in [temperature - range, temperature + "
                      "range] (0 = disabled)"));

  vAdd((new field_num("dynatemp_exponent", pParams.vSampling.vDynatempExponent))
           ->mSetDesc("Dynamic temperature exponent, controls how entropy maps to temperature"));

  vAdd((new field_num("repeat_last_n", pParams.vSampling.vPenaltyLastN))
           ->mSetHardLimits(-1, INT32_MAX)
           ->mSetDesc("Last n tokens to consider for penalizing repetition (0 = disabled, -1 = ctx-size)"));

  vAdd((new field_num("repeat_penalty", pParams.vSampling.vPenaltyRepeat))
           ->mSetDesc("Control the repetition of token sequences in the generated text (1.0 = disabled)"));

  vAdd((new field_num("frequency_penalty", pParams.vSampling.vPenaltyFreq))
           ->mSetDesc("Repeat alpha frequency penalty (0 = disabled)"));

  vAdd((new field_num("presence_penalty", pParams.vSampling.vPenaltyPresent))
           ->mSetDesc("Repeat alpha presence penalty (0 = disabled)"));

  vAdd((new field_num("dry_multiplier", pParams.vSampling.vDryMultiplier))
           ->mSetDesc("Set the DRY (Don't Repeat Yourself) repetition penalty multiplier (0 = disabled)"));

  vAdd((new field_num("dry_base", pParams.vSampling.vDryBase))
           ->mSetDesc("Set the DRY repetition penalty base value (must be >= 1.0, any values < 1.0 will be replaced "
                      "with the default value)")
           ->mSetHandler([&](field_eval_context &pCtx, const json &pData) {
             float vV = pData.at("dry_base").get<float>();
             pCtx.vParams.vSampling.vDryBase = (vV < 1.0f) ? pParamsBase.vSampling.vDryBase : vV;
           }));

  vAdd((new field_num("dry_allowed_length", pParams.vSampling.vDryAllowedLength))
           ->mSetHardLimits(0, INT32_MAX)
           ->mSetDesc("Tokens that extend repetition beyond this length receive exponentially increasing penalty: "
                      "multiplier * base ^ (sequence_length - allowed_length)"));

  vAdd((new field_num("dry_penalty_last_n", pParams.vSampling.vDryPenaltyLastN))
           ->mSetHardLimits(-1, INT32_MAX)
           ->mSetDesc("How many tokens to scan for repetitions (0 = disabled, -1 = context size)"));

  vAdd((new field_num("mirostat", pParams.vSampling.vMirostat))
           ->mSetLimits(0, 2)
           ->mSetDesc("Enable Mirostat sampling, controlling perplexity during text generation (0 = disabled, 1 = "
                      "Mirostat, 2 = Mirostat 2.0)"));

  vAdd((new field_num("mirostat_tau", pParams.vSampling.vMirostatTau))
           ->mSetDesc("Set the Mirostat target entropy, parameter tau"));

  vAdd((new field_num("mirostat_eta", pParams.vSampling.vMirostatEta))
           ->mSetDesc("Set the Mirostat learning rate, parameter eta"));

  vAdd((new field_num("adaptive_target", pParams.vSampling.vAdaptiveTarget))
           ->mSetLimits(-std::numeric_limits<float>::max(), 1.0f)
           ->mSetDesc("Adaptive sampling target entropy (valid range 0.0 to 1.0; negative = disabled)"));

  vAdd((new field_num("adaptive_decay", pParams.vSampling.vAdaptiveDecay))
           ->mSetHardLimits(0.0f, 0.99f)
           ->mSetDesc("EMA decay for adaptive sampling; history approximates 1/(1-decay) tokens"));

  // seed is uint32_t; field_num uses int32_t so use a handler
  vAdd((new field_num("seed", pParams.vSampling.vSeed))
           ->mSetDesc("Set the random number generator (RNG) seed (-1 = random)"));

  vAdd((new field_num("n_probs", pParams.vSampling.vNProbs))
           ->mAddAlias("logprobs") // use "logprobs" if "n_probs" wasn't provided
           ->mSetDesc("If greater than 0, output the probabilities of top N tokens for each generated token"));

  vAdd((new field_num("min_keep", pParams.vSampling.vMinKeep))
           ->mSetHardLimits(0, INT32_MAX)
           ->mSetDesc("If greater than 0, force samplers to return at least N possible tokens"));

  vAdd((new field_bool("backend_sampling", pParams.vSampling.vBackendSampling))
           ->mSetDesc("Use backend sampling instead of mgs sampling"));

  vAdd((new field_bool("post_sampling_probs", pParams.vPostSamplingProbs))
           ->mSetDesc("Return probabilities of top n_probs tokens after applying the sampling chain"));

  //
  // Speculative decoding params
  //

  // TODO: to keep things simple, we disable speculative parameter adjustments for now
#if 0
  // TODO: for now, be able to adjust only the draft-model based speculative parameters
  vAdd((new field_num("speculative.n_max", pParams.vSpeculative.vDraft.vNMax))
           ->mSetHardLimits(0, INT32_MAX)
           ->mSetDesc("Maximum number of tokens to draft during speculative decoding"));

  vAdd((new field_num("speculative.n_min", pParams.vSpeculative.vDraft.vNMin))
           ->mSetHardLimits(0, INT32_MAX)
           ->mSetDesc("Minimum number of draft tokens to use for speculative decoding"));

  vAdd((new field_num("speculative.p_min", pParams.vSpeculative.vDraft.vPMin))
           ->mSetHardLimits(0.0f, 1.0f)
           ->mSetDesc("Minimum speculative decoding probability for draft tokens (0 = greedy)"));

  vAdd((new field_str("speculative.type"))
           ->mSetDesc("Speculative decoding method (for debugging and research purposes)")
           ->mSetHandler([&](field_eval_context &pCtx, const json &pData) {
             pCtx.vParams.vSpeculative.lTypes = {
                 fCommonSpeculativeTypeFromName(pData.at("speculative.type").get<std::string>())};
             pCtx.vParams.vSpeculative.vAutoMtp = false;
           }));

  vAdd((new field_num("speculative.ngram_size_n", pParams.vSpeculative.vNgramSimple.vSizeN))
           ->mSetDesc("Ngram size for lookup in ngram-based speculative decoding"));

  vAdd((new field_num("speculative.ngram_size_m", pParams.vSpeculative.vNgramSimple.vSizeM))
           ->mSetDesc("Mgram size for speculative tokens in ngram-based speculative decoding"));

  vAdd((new field_num("speculative.ngram_min_hits", pParams.vSpeculative.vNgramSimple.vMinHits))
           ->mSetDesc("Minimum hits at ngram lookup for mgram to be proposed"));
#endif

  vAdd((new field_json("lora"))
           ->mSetDesc("A list of LoRA adapters to apply to this request. Each entry must have `id` and `scale` fields. "
                      "Adapters not listed default to scale 0.0")
           ->mSetHandler([&](field_eval_context &pCtx, const json &pData) {
             const auto &vLora = pData.at("lora");
             if (!vLora.is_array()) {
               throw std::runtime_error("Error: 'lora' must be an array of objects with 'id' and 'scale' fields");
             }
             pCtx.vParams.dLora = fParseLoraRequest(vLora);
           }));

  // sequence breakers for DRY
  // Currently, this is not compatible with TextGen WebUI, Koboldcpp and SillyTavern format
  // Ref:
  // https://github.com/oobabooga/text-generation-webui/blob/d1af7a41ade7bd3c3a463bfa640725edb818ebaf/extensions/openai/typing.py#L39
  vAdd(
      (new field_json("dry_sequence_breakers"))
          ->mSetDesc("Specify an array of sequence breakers for DRY sampling. Only a JSON array of strings is accepted")
          ->mSetHandler([&](field_eval_context &pCtx, const json &pData) {
            pCtx.vParams.vSampling.lDrySequenceBreakers =
                fJsonValue(pData, "dry_sequence_breakers", std::vector<std::string>());
            if (pCtx.vParams.vSampling.lDrySequenceBreakers.empty()) {
              throw std::runtime_error("Error: dry_sequence_breakers must be a non-empty array of strings");
            }
          }));

  // handle both "json_schema" and "grammar"
  vAdd((new field_json("json_schema"))
           ->mAddAlias("grammar")
           ->mSetDesc("Set a JSON schema (json_schema) or GBNF grammar string (grammar) for constrained generation. "
                      "json_schema takes precedence if both are provided")
           ->mSetHandler([&](field_eval_context &pCtx, const json &pData) {
             auto &vParams = pCtx.vParams;
             if (pData.contains("json_schema") && !pData.contains("grammar")) {
               try {
                 auto vSchema = fJsonValue(pData, "json_schema", json::object());
                 SRV_DBG("JSON schema: %s\n", vSchema.dump(2).c_str());
                 std::string vGrammarStr = fJsonSchemaToGrammar(vSchema);
                 SRV_DBG("Converted grammar: %s\n", vGrammarStr.c_str());
                 vParams.vSampling.vGrammar = {cCommonGrammarTypeOutputFormat, std::move(vGrammarStr)};
               } catch (const std::exception &vE) {
                 throw std::runtime_error(std::string("\"json_schema\": ") + vE.what());
               }
             } else {
               std::string vGrammarStr = fJsonValue(pData, "grammar", std::string());
               if (!vGrammarStr.empty()) {
                 // grammar_type key is set by the server when converting chat template grammars
                 std::string vGrammarType = fJsonValue(pData, "grammar_type", std::string());
                 if (vGrammarType == "tool_calls") {
                   vParams.vSampling.vGrammar = {cCommonGrammarTypeToolCalls, std::move(vGrammarStr)};
                 } else {
                   // explicit grammar from the user (API field "grammar")
                   vParams.vSampling.vGrammar = {cCommonGrammarTypeUser, std::move(vGrammarStr)};
                 }
                 SRV_DBG("Grammar (%s): %s\n", vGrammarType.c_str(),
                         fCommonGrammarValue(vParams.vSampling.vGrammar).c_str());
               }
             }
           }));

  vAdd((new field_bool("grammar_lazy", pParams.vSampling.vGrammarLazy))
           ->mSetDesc("Whether to apply grammar constraints lazily, only when triggered (instead of at every step)"));

  //
  // Chat parser params
  //

  // TODO: change this to string field instead
  vAdd((new field_json("chat_format"))
           ->mSetDesc("Chat format used internally by the server")
           ->mSetHandler([&](field_eval_context &pCtx, const json &pData) {
             pCtx.vParams.vChatParserParams.vFormat =
                 static_cast<common_chat_format>(pData.at("chat_format").get<int>());
             SRV_TRC("chat format: %s\n", fCommonChatFormatName(pCtx.vParams.vChatParserParams.vFormat));
           }));

  vAdd((new field_str("reasoning_format"))
           ->mSetDesc("Reasoning format for chain-of-thought models")
           ->mSetHandler([&](field_eval_context &pCtx, const json &pData) {
             pCtx.vParams.vChatParserParams.vReasoningFormat =
                 fCommonReasoningFormatFromName(pData.at("reasoning_format").get<std::string>());
           }));

  vAdd((new field_str("generation_prompt"))
           ->mSetDesc("Generation prompt appended to the chat template output")
           ->mSetHandler([&](field_eval_context &pCtx, const json &pData) {
             std::string vS = pData.at("generation_prompt").get<std::string>();
             pCtx.vParams.vChatParserParams.vGenerationPrompt = vS;
             pCtx.vParams.vSampling.vGenerationPrompt = vS;
           }));

  vAdd((new field_bool("parse_tool_calls", pParams.vChatParserParams.vParseToolCalls))
           ->mSetDesc("Whether to parse tool calls from the generated output"));

  vAdd((new field_str("chat_parser"))
           ->mSetDesc("Chat parser configuration string")
           ->mSetHandler([&](field_eval_context &pCtx, const json &pData) {
             pCtx.vParams.vChatParserParams.vParser.mLoad(pData.at("chat_parser").get<std::string>());
           }));

  vAdd((new field_json("continue_final_message"))
           ->mSetDesc("Whether to continue the final message of the chat template")
           ->mSetHandler([&](field_eval_context &pCtx, const json &pData) {
             auto vContinuation = fCommonChatContinuationParse(pData.at("continue_final_message"));
             pCtx.vParams.vChatParserParams.vIsContinuation = vContinuation != cCommonChatContinuationNone;
           }));

  vAdd((new field_bool("echo", pParams.vChatParserParams.vEcho))
           ->mSetDesc("Whether to echo the input tokens in the output"));

  //
  // Token-level fields (require vocab)
  //

  vAdd((new field_json("preserved_tokens"))
           ->mSetDesc("List of token strings that must not be split during tokenization")
           ->mSetHandler([&](field_eval_context &pCtx, const json &pData) {
             GGML_ASSERT(pCtx.vVocab != nullptr);
             for (const auto &vT : pData.at("preserved_tokens")) {
               auto vIds = fCommonTokenize(pCtx.vVocab, vT.get<std::string>(), false, true);
               if (vIds.size() == 1) {
                 pCtx.vParams.vSampling.sPreservedTokens.insert(vIds[0]);
               }
             }
           }));

  vAdd((new field_json("grammar_triggers"))
           ->mSetDesc("List of strings or patterns that trigger grammar-constrained generation")
           ->mSetHandler([&](field_eval_context &pCtx, const json &pData) {
             GGML_ASSERT(pCtx.vVocab != nullptr);
             for (const auto &vT : pData.at("grammar_triggers")) {
               server_grammar_trigger vCt(vT);
               if (vCt.vValue.vType == cCommonGrammarTriggerTypeWord) {
                 const auto &vWord = vCt.vValue.vValue;
                 auto vIds = fCommonTokenize(pCtx.vVocab, vWord, false, true);
                 if (vIds.size() == 1) {
                   auto vToken = vIds[0];
                   if (std::find(pCtx.vParams.vSampling.sPreservedTokens.begin(),
                                 pCtx.vParams.vSampling.sPreservedTokens.end(),
                                 (mgs_token)vToken) == pCtx.vParams.vSampling.sPreservedTokens.end()) {
                     throw std::runtime_error("Grammar trigger word should be marked as preserved token: " + vWord);
                   }
                   common_grammar_trigger vTrigger;
                   vTrigger.vType = cCommonGrammarTriggerTypeToken;
                   vTrigger.vValue = vWord;
                   vTrigger.vToken = vToken;
                   pCtx.vParams.vSampling.lGrammarTriggers.push_back(std::move(vTrigger));
                 } else {
                   pCtx.vParams.vSampling.lGrammarTriggers.push_back({cCommonGrammarTriggerTypeWord, vWord});
                 }
               } else {
                 pCtx.vParams.vSampling.lGrammarTriggers.emplace_back(std::move(vCt.vValue));
               }
             }
             if (pCtx.vParams.vSampling.vGrammarLazy && pCtx.vParams.vSampling.lGrammarTriggers.empty()) {
               throw std::runtime_error("Error: no triggers set for lazy grammar!");
             }
           }));

  vAdd((new field_bool("reasoning_control", pParams.vSampling.vReasoningControl))
           ->mSetDesc("Create the budget sampler on demand so reasoning can be ended at runtime"));

  vAdd((new field_num("reasoning_budget_tokens", pParams.vSampling.vReasoningBudgetTokens))
           ->mSetHardLimits(-1, INT32_MAX)
           ->mSetDesc("Number of tokens in the reasoning budget (-1 = disabled)"));

  vAdd((new field_str("reasoning_budget_start_tag"))
           ->mSetDesc("Token string marking the start of the reasoning budget section")
           ->mSetHandler([&](field_eval_context &pCtx, const json &pData) {
             GGML_ASSERT(pCtx.vVocab != nullptr);
             pCtx.vParams.vSampling.lReasoningBudgetStart =
                 fCommonTokenize(pCtx.vVocab, pData.at("reasoning_budget_start_tag").get<std::string>(), false, true);
           }));

  vAdd((new field_json("reasoning_budget_end_tags"))
           ->mSetDesc("Token strings marking the end of the reasoning budget section; the first is forced when the "
                      "budget expires")
           ->mSetHandler([&](field_eval_context &pCtx, const json &pData) {
             GGML_ASSERT(pCtx.vVocab != nullptr);
             pCtx.vParams.vSampling.vReasoningBudgetEnd.clear();
             if (pData.contains("reasoning_budget_end_tags")) {
               for (const auto &vT : pData.at("reasoning_budget_end_tags")) {
                 std::string vTag = vT.get<std::string>();
                 if (!vTag.empty()) {
                   pCtx.vParams.vSampling.vReasoningBudgetEnd.push_back(
                       fCommonTokenize(pCtx.vVocab, vTag, false, true));
                 }
               }
             }
           }));

  vAdd((new field_str("reasoning_budget_message"))
           ->mSetDesc("Message to prepend to the reasoning budget end tag when forcing it")
           ->mSetHandler([&](field_eval_context &pCtx, const json &pData) {
             GGML_ASSERT(pCtx.vVocab != nullptr);
             if (!pCtx.vParams.vSampling.vReasoningBudgetEnd.empty()) {
               mgs_tokens vEndTag = pCtx.vParams.vSampling.vReasoningBudgetEnd.front();
               std::string vMessage = fJsonValue(pData, "reasoning_budget_message", std::string());
               if (!vMessage.empty()) {
                 mgs_tokens vMessageTokens = fCommonTokenize(pCtx.vVocab, vMessage, false, true);
                 vEndTag.insert(vEndTag.begin(), vMessageTokens.begin(), vMessageTokens.end());
               }
               pCtx.vParams.vSampling.lReasoningBudgetForced = std::move(vEndTag);
             }
           }));

  vAdd((new field_json("logit_bias"))
           ->mSetDesc("Modify the likelihood of specific tokens. Accepts an array of [token, bias] pairs or an object "
                      "mapping token to bias. Use false as bias to ban a token")
           ->mSetHandler([&](field_eval_context &pCtx, const json &pData) {
             GGML_ASSERT(pCtx.vVocab != nullptr);
             pCtx.vParams.vSampling.lLogitBias.clear();
             const auto &vLogitBias = pData.at("logit_bias");
             const int cNVocab = mgs_vocab_n_tokens(pCtx.vVocab);
             auto vParseBias = [](const json &pV, float &pBias) -> bool {
               if (pV.is_number()) {
                 pBias = pV.get<float>();
                 return true;
               }
               if (pV.is_boolean() && !pV.get<bool>()) {
                 pBias = -INFINITY;
                 return true;
               }
               return false;
             };
             if (vLogitBias.is_array()) {
               for (const auto &vEl : vLogitBias) {
                 if (!vEl.is_array() || vEl.size() != 2)
                   continue;
                 float vBias;
                 if (!vParseBias(vEl[1], vBias))
                   continue;
                 if (vEl[0].is_number_integer()) {
                   mgs_token vTok = vEl[0].get<mgs_token>();
                   if (vTok >= 0 && vTok < cNVocab)
                     pCtx.vParams.vSampling.lLogitBias.push_back({vTok, vBias});
                 } else if (vEl[0].is_string()) {
                   for (auto vTok : fCommonTokenize(pCtx.vVocab, vEl[0].get<std::string>(), false))
                     pCtx.vParams.vSampling.lLogitBias.push_back({vTok, vBias});
                 }
               }
             } else if (vLogitBias.is_object()) {
               for (const auto &vEl : vLogitBias.items()) {
                 float vBias;
                 if (!vParseBias(vEl.value(), vBias))
                   continue;
                 char *vEnd;
                 mgs_token vTok = strtol(vEl.key().c_str(), &vEnd, 10);
                 if (*vEnd == 0) {
                   if (vTok >= 0 && vTok < cNVocab)
                     pCtx.vParams.vSampling.lLogitBias.push_back({vTok, vBias});
                 } else {
                   for (auto vT : fCommonTokenize(pCtx.vVocab, vEl.key(), false))
                     pCtx.vParams.vSampling.lLogitBias.push_back({vT, vBias});
                 }
               }
             }
           }));

  vAdd((new field_bool("ignore_eos", pParams.vSampling.vIgnoreEos))
           ->mSetDesc("Ignore the end-of-sequence token and continue generating")
           ->mSetHandler([&](field_eval_context &pCtx, const json &pData) {
             GGML_ASSERT(pCtx.lLogitBiasEog != nullptr);
             pCtx.vParams.vSampling.vIgnoreEos = pData.at("ignore_eos").get<bool>();
             if (pCtx.vParams.vSampling.vIgnoreEos && pCtx.lLogitBiasEog) {
               pCtx.vParams.vSampling.lLogitBias.insert(pCtx.vParams.vSampling.lLogitBias.end(),
                                                        pCtx.lLogitBiasEog->begin(), pCtx.lLogitBiasEog->end());
             }
           }));

  vAdd((new field_json("stop"))
           ->mSetDesc("Specify stopping strings. Generation stops when one is produced, and the string is not included "
                      "in the output")
           ->mSetHandler([&](field_eval_context &pCtx, const json &pData) {
             pCtx.vParams.lAntiprompt.clear();
             const auto &vStop = pData.at("stop");
             if (vStop.is_array()) {
               for (const auto &vWord : vStop) {
                 if (!vWord.empty())
                   pCtx.vParams.lAntiprompt.push_back(vWord);
               }
             } else if (vStop.is_string()) {
               pCtx.vParams.lAntiprompt.push_back(vStop.get<std::string>());
             }
             // fall back to CLI defaults if the request provided no effective stop strings
             if (pCtx.vParams.lAntiprompt.empty()) {
               pCtx.vParams.lAntiprompt = pParamsBase.lAntiprompt;
             }
           }));

  vAdd((new field_json("samplers"))
           ->mSetDesc("The order in which samplers are applied. An array of sampler type names, or a single string of "
                      "sampler chars")
           ->mSetHandler([&](field_eval_context &pCtx, const json &pData) {
             const auto &vSamplers = pData.at("samplers");
             if (vSamplers.is_array()) {
               pCtx.vParams.vSampling.lSamplers = fCommonSamplerTypesFromNames(vSamplers);
             } else if (vSamplers.is_string()) {
               pCtx.vParams.vSampling.lSamplers = fCommonSamplerTypesFromChars(vSamplers.get<std::string>());
             }
           }));

  return lFields;
}

task_params fEvalMgsCmplSchema(const mgs_vocab *pVocab, const common_params &pParamsBase, const int pNCtxSlot,
                               const std::vector<mgs_logit_bias> &pLogitBiasEog, const json &pData) {
  task_params vParams;

  // Sampling parameter defaults are loaded from the global server context (but individual requests can still them)
  vParams.vSampling = pParamsBase.vSampling;
  vParams.vSpeculative = pParamsBase.vSpeculative;
  vParams.vNKeep = pParamsBase.vNKeep;
  vParams.vNPredict = pParamsBase.vNPredict;
  vParams.vNCacheReuse = pParamsBase.vNCacheReuse;
  vParams.vCachePrompt = pParamsBase.vCachePrompt;
  vParams.lAntiprompt = pParamsBase.lAntiprompt;
  vParams.vSsePingInterval = pParamsBase.vSsePingInterval;

  // enabling this will output extra debug information in the HTTP responses from the server
  vParams.vVerbose = pParamsBase.vVerbosity > 9;

  vParams.vChatParserParams.vReasoningFormat = pParamsBase.vReasoningFormat;

  // create context and schema
  field_eval_context vCtx(vParams);
  vCtx.vVocab = pVocab;
  vCtx.lLogitBiasEog = &pLogitBiasEog;

  auto vSchema = fMakeMgsCmplSchema(pParamsBase, vParams);

  // eval all fields in the schema
  for (const auto &vF : vSchema) {
    vF->mEval(vCtx, pData);
  }

  // post-processing
  {
    if (vParams.vSampling.vPenaltyLastN == -1) {
      // note: should be the slot's context and not the full context, but it's ok
      vParams.vSampling.vPenaltyLastN = pNCtxSlot;
    }

    if (vParams.vSampling.vDryPenaltyLastN == -1) {
      vParams.vSampling.vDryPenaltyLastN = pNCtxSlot;
    }
  }

  // debugging
  {
    auto vBudget = vParams.vSampling.vReasoningBudgetTokens;
    SRV_DBG("reasoning budget: tokens=%d, generation_prompt='%s', start=%zu toks, end=%zu seqs, forced=%zu toks\n",
            vBudget, vParams.vSampling.vGenerationPrompt.c_str(), vParams.vSampling.lReasoningBudgetStart.size(),
            vParams.vSampling.vReasoningBudgetEnd.size(), vParams.vSampling.lReasoningBudgetForced.size());
  }

  return vParams;
}

//
// mEval() implementations
//

static void fHandleWithCatch(const char *pName, std::function<void()> pFunc) {
  try {
    pFunc();
  } catch (const std::exception &vE) {
    throw std::invalid_argument(fStringFormat("Field '%s': %s", pName, vE.what()));
  }
}

// treat a null value as absent so clients can send null to request the server default
static bool fHasValue(const json &pData, const char *pN) {
  auto vIt = pData.find(pN);
  return vIt != pData.end() && !vIt->is_null();
}

template <typename T> void field_num<T>::mEval(field_eval_context &pCtx, const json &pData) {
  for (const auto &vN : lName) {
    if (fHasValue(pData, vN)) {
      fHandleWithCatch(vN, [&]() {
        if (vCustomHandler) {
          vCustomHandler(pCtx, pData);
        } else if (!vIsHardLimit) {
          vVal = std::max(vMin, std::min(vMax, pData.at(vN).template get<T>()));
        } else {
          T vTmp = pData.at(vN).template get<T>();
          if (vTmp < vMin || vTmp > vMax) {
            throw std::invalid_argument(std::string("Value must be between ") + std::to_string(vMin) +
                                        " <= value <= " + std::to_string(vMax) + ", but got " + std::to_string(vTmp));
          }
          vVal = vTmp;
        }
      });
      return;
    }
  }
}

void field_str::mEval(field_eval_context &pCtx, const json &pData) {
  GGML_ASSERT(vCustomHandler);
  for (const auto &vN : lName) {
    if (fHasValue(pData, vN)) {
      fHandleWithCatch(vN, [&]() { vCustomHandler(pCtx, pData); });
      return;
    }
  }
}

void field_bool::mEval(field_eval_context &pCtx, const json &pData) {
  for (const auto &vN : lName) {
    if (fHasValue(pData, vN)) {
      fHandleWithCatch(vN, [&]() {
        if (vCustomHandler) {
          vCustomHandler(pCtx, pData);
        } else {
          vVal = pData.at(vN).get<bool>();
        }
      });
      return;
    }
  }
}

void field_json::mEval(field_eval_context &pCtx, const json &pData) {
  GGML_ASSERT(vCustomHandler);
  for (const auto &vN : lName) {
    if (fHasValue(pData, vN)) {
      fHandleWithCatch(vN, [&]() { vCustomHandler(pCtx, pData); });
      return;
    }
  }
}

void field_nested::mEval(field_eval_context &pCtx, const json &pData) {
  for (const auto &vN : lName) {
    if (pData.contains(vN) && pData.at(vN).is_object()) {
      for (auto &vF : lSubfields) {
        vF->mEval(pCtx, pData.at(vN));
      }
      return;
    }
  }
}

} // namespace server_schema
