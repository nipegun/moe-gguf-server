#include "mgs-arch.h"

#include "mgs-impl.h"

#include <map>
#include <vector>

static const std::map<llm_arch, const char *> dLlmArchNames = {
    {cLlmArchFlint, "flint"},
    {cLlmArchQwen35, "qwen35"},
    {cLlmArchQwen35moe, "qwen35moe"},
    {cLlmArchQwen4exp, "qwen4exp"},
    {cLlmArchDeepseek41, "deepseek41"},
    {cLlmArchGlmDsa, "glm-dsa"},
    {cLlmArchGlm5Next, "glm5-next"},
    {cLlmArchNemotronHMoe, "nemotron_h_moe"},
    {cLlmArchMinimaxM3, "minimax-m3"},
    {cLlmArchMimo2, "mimo2"},
    {cLlmArchKimiK3, "kimi-k3"},
    {cLlmArchUnknown, "(unknown)"},
};

static const std::map<llm_kv, const char *> dLlmKvNames = {
    {cLlmKvGeneralType, "general.type"},
    {cLlmKvGeneralArchitecture, "general.architecture"},
    {cLlmKvGeneralQuantizationVersion, "general.quantization_version"},
    {cLlmKvGeneralAlignment, "general.alignment"},
    {cLlmKvGeneralFileType, "general.file_type"},
    {cLlmKvGeneralSamplingSequence, "general.sampling.sequence"},
    {cLlmKvGeneralSamplingTopK, "general.sampling.top_k"},
    {cLlmKvGeneralSamplingTopP, "general.sampling.top_p"},
    {cLlmKvGeneralSamplingMinP, "general.sampling.min_p"},
    {cLlmKvGeneralSamplingXtcProbability, "general.sampling.xtc_probability"},
    {cLlmKvGeneralSamplingXtcThreshold, "general.sampling.xtc_threshold"},
    {cLlmKvGeneralSamplingTemp, "general.sampling.temp"},
    {cLlmKvGeneralSamplingPenaltyLastN, "general.sampling.penalty_last_n"},
    {cLlmKvGeneralSamplingPenaltyRepeat, "general.sampling.penalty_repeat"},
    {cLlmKvGeneralSamplingMirostat, "general.sampling.mirostat"},
    {cLlmKvGeneralSamplingMirostatTau, "general.sampling.mirostat_tau"},
    {cLlmKvGeneralSamplingMirostatEta, "general.sampling.mirostat_eta"},
    {cLlmKvGeneralName, "general.name"},
    {cLlmKvGeneralAuthor, "general.author"},
    {cLlmKvGeneralVersion, "general.version"},
    {cLlmKvGeneralUrl, "general.url"},
    {cLlmKvGeneralDescription, "general.description"},
    {cLlmKvGeneralLicense, "general.license"},
    {cLlmKvGeneralSourceUrl, "general.source.url"},
    {cLlmKvGeneralSourceHfRepo, "general.source.huggingface.repository"},

    {cLlmKvVocabSize, "%s.vocab_size"},
    {cLlmKvContextLength, "%s.context_length"},
    {cLlmKvEmbeddingLength, "%s.embedding_length"},
    {cLlmKvEmbeddingLengthOut, "%s.embedding_length_out"},
    {cLlmKvEmbeddingLengthPerLayer, "%s.embedding_length_per_layer_input"},
    {cLlmKvBlockCount, "%s.block_count"},
    {cLlmKvLeadingDenseBlockCount, "%s.leading_dense_block_count"},
    {cLlmKvAttnResBlockSize, "%s.attn_res.block_size"},
    {cLlmKvActivationSituBeta, "%s.activation.situ_beta"},
    {cLlmKvActivationSituLinearBeta, "%s.activation.situ_linear_beta"},
    {cLlmKvFeedForwardLength, "%s.feed_forward_length"},
    {cLlmKvExpertFeedForwardLength, "%s.expert_feed_forward_length"},
    {cLlmKvExpertSharedFeedForwardLength, "%s.expert_shared_feed_forward_length"},
    {cLlmKvSwigluClampExp, "%s.swiglu_clamp_exp"},
    {cLlmKvSwigluClampShexp, "%s.swiglu_clamp_shexp"},
    {cLlmKvTensorDataLayout, "%s.tensor_data_layout"},
    {cLlmKvExpertCount, "%s.expert_count"},
    {cLlmKvExpertUsedCount, "%s.expert_used_count"},
    {cLlmKvExpertSharedCount, "%s.expert_shared_count"},
    {cLlmKvExpertGroupCount, "%s.expert_group_count"},
    {cLlmKvExpertGroupUsedCount, "%s.expert_group_used_count"},
    {cLlmKvExpertWeightsScale, "%s.expert_weights_scale"},
    {cLlmKvExpertWeightsNorm, "%s.expert_weights_norm"},
    {cLlmKvExpertLatentLength, "%s.expert_latent_length"},
    {cLlmKvExpertGatingFunc, "%s.expert_gating_func"},
    {cLlmKvMoeLatentSize, "%s.moe_latent_size"},
    {cLlmKvNextnPredictLayers, "%s.nextn_predict_layers"},
    {cLlmKvNumDeepstackLayers, "%s.n_deepstack_layers"},
    {cLlmKvDeepstackMapping, "%s.deepstack_mapping"},
    {cLlmKvPoolingType, "%s.pooling_type"},
    {cLlmKvLogitScale, "%s.logit_scale"},
    {cLlmKvResidualScale, "%s.residual_scale"},
    {cLlmKvEmbeddingScale, "%s.embedding_scale"},
    {cLlmKvFullAttentionInterval, "%s.full_attention_interval"},

    {cLlmKvAttentionHeadCount, "%s.attention.head_count"},
    {cLlmKvAttentionHeadCountKv, "%s.attention.head_count_kv"},
    {cLlmKvAttentionMaxAlibiBias, "%s.attention.max_alibi_bias"},
    {cLlmKvAttentionClampKqv, "%s.attention.clamp_kqv"},
    {cLlmKvAttentionKeyLength, "%s.attention.key_length"},
    {cLlmKvAttentionValueLength, "%s.attention.value_length"},
    {cLlmKvAttentionLayernormEps, "%s.attention.layer_norm_epsilon"},
    {cLlmKvAttentionLayernormRmsEps, "%s.attention.layer_norm_rms_epsilon"},
    {cLlmKvAttentionCausal, "%s.attention.causal"},
    {cLlmKvAttentionQLoraRank, "%s.attention.q_lora_rank"},
    {cLlmKvAttentionKvLoraRank, "%s.attention.kv_lora_rank"},
    {cLlmKvAttentionSlidingWindow, "%s.attention.sliding_window"},
    {cLlmKvAttentionSlidingWindowPattern, "%s.attention.sliding_window_pattern"},
    {cLlmKvAttentionRopePattern, "%s.attention.rope_pattern"},
    {cLlmKvAttentionScale, "%s.attention.scale"},
    {cLlmKvAttentionValueScale, "%s.attention.value_scale"},
    {cLlmKvAttentionKeyLengthMla, "%s.attention.key_length_mla"},
    {cLlmKvAttentionValueLengthMla, "%s.attention.value_length_mla"},
    {cLlmKvAttentionKeyLengthSwa, "%s.attention.key_length_swa"},
    {cLlmKvAttentionValueLengthSwa, "%s.attention.value_length_swa"},
    {cLlmKvAttentionIndexerHeadCount, "%s.attention.indexer.head_count"},
    {cLlmKvAttentionIndexerKeyLength, "%s.attention.indexer.key_length"},
    {cLlmKvAttentionIndexerTopK, "%s.attention.indexer.top_k"},
    {cLlmKvAttentionIndexerBlockSize, "%s.attention.indexer.block_size"},
    {cLlmKvAttentionIndexerLocalBlocks, "%s.attention.indexer.local_blocks"},
    {cLlmKvAttentionIndexerTypes, "%s.attention.indexer.types"},
    {cLlmKvAttentionIndexerKpool, "%s.attention.indexer.kpool"},
    {cLlmKvAttentionIndexerKpoolSelectTail, "%s.attention.indexer.kpool_select_tail"},
    {cLlmKvAttentionOutputGroupCount, "%s.attention.output_group_count"},
    {cLlmKvAttentionOutputLoraRank, "%s.attention.output_lora_rank"},
    {cLlmKvAttentionCompressRopeFreqBase, "%s.attention.compress_rope_freq_base"},
    {cLlmKvAttentionCompressRatios, "%s.attention.compress_ratios"},
    {cLlmKvAttentionRecurrentLayers, "%s.attention.recurrent_layers"},

    {cLlmKvHyperConnectionCount, "%s.hyper_connection.count"},
    {cLlmKvHyperConnectionSinkhornIterations, "%s.hyper_connection.sinkhorn_iterations"},
    {cLlmKvHyperConnectionEpsilon, "%s.hyper_connection.epsilon"},
    {cLlmKvHyperConnectionLowRank, "%s.hyper_connection.low_rank"},

    {cLlmKvPleLayers, "%s.ple.layers"},
    {cLlmKvPleNgramSize, "%s.ple.ngram_size"},
    {cLlmKvPleHeadsPerNgram, "%s.ple.heads_per_ngram"},
    {cLlmKvPleConvKernel, "%s.ple.conv_kernel"},
    {cLlmKvPleLayerMultipliers, "%s.ple.layer_multipliers"},
    {cLlmKvPleHeadOffsets, "%s.ple.head_offsets"},
    {cLlmKvPleHeadVocabSizes, "%s.ple.head_vocab_sizes"},
    {cLlmKvPleEosTokenId, "%s.ple.eos_token_id"},
    {cLlmKvPleImageTokenId, "%s.ple.image_token_id"},

    {cLlmKvRopeDimensionCount, "%s.rope.dimension_count"},
    {cLlmKvRopeDimensionCountSwa, "%s.rope.dimension_count_swa"},
    {cLlmKvRopeDimensionSections, "%s.rope.dimension_sections"},
    {cLlmKvRopeFreqBase, "%s.rope.freq_base"},
    {cLlmKvRopeFreqBaseSwa, "%s.rope.freq_base_swa"},
    {cLlmKvRopeScalingType, "%s.rope.scaling.type"},
    {cLlmKvRopeScalingFactor, "%s.rope.scaling.factor"},
    {cLlmKvRopeScalingAttnFactor, "%s.rope.scaling.attn_factor"},
    {cLlmKvRopeScalingOrigCtxLen, "%s.rope.scaling.original_context_length"},
    {cLlmKvRopeScalingFinetuned, "%s.rope.scaling.finetuned"},
    {cLlmKvRopeScalingYarnLogMul, "%s.rope.scaling.yarn_log_multiplier"},
    {cLlmKvRopeScalingYarnExtFactor, "%s.rope.scaling.yarn_ext_factor"},
    {cLlmKvRopeScalingYarnAttnFactor, "%s.rope.scaling.yarn_attn_factor"},
    {cLlmKvRopeScalingYarnBetaFast, "%s.rope.scaling.yarn_beta_fast"},
    {cLlmKvRopeScalingYarnBetaSlow, "%s.rope.scaling.yarn_beta_slow"},

    {cLlmKvSplitNo, "split.no"},
    {cLlmKvSplitCount, "split.count"},
    {cLlmKvSplitTensorsCount, "split.tensors.count"},

    {cLlmKvSsmConvKernel, "%s.ssm.conv_kernel"},
    {cLlmKvSsmInnerSize, "%s.ssm.inner_size"},
    {cLlmKvSsmStateSize, "%s.ssm.state_size"},
    {cLlmKvSsmTimeStepRank, "%s.ssm.time_step_rank"},
    {cLlmKvSsmGroupCount, "%s.ssm.group_count"},
    {cLlmKvSsmDtBCRms, "%s.ssm.dt_b_c_rms"},

    {cLlmKvKdaHeadDim, "%s.kda.head_dim"},
    {cLlmKvKdaGateLowerBound, "%s.kda.gate_lower_bound"},

    {cLlmKvClassifierOutputLabels, "%s.classifier.output_labels"},

    {cLlmKvShortconvLCache, "%s.shortconv.l_cache"},

    {cLlmKvTokenizerModel, "tokenizer.ggml.model"},
    {cLlmKvTokenizerPre, "tokenizer.ggml.pre"},
    {cLlmKvTokenizerList, "tokenizer.ggml.tokens"},
    {cLlmKvTokenizerTokenType, "tokenizer.ggml.token_type"},
    {cLlmKvTokenizerTokenTypeCount, "tokenizer.ggml.token_type_count"},
    {cLlmKvTokenizerScores, "tokenizer.ggml.scores"},
    {cLlmKvTokenizerMerges, "tokenizer.ggml.merges"},
    {cLlmKvTokenizerBosId, "tokenizer.ggml.bos_token_id"},
    {cLlmKvTokenizerEosId, "tokenizer.ggml.eos_token_id"},
    {cLlmKvTokenizerEotId, "tokenizer.ggml.eot_token_id"},
    {cLlmKvTokenizerEomId, "tokenizer.ggml.eom_token_id"},
    {cLlmKvTokenizerUnkId, "tokenizer.ggml.unknown_token_id"},
    {cLlmKvTokenizerSepId, "tokenizer.ggml.seperator_token_id"},
    {cLlmKvTokenizerPadId, "tokenizer.ggml.padding_token_id"},
    {cLlmKvTokenizerMaskId, "tokenizer.ggml.mask_token_id"},
    {cLlmKvTokenizerAddBos, "tokenizer.ggml.add_bos_token"},
    {cLlmKvTokenizerAddEos, "tokenizer.ggml.add_eos_token"},
    {cLlmKvTokenizerAddSep, "tokenizer.ggml.add_sep_token"},
    {cLlmKvTokenizerAddPrefix, "tokenizer.ggml.add_space_prefix"},
    {cLlmKvTokenizerRemoveExtraWs, "tokenizer.ggml.remove_extra_whitespaces"},
    {cLlmKvTokenizerPrecompiledCharsmap, "tokenizer.ggml.precompiled_charsmap"},
    {cLlmKvTokenizerHfJson, "tokenizer.huggingface.json"},
    {cLlmKvTokenizerRwkv, "tokenizer.rwkv.world"},
    {cLlmKvTokenizerChatTemplate, "tokenizer.chat_template"},
    {cLlmKvTokenizerNormalizerLowercase, "tokenizer.ggml.normalizer.lowercase"},
    {cLlmKvTokenizerNormalizerStripAccents, "tokenizer.ggml.normalizer.strip_accents"},
    {cLlmKvTokenizerFimPreId, "tokenizer.ggml.fim_pre_token_id"},
    {cLlmKvTokenizerFimSufId, "tokenizer.ggml.fim_suf_token_id"},
    {cLlmKvTokenizerFimMidId, "tokenizer.ggml.fim_mid_token_id"},
    {cLlmKvTokenizerFimPadId, "tokenizer.ggml.fim_pad_token_id"},
    {cLlmKvTokenizerFimRepId, "tokenizer.ggml.fim_rep_token_id"},
    {cLlmKvTokenizerFimSepId, "tokenizer.ggml.fim_sep_token_id"},
    {cLlmKvTokenizerSuppressTokens, "tokenizer.ggml.suppress_tokens"},

    {cLlmKvAdapterType, "adapter.type"},
    {cLlmKvAdapterLoraAlpha, "adapter.lora.alpha"},
    {cLlmKvAdapterLoraTaskName, "adapter.lora.task_name"},
    {cLlmKvAdapterLoraPromptPrefix, "adapter.lora.prompt_prefix"},
    {cLlmKvAdapterAloraInvocationTokens, "adapter.alora.invocation_tokens"},

};

static const std::map<llm_tensor, const char *> dLlmTensorNames = {
    {cLlmTensorTokenEmbd, "token_embd"},
    {cLlmTensorOutputNorm, "output_norm"},
    {cLlmTensorOutput, "output"},
    {cLlmTensorAttnNorm, "blk.%d.attn_norm"},
    {cLlmTensorAttnQ, "blk.%d.attn_q"},
    {cLlmTensorAttnK, "blk.%d.attn_k"},
    {cLlmTensorAttnV, "blk.%d.attn_v"},
    {cLlmTensorAttnOut, "blk.%d.attn_output"},
    {cLlmTensorFfnGateInp, "blk.%d.ffn_gate_inp"},
    {cLlmTensorFfnNorm, "blk.%d.ffn_norm"},
    {cLlmTensorFfnGate, "blk.%d.ffn_gate"},
    {cLlmTensorFfnDown, "blk.%d.ffn_down"},
    {cLlmTensorFfnUp, "blk.%d.ffn_up"},
    {cLlmTensorFfnGateExps, "blk.%d.ffn_gate_exps"},
    {cLlmTensorFfnGateUpExps, "blk.%d.ffn_gate_up_exps"},
    {cLlmTensorFfnDownExps, "blk.%d.ffn_down_exps"},
    {cLlmTensorFfnUpExps, "blk.%d.ffn_up_exps"},
    {cLlmTensorAttnPostNorm, "blk.%d.post_attention_norm"},
    {cLlmTensorAttnQNorm, "blk.%d.attn_q_norm"},
    {cLlmTensorAttnKNorm, "blk.%d.attn_k_norm"},
    {cLlmTensorAttnGate, "blk.%d.attn_gate"},
    {cLlmTensorFfnGateShexp, "blk.%d.ffn_gate_shexp"},
    {cLlmTensorFfnUpShexp, "blk.%d.ffn_up_shexp"},
    {cLlmTensorFfnDownShexp, "blk.%d.ffn_down_shexp"},
    {cLlmTensorFfnExpProbsB, "blk.%d.exp_probs_b"},
    {cLlmTensorFfnLatentDown, "blk.%d.ffn_latent_down"},
    {cLlmTensorFfnLatentUp, "blk.%d.ffn_latent_up"},
    {cLlmTensorAttnQkv, "blk.%d.attn_qkv"},
    {cLlmTensorLayerOutNorm, "blk.%d.layer_output_norm"},
    {cLlmTensorPosEmbd, "position_embd"},
    {cLlmTensorTokenTypes, "token_types"},
    {cLlmTensorFfnGateInpShexp, "blk.%d.ffn_gate_inp_shexp"},
    {cLlmTensorSsmANoscan, "blk.%d.ssm_a"},
    {cLlmTensorSsmConv1d, "blk.%d.ssm_conv1d"},
    {cLlmTensorSsmDt, "blk.%d.ssm_dt"},
    {cLlmTensorSsmAlpha, "blk.%d.ssm_alpha"},
    {cLlmTensorSsmIn, "blk.%d.ssm_in"},
    {cLlmTensorSsmNorm, "blk.%d.ssm_norm"},
    {cLlmTensorSsmOut, "blk.%d.ssm_out"},
    {cLlmTensorSsmA, "blk.%d.ssm_a"},
    {cLlmTensorSsmD, "blk.%d.ssm_d"},
    {cLlmTensorSsmConv1dQ, "blk.%d.ssm_conv1d_q"},
    {cLlmTensorSsmConv1dK, "blk.%d.ssm_conv1d_k"},
    {cLlmTensorSsmConv1dV, "blk.%d.ssm_conv1d_v"},
    {cLlmTensorSsmFA, "blk.%d.ssm_f_a"},
    {cLlmTensorSsmFB, "blk.%d.ssm_f_b"},
    {cLlmTensorSsmBeta, "blk.%d.ssm_beta"},
    {cLlmTensorSsmGA, "blk.%d.ssm_g_a"},
    {cLlmTensorSsmGB, "blk.%d.ssm_g_b"},
    {cLlmTensorSsmG, "blk.%d.ssm_g"},
    {cLlmTensorAttnResScore, "blk.%d.attn_res_score"},
    {cLlmTensorFfnResScore, "blk.%d.ffn_res_score"},
    {cLlmTensorOutputResScore, "output_res_score"},
    {cLlmTensorFfnRoutedDown, "blk.%d.ffn_routed_down"},
    {cLlmTensorFfnRoutedUp, "blk.%d.ffn_routed_up"},
    {cLlmTensorFfnRoutedNorm, "blk.%d.ffn_routed_norm"},
    {cLlmTensorSsmNorm, "blk.%d.ssm_norm"},
    {cLlmTensorAttnQANorm, "blk.%d.attn_q_a_norm"},
    {cLlmTensorAttnKvANorm, "blk.%d.attn_kv_a_norm"},
    {cLlmTensorAttnQA, "blk.%d.attn_q_a"},
    {cLlmTensorAttnQB, "blk.%d.attn_q_b"},
    {cLlmTensorAttnKvAMqa, "blk.%d.attn_kv_a_mqa"},
    {cLlmTensorAttnKv, "blk.%d.attn_kv"},
    {cLlmTensorAttnKvNorm, "blk.%d.attn_kv_a_norm"},
    {cLlmTensorAttnOutA, "blk.%d.attn_output_a"},
    {cLlmTensorAttnOutB, "blk.%d.attn_output_b"},
    {cLlmTensorHcHeadNorm, "output_hc_norm"},
    {cLlmTensorHcHeadDown, "output_hc_down"},
    {cLlmTensorHcHeadUp, "output_hc_up"},
    {cLlmTensorHcAttnFn, "blk.%d.hc_attn_fn"},
    {cLlmTensorHcAttnBase, "blk.%d.hc_attn_base"},
    {cLlmTensorHcAttnScale, "blk.%d.hc_attn_scale"},
    {cLlmTensorHcFfnFn, "blk.%d.hc_ffn_fn"},
    {cLlmTensorHcFfnBase, "blk.%d.hc_ffn_base"},
    {cLlmTensorHcFfnScale, "blk.%d.hc_ffn_scale"},
    {cLlmTensorHcAttnNorm, "blk.%d.hc_attn_norm"},
    {cLlmTensorHcAttnDown, "blk.%d.hc_attn_down"},
    {cLlmTensorHcAttnUp, "blk.%d.hc_attn_up"},
    {cLlmTensorHcAttnInject, "blk.%d.hc_attn_inject"},
    {cLlmTensorHcFfnNorm, "blk.%d.hc_ffn_norm"},
    {cLlmTensorHcFfnDown, "blk.%d.hc_ffn_down"},
    {cLlmTensorHcFfnUp, "blk.%d.hc_ffn_up"},
    {cLlmTensorHcFfnInject, "blk.%d.hc_ffn_inject"},
    {cLlmTensorPleKey, "blk.%d.ple_key"},
    {cLlmTensorPleValue, "blk.%d.ple_value"},
    {cLlmTensorPleNormKey, "blk.%d.ple_norm_key"},
    {cLlmTensorPleNormQuery, "blk.%d.ple_norm_query"},
    {cLlmTensorPleNormConv, "blk.%d.ple_norm_conv"},
    {cLlmTensorPleConv1d, "blk.%d.ple_conv1d"},
    {cLlmTensorAttnCompressorWkv, "blk.%d.attn_compressor_kv"},
    {cLlmTensorAttnCompressorWgate, "blk.%d.attn_compressor_gate"},
    {cLlmTensorAttnCompressorNorm, "blk.%d.attn_compressor_norm"},
    {cLlmTensorPerLayerTokenEmbd, "per_layer_token_embd"},
    {cLlmTensorAttnKB, "blk.%d.attn_k_b"},
    {cLlmTensorAttnVB, "blk.%d.attn_v_b"},
    {cLlmTensorNextnEhProj, "blk.%d.nextn.eh_proj"},
    {cLlmTensorNextnEmbedTokens, "blk.%d.nextn.embed_tokens"},
    {cLlmTensorNextnEnorm, "blk.%d.nextn.enorm"},
    {cLlmTensorNextnHnorm, "blk.%d.nextn.hnorm"},
    {cLlmTensorNextnSharedHeadHead, "blk.%d.nextn.shared_head_head"},
    {cLlmTensorNextnSharedHeadNorm, "blk.%d.nextn.shared_head_norm"},
    {cLlmTensorNextnHcHeadNorm, "blk.%d.nextn.hc_head_norm"},
    {cLlmTensorNextnHcHeadDown, "blk.%d.nextn.hc_head_down"},
    {cLlmTensorNextnHcHeadUp, "blk.%d.nextn.hc_head_up"},
    {cLlmTensorAttnSinks, "blk.%d.attn_sinks"},
    {cLlmTensorIndexerKNorm, "blk.%d.indexer.k_norm"},
    {cLlmTensorIndexerProj, "blk.%d.indexer.proj"},
    {cLlmTensorIndexerAttnK, "blk.%d.indexer.attn_k"},
    {cLlmTensorIndexerAttnQB, "blk.%d.indexer.attn_q_b"},
    {cLlmTensorIndexerKpoolGate, "blk.%d.indexer_compressor_gate"},
    {cLlmTensorIndexerKpoolApe, "blk.%d.indexer_compressor_ape"},
    {cLlmTensorIndexerQProj, "blk.%d.indexer.q_proj"},
    {cLlmTensorIndexerKProj, "blk.%d.indexer.k_proj"},
    {cLlmTensorIndexerQNorm, "blk.%d.indexer.q_norm"},
    {cLlmTensorFfnExpProbsBVl, "blk.%d.exp_probs_b_vl"},
    {cLlmTensorEngramEmbd, "blk.%d.engram_embd"},
    {cLlmTensorEngramWkv, "blk.%d.engram_wkv"},
    {cLlmTensorEngramQ, "blk.%d.engram_q"},
    {cLlmTensorEngramK, "blk.%d.engram_k"},
    {cLlmTensorEngramKv, "blk.%d.engram_kv"},
    {cLlmTensorEngramQNorm, "blk.%d.engram_q_norm"},
    {cLlmTensorEngramKNorm, "blk.%d.engram_k_norm"},
};

// declare information about the model weight tensors:
// - the layer in which the tensor is going to be used. this is needed in order to assign the correct buffer type for
// the weight
// - the operator which is going to use the weight. this is needed to determine if the respective backend supports the
// operator
//
// for example, input layers are usually assigned to CPU/host buffer types
//
// a mismatch between the declared information and the actual layer/op in which the tensor is used can lead to
// sub-optimal
//   assignment of the buffer types and extra overhead during computation
//
static const std::map<llm_tensor, llm_tensor_info> dLlmTensorInfos = {
    {cLlmTensorTokenEmbd, {cLlmTensorLayerInput, GGML_OP_GET_ROWS}},
    {cLlmTensorPosEmbd, {cLlmTensorLayerInput, GGML_OP_GET_ROWS}},
    {cLlmTensorTokenTypes, {cLlmTensorLayerInput, GGML_OP_GET_ROWS}},
    {cLlmTensorOutput, {cLlmTensorLayerOutput, GGML_OP_MUL_MAT}},
    {cLlmTensorOutputNorm, {cLlmTensorLayerOutput, GGML_OP_MUL}},
    {cLlmTensorAttnQ, {cLlmTensorLayerRepeating, GGML_OP_MUL_MAT}},
    {cLlmTensorAttnK, {cLlmTensorLayerRepeating, GGML_OP_MUL_MAT}},
    {cLlmTensorAttnV, {cLlmTensorLayerRepeating, GGML_OP_MUL_MAT}},
    {cLlmTensorAttnQkv, {cLlmTensorLayerRepeating, GGML_OP_MUL_MAT}},
    {cLlmTensorAttnOut, {cLlmTensorLayerRepeating, GGML_OP_MUL_MAT}},
    {cLlmTensorAttnGate, {cLlmTensorLayerRepeating, GGML_OP_MUL_MAT}},
    {cLlmTensorFfnGate, {cLlmTensorLayerRepeating, GGML_OP_MUL_MAT}},
    {cLlmTensorFfnDown, {cLlmTensorLayerRepeating, GGML_OP_MUL_MAT}},
    {cLlmTensorFfnUp, {cLlmTensorLayerRepeating, GGML_OP_MUL_MAT}},
    {cLlmTensorFfnDownShexp, {cLlmTensorLayerRepeating, GGML_OP_MUL_MAT}},
    {cLlmTensorFfnGateShexp, {cLlmTensorLayerRepeating, GGML_OP_MUL_MAT}},
    {cLlmTensorFfnUpShexp, {cLlmTensorLayerRepeating, GGML_OP_MUL_MAT}},
    {cLlmTensorAttnQA, {cLlmTensorLayerRepeating, GGML_OP_MUL_MAT}},
    {cLlmTensorAttnQB, {cLlmTensorLayerRepeating, GGML_OP_MUL_MAT}},
    {cLlmTensorAttnKvAMqa, {cLlmTensorLayerRepeating, GGML_OP_MUL_MAT}},
    {cLlmTensorAttnKv, {cLlmTensorLayerRepeating, GGML_OP_MUL_MAT}},
    {cLlmTensorAttnKvNorm, {cLlmTensorLayerRepeating, GGML_OP_MUL}},
    {cLlmTensorAttnOutA, {cLlmTensorLayerRepeating, GGML_OP_MUL_MAT}},
    {cLlmTensorAttnOutB, {cLlmTensorLayerRepeating, GGML_OP_MUL_MAT}},
    {cLlmTensorHcHeadNorm, {cLlmTensorLayerOutput, GGML_OP_MUL}},
    {cLlmTensorHcHeadDown, {cLlmTensorLayerOutput, GGML_OP_MUL_MAT}},
    {cLlmTensorHcHeadUp, {cLlmTensorLayerOutput, GGML_OP_MUL_MAT}},
    {cLlmTensorHcAttnFn, {cLlmTensorLayerRepeating, GGML_OP_MUL_MAT}},
    {cLlmTensorHcAttnBase, {cLlmTensorLayerRepeating, GGML_OP_ADD}},
    {cLlmTensorHcAttnScale, {cLlmTensorLayerRepeating, GGML_OP_MUL}},
    {cLlmTensorHcFfnFn, {cLlmTensorLayerRepeating, GGML_OP_MUL_MAT}},
    {cLlmTensorHcFfnBase, {cLlmTensorLayerRepeating, GGML_OP_ADD}},
    {cLlmTensorHcFfnScale, {cLlmTensorLayerRepeating, GGML_OP_MUL}},
    {cLlmTensorHcAttnNorm, {cLlmTensorLayerRepeating, GGML_OP_MUL}},
    {cLlmTensorHcAttnDown, {cLlmTensorLayerRepeating, GGML_OP_MUL_MAT}},
    {cLlmTensorHcAttnUp, {cLlmTensorLayerRepeating, GGML_OP_MUL_MAT}},
    {cLlmTensorHcAttnInject, {cLlmTensorLayerRepeating, GGML_OP_MUL_MAT}},
    {cLlmTensorHcFfnNorm, {cLlmTensorLayerRepeating, GGML_OP_MUL}},
    {cLlmTensorHcFfnDown, {cLlmTensorLayerRepeating, GGML_OP_MUL_MAT}},
    {cLlmTensorHcFfnUp, {cLlmTensorLayerRepeating, GGML_OP_MUL_MAT}},
    {cLlmTensorHcFfnInject, {cLlmTensorLayerRepeating, GGML_OP_MUL_MAT}},
    {cLlmTensorPleKey, {cLlmTensorLayerRepeating, GGML_OP_MUL_MAT}},
    {cLlmTensorPleValue, {cLlmTensorLayerRepeating, GGML_OP_MUL_MAT}},
    {cLlmTensorPleNormKey, {cLlmTensorLayerRepeating, GGML_OP_MUL}},
    {cLlmTensorPleNormQuery, {cLlmTensorLayerRepeating, GGML_OP_MUL}},
    {cLlmTensorPleNormConv, {cLlmTensorLayerRepeating, GGML_OP_MUL}},
    {cLlmTensorPleConv1d, {cLlmTensorLayerRepeating, GGML_OP_SSM_CONV}},
    {cLlmTensorAttnCompressorWkv, {cLlmTensorLayerRepeating, GGML_OP_MUL_MAT}},
    {cLlmTensorAttnCompressorWgate, {cLlmTensorLayerRepeating, GGML_OP_MUL_MAT}},
    {cLlmTensorAttnCompressorNorm, {cLlmTensorLayerRepeating, GGML_OP_MUL}},
    {cLlmTensorAttnKB, {cLlmTensorLayerRepeating, GGML_OP_MUL_MAT}},
    {cLlmTensorAttnVB, {cLlmTensorLayerRepeating, GGML_OP_MUL_MAT}},
    {cLlmTensorAttnSinks, {cLlmTensorLayerRepeating, GGML_OP_SCALE}},
    {cLlmTensorFfnGateInpShexp, {cLlmTensorLayerRepeating, GGML_OP_MUL_MAT}},
    {cLlmTensorFfnGateInp, {cLlmTensorLayerRepeating, GGML_OP_MUL_MAT}},
    {cLlmTensorSsmIn, {cLlmTensorLayerRepeating, GGML_OP_MUL_MAT}},
    {cLlmTensorSsmDt, {cLlmTensorLayerRepeating, GGML_OP_MUL_MAT}},
    {cLlmTensorSsmOut, {cLlmTensorLayerRepeating, GGML_OP_MUL_MAT}},
    {cLlmTensorSsmAlpha, {cLlmTensorLayerRepeating, GGML_OP_MUL_MAT}},
    {cLlmTensorSsmConv1d, {cLlmTensorLayerRepeating, GGML_OP_SSM_CONV}},
    {cLlmTensorSsmA, {cLlmTensorLayerRepeating, GGML_OP_SSM_SCAN}},
    {cLlmTensorSsmANoscan,
     {cLlmTensorLayerRepeating, GGML_OP_MUL}}, // a version of SSM_A used for MUL instead of SSM_SCAN
    {cLlmTensorSsmD, {cLlmTensorLayerRepeating, GGML_OP_MUL}},
    {cLlmTensorSsmNorm, {cLlmTensorLayerRepeating, GGML_OP_MUL}},
    // Kimi KDA - Conv tensors are 4D [d_conv, 1, d_inner, 1], reshaped to 2D at runtime
    {cLlmTensorSsmConv1dQ, {cLlmTensorLayerRepeating, GGML_OP_MUL}},
    {cLlmTensorSsmConv1dK, {cLlmTensorLayerRepeating, GGML_OP_MUL}},
    {cLlmTensorSsmConv1dV, {cLlmTensorLayerRepeating, GGML_OP_MUL}},
    {cLlmTensorSsmFA, {cLlmTensorLayerRepeating, GGML_OP_MUL_MAT}},
    {cLlmTensorSsmFB, {cLlmTensorLayerRepeating, GGML_OP_MUL_MAT}},
    {cLlmTensorSsmBeta, {cLlmTensorLayerRepeating, GGML_OP_MUL_MAT}},
    {cLlmTensorSsmGA, {cLlmTensorLayerRepeating, GGML_OP_MUL_MAT}},
    {cLlmTensorSsmGB, {cLlmTensorLayerRepeating, GGML_OP_MUL_MAT}},
    {cLlmTensorSsmG, {cLlmTensorLayerRepeating, GGML_OP_MUL_MAT}},
    {cLlmTensorAttnResScore, {cLlmTensorLayerRepeating, GGML_OP_MUL}},
    {cLlmTensorFfnResScore, {cLlmTensorLayerRepeating, GGML_OP_MUL}},
    {cLlmTensorOutputResScore, {cLlmTensorLayerOutput, GGML_OP_MUL}},
    {cLlmTensorFfnRoutedDown, {cLlmTensorLayerRepeating, GGML_OP_MUL_MAT}},
    {cLlmTensorFfnRoutedUp, {cLlmTensorLayerRepeating, GGML_OP_MUL_MAT}},
    {cLlmTensorFfnRoutedNorm, {cLlmTensorLayerRepeating, GGML_OP_MUL}},
    {cLlmTensorAttnNorm, {cLlmTensorLayerRepeating, GGML_OP_MUL}},
    {cLlmTensorAttnPostNorm, {cLlmTensorLayerRepeating, GGML_OP_MUL}},
    {cLlmTensorFfnNorm, {cLlmTensorLayerRepeating, GGML_OP_MUL}},
    {cLlmTensorAttnQNorm, {cLlmTensorLayerRepeating, GGML_OP_MUL}},
    {cLlmTensorAttnKNorm, {cLlmTensorLayerRepeating, GGML_OP_MUL}},
    {cLlmTensorLayerOutNorm, {cLlmTensorLayerRepeating, GGML_OP_MUL}},
    {cLlmTensorAttnQANorm, {cLlmTensorLayerRepeating, GGML_OP_MUL}},
    {cLlmTensorAttnKvANorm, {cLlmTensorLayerRepeating, GGML_OP_MUL}},
    {cLlmTensorFfnDownExps, {cLlmTensorLayerRepeating, GGML_OP_MUL_MAT_ID}},
    {cLlmTensorFfnGateExps, {cLlmTensorLayerRepeating, GGML_OP_MUL_MAT_ID}},
    {cLlmTensorFfnUpExps, {cLlmTensorLayerRepeating, GGML_OP_MUL_MAT_ID}},
    {cLlmTensorFfnGateUpExps, {cLlmTensorLayerRepeating, GGML_OP_MUL_MAT_ID}},
    {cLlmTensorFfnExpProbsB, {cLlmTensorLayerRepeating, GGML_OP_ADD}},
    // per-layer token embeddings (qwen4exp)
    {cLlmTensorPerLayerTokenEmbd, {cLlmTensorLayerInput, GGML_OP_GET_ROWS}},
    {cLlmTensorIndexerKNorm, {cLlmTensorLayerRepeating, GGML_OP_MUL}},
    {cLlmTensorIndexerProj, {cLlmTensorLayerRepeating, GGML_OP_MUL_MAT}},
    {cLlmTensorIndexerAttnK, {cLlmTensorLayerRepeating, GGML_OP_MUL_MAT}},
    {cLlmTensorIndexerAttnQB, {cLlmTensorLayerRepeating, GGML_OP_MUL_MAT}},
    {cLlmTensorIndexerKpoolGate, {cLlmTensorLayerRepeating, GGML_OP_MUL_MAT}},
    {cLlmTensorIndexerKpoolApe, {cLlmTensorLayerRepeating, GGML_OP_ADD}},
    {cLlmTensorIndexerQProj, {cLlmTensorLayerRepeating, GGML_OP_MUL_MAT}},
    {cLlmTensorIndexerKProj, {cLlmTensorLayerRepeating, GGML_OP_MUL_MAT}},
    {cLlmTensorIndexerQNorm, {cLlmTensorLayerRepeating, GGML_OP_MUL}},
    {cLlmTensorFfnExpProbsBVl, {cLlmTensorLayerRepeating, GGML_OP_ADD}},
    {cLlmTensorEngramEmbd, {cLlmTensorLayerRepeating, GGML_OP_GET_ROWS}},
    {cLlmTensorEngramWkv, {cLlmTensorLayerRepeating, GGML_OP_MUL_MAT}},
    {cLlmTensorEngramQ, {cLlmTensorLayerRepeating, GGML_OP_GET_ROWS}},
    {cLlmTensorEngramK, {cLlmTensorLayerRepeating, GGML_OP_GET_ROWS}},
    {cLlmTensorEngramKv, {cLlmTensorLayerRepeating, GGML_OP_MUL_MAT}},
    {cLlmTensorEngramQNorm, {cLlmTensorLayerRepeating, GGML_OP_GET_ROWS}},
    {cLlmTensorEngramKNorm, {cLlmTensorLayerRepeating, GGML_OP_GET_ROWS}},
    // NextN/MTP tensors are stored per-block (blk.%d.nextn.*) even though only the
    // last nextn_predict_layers blocks carry them. Classify as LAYER_REPEATING so
    // the model loader doesn't fault on the block index.
    {cLlmTensorNextnEhProj, {cLlmTensorLayerRepeating, GGML_OP_MUL_MAT}},
    {cLlmTensorNextnEmbedTokens, {cLlmTensorLayerRepeating, GGML_OP_GET_ROWS}},
    {cLlmTensorNextnEnorm, {cLlmTensorLayerRepeating, GGML_OP_MUL}},
    {cLlmTensorNextnHnorm, {cLlmTensorLayerRepeating, GGML_OP_MUL}},
    {cLlmTensorNextnSharedHeadHead, {cLlmTensorLayerRepeating, GGML_OP_MUL_MAT}},
    {cLlmTensorNextnSharedHeadNorm, {cLlmTensorLayerRepeating, GGML_OP_MUL}},
    {cLlmTensorNextnHcHeadNorm, {cLlmTensorLayerRepeating, GGML_OP_MUL}},
    {cLlmTensorNextnHcHeadDown, {cLlmTensorLayerRepeating, GGML_OP_MUL_MAT}},
    {cLlmTensorNextnHcHeadUp, {cLlmTensorLayerRepeating, GGML_OP_MUL_MAT}},
    // Nemotron 3 Super
    // latent projections feed ggml_mul_mat, the buft probe must use MUL_MAT to keep them on GPU
    {cLlmTensorFfnLatentDown, {cLlmTensorLayerRepeating, GGML_OP_MUL_MAT}},
    {cLlmTensorFfnLatentUp, {cLlmTensorLayerRepeating, GGML_OP_MUL_MAT}},
};

LLM_KV::LLM_KV(llm_arch pArch, const char *pSuffix) : vArch(pArch), vSuffix(pSuffix) {}

std::string LLM_KV::operator()(llm_kv pKv) const {
  std::string vName = ::fFormat(dLlmKvNames.at(pKv), dLlmArchNames.at(vArch));

  if (vSuffix != nullptr) {
    vName += ".";
    vName += vSuffix;
  }

  return vName;
}

std::string fLlmKvNameForArch(llm_kv pKv, const std::string &pArchName) {
  return ::fFormat(dLlmKvNames.at(pKv), pArchName.c_str());
}

LLM_TN_IMPL::LLM_TN_IMPL(llm_arch pArch, llm_tensor pTensor, const char *pSuffix, int pBid, int pXid)
    : cArch(pArch), cTensor(pTensor), cSuffix(pSuffix), cBid(pBid), cXid(pXid) {}

std::string LLM_TN_IMPL::mStr() const {
  if (dLlmTensorNames.find(cTensor) == dLlmTensorNames.end()) {
    GGML_ABORT("unknown tensor name for tensor id %d", static_cast<int>(cTensor));
  }

  std::string vName = ::fFormat(dLlmTensorNames.at(cTensor), cBid, cXid);
  if (cSuffix != nullptr) {
    vName += ".";
    vName += cSuffix;
  }

  return vName;
}

std::vector<llm_arch> fLlmArchAll() {
  std::vector<llm_arch> lRet;
  lRet.reserve(dLlmArchNames.size());
  for (const auto &[vArch, _] : dLlmArchNames) {
    lRet.push_back(vArch);
  }
  return lRet;
}

const char *fLlmArchName(llm_arch pArch) {
  auto vIt = dLlmArchNames.find(pArch);
  if (vIt == dLlmArchNames.end()) {
    return "unknown";
  }
  return vIt->second;
}

llm_arch fLlmArchFromString(const std::string &pName) {
  for (const auto &vKv : dLlmArchNames) { // NOLINT
    if (vKv.second == pName) {
      return vKv.first;
    }
  }

  return cLlmArchUnknown;
}

const llm_tensor_info &fLlmTensorInfoFor(llm_tensor pTensor) { return dLlmTensorInfos.at(pTensor); }

bool fLlmArchIsHybrid(const llm_arch &pArch) {
  switch (pArch) {
  case cLlmArchNemotronHMoe:
  case cLlmArchKimiK3:
  case cLlmArchQwen35:
  case cLlmArchQwen35moe:
  case cLlmArchQwen4exp:
  case cLlmArchDeepseek41:
  case cLlmArchGlm5Next:
    return true;
  default:
    return false;
  }
}

bool fLlmArchSupportsRsRollback(const llm_arch &pArch) {
  switch (pArch) {
  case cLlmArchKimiK3:
  case cLlmArchQwen35:
  case cLlmArchQwen35moe:
  case cLlmArchQwen4exp:
  case cLlmArchDeepseek41:
  case cLlmArchGlm5Next:
    return true;
  default:
    return false;
  }
}

bool fLlmArchSupportsSmTensor(const llm_arch &pArch) {
  switch (pArch) {
  case cLlmArchGlmDsa:
  case cLlmArchNemotronHMoe:
  case cLlmArchMinimaxM3:
  case cLlmArchKimiK3:
  case cLlmArchGlm5Next:
  case cLlmArchFlint:
    return false;
  default:
    return true;
  }
}
