#include "mgs-arch.h"

#include "mgs-impl.h"

#include <map>
#include <vector>

static const std::map<llm_arch, const char *> dLlmArchNames = {
    {cLlmArchClip, "clip"}, // dummy, only used by quantization
    {cLlmArchDeci, "deci"},
    {cLlmArchFalcon, "falcon"},
    {cLlmArchGrok, "grok"},
    {cLlmArchGpt2, "gpt2"},
    {cLlmArchGptj, "gptj"},
    {cLlmArchGptneox, "gptneox"},
    {cLlmArchMpt, "mpt"},
    {cLlmArchBaichuan, "baichuan"},
    {cLlmArchStarcoder, "starcoder"},
    {cLlmArchRefact, "refact"},
    {cLlmArchBert, "bert"},
    {cLlmArchModernBert, "modern-bert"},
    {cLlmArchNomicBert, "nomic-bert"},
    {cLlmArchNomicBertMoe, "nomic-bert-moe"},
    {cLlmArchNeoBert, "neo-bert"},
    {cLlmArchJinaBertV2, "jina-bert-v2"},
    {cLlmArchJinaBertV3, "jina-bert-v3"},
    {cLlmArchEurobert, "eurobert"},
    {cLlmArchBloom, "bloom"},
    {cLlmArchStablelm, "stablelm"},
    {cLlmArchQwen, "qwen"},
    {cLlmArchQwen2, "qwen2"},
    {cLlmArchQwen2moe, "qwen2moe"},
    {cLlmArchQwen2vl, "qwen2vl"},
    {cLlmArchQwen3, "qwen3"},
    {cLlmArchQwen3moe, "qwen3moe"},
    {cLlmArchQwen3next, "qwen3next"},
    {cLlmArchQwen3vl, "qwen3vl"},
    {cLlmArchQwen3vlmoe, "qwen3vlmoe"},
    {cLlmArchQwen35, "qwen35"},
    {cLlmArchQwen35moe, "qwen35moe"},
    {cLlmArchQwen4exp, "qwen4exp"},
    {cLlmArchPhi2, "phi2"},
    {cLlmArchPhi3, "phi3"},
    {cLlmArchPhimoe, "phimoe"},
    {cLlmArchPlamo, "plamo"},
    {cLlmArchPlamo2, "plamo2"},
    {cLlmArchPlamo3, "plamo3"},
    {cLlmArchCodeshell, "codeshell"},
    {cLlmArchOrion, "orion"},
    {cLlmArchInternlm2, "internlm2"},
    {cLlmArchMinicpm, "minicpm"},
    {cLlmArchMinicpm3, "minicpm3"},
    {cLlmArchGemma, "gemma"},
    {cLlmArchGemma2, "gemma2"},
    {cLlmArchGemma3, "gemma3"},
    {cLlmArchGemma3n, "gemma3n"},
    {cLlmArchGemmaEmbedding, "gemma-embedding"},
    {cLlmArchStarcoder2, "starcoder2"},
    {cLlmArchMamba, "mamba"},
    {cLlmArchMamba2, "mamba2"},
    {cLlmArchJamba, "jamba"},
    {cLlmArchFalconH1, "falcon-h1"},
    {cLlmArchXverse, "xverse"},
    {cLlmArchCommandR, "command-r"},
    {cLlmArchCohere2, "cohere2"},
    {cLlmArchCohere2moe, "cohere2moe"},
    {cLlmArchDbrx, "dbrx"},
    {cLlmArchOlmo, "olmo"},
    {cLlmArchOlmo2, "olmo2"},
    {cLlmArchOlmoe, "olmoe"},
    {cLlmArchOpenelm, "openelm"},
    {cLlmArchArctic, "arctic"},
    {cLlmArchDeepseek41, "deepseek41"},
    {cLlmArchChatglm, "chatglm"},
    {cLlmArchGlm4, "glm4"},
    {cLlmArchGlm4Moe, "glm4moe"},
    {cLlmArchGlmDsa, "glm-dsa"},
    {cLlmArchGlm5Next, "glm5-next"},
    {cLlmArchBitnet, "bitnet"},
    {cLlmArchT5, "t5"},
    {cLlmArchT5encoder, "t5encoder"},
    {cLlmArchJais, "jais"},
    {cLlmArchJais2, "jais2"},
    {cLlmArchNemotron, "nemotron"},
    {cLlmArchNemotronH, "nemotron_h"},
    {cLlmArchNemotronHMoe, "nemotron_h_moe"},
    {cLlmArchExaone, "exaone"},
    {cLlmArchExaone4, "exaone4"},
    {cLlmArchExaoneMoe, "exaone-moe"},
    {cLlmArchRwkv6, "rwkv6"},
    {cLlmArchRwkv6qwen2, "rwkv6qwen2"},
    {cLlmArchRwkv7, "rwkv7"},
    {cLlmArchArwkv7, "arwkv7"},
    {cLlmArchGranite, "granite"},
    {cLlmArchGraniteMoe, "granitemoe"},
    {cLlmArchGraniteHybrid, "granitehybrid"},
    {cLlmArchChameleon, "chameleon"},
    {cLlmArchWavtokenizerDec, "wavtokenizer-dec"},
    {cLlmArchPlm, "plm"},
    {cLlmArchBailingmoe, "bailingmoe"},
    {cLlmArchBailingmoe2, "bailingmoe2"},
    {cLlmArchDots1, "dots1"},
    {cLlmArchArcee, "arcee"},
    {cLlmArchAfmoe, "afmoe"},
    {cLlmArchLaguna, "laguna"},
    {cLlmArchERNIE4Point5, "ernie4_5"},
    {cLlmArchERNIE4Point5Moe, "ernie4_5-moe"},
    {cLlmArchHunyuanMoe, "hunyuan-moe"},
    {cLlmArchHunyuanDense, "hunyuan-dense"},
    {cLlmArchHunyuanVl, "hunyuan_vl"},
    {cLlmArchHyV3, "hy_v3"},
    {cLlmArchSmollm3, "smollm3"},
    {cLlmArchOpenaiMoe, "gpt-oss"},
    {cLlmArchLfm2, "lfm2"},
    {cLlmArchLfm2moe, "lfm2moe"},
    {cLlmArchDream, "dream"},
    {cLlmArchSmallthinker, "smallthinker"},
    {cLlmArchLlada, "llada"},
    {cLlmArchLladaMoe, "llada-moe"},
    {cLlmArchSeedOss, "seed_oss"},
    {cLlmArchGrovemoe, "grovemoe"},
    {cLlmArchApertus, "apertus"},
    {cLlmArchMinimaxM2, "minimax-m2"},
    {cLlmArchMinimaxM3, "minimax-m3"},
    {cLlmArchCogvlm, "cogvlm"},
    {cLlmArchRnd1, "rnd1"},
    {cLlmArchPanguEmbed, "pangu-embedded"},
    {cLlmArchMistral3, "mistral3"},
    {cLlmArchEagle3, "eagle3"},
    {cLlmArchDflash, "dflash"},
    {cLlmArchPaddleocr, "paddleocr"},
    {cLlmArchMimo2, "mimo2"},
    {cLlmArchStep35, "step35"},
    {cLlmArchMaincoder, "maincoder"},
    {cLlmArchKimiLinear, "kimi-linear"},
    {cLlmArchKimiK3, "kimi-k3"},
    {cLlmArchTalkie, "talkie"},
    {cLlmArchMellum, "mellum"},
    {cLlmArchNanbeige, "nanbeige"},
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
    {cLlmKvFeaturesLength, "%s.features_length"},
    {cLlmKvBlockCount, "%s.block_count"},
    {cLlmKvLeadingDenseBlockCount, "%s.leading_dense_block_count"},
    {cLlmKvAttnResBlockSize, "%s.attn_res.block_size"},
    {cLlmKvActivationSituBeta, "%s.activation.situ_beta"},
    {cLlmKvActivationSituLinearBeta, "%s.activation.situ_linear_beta"},
    {cLlmKvFeedForwardLength, "%s.feed_forward_length"},
    {cLlmKvExpertFeedForwardLength, "%s.expert_feed_forward_length"},
    {cLlmKvExpertSharedFeedForwardLength, "%s.expert_shared_feed_forward_length"},
    {cLlmKvExpertChunkFeedForwardLength, "%s.expert_chunk_feed_forward_length"},
    {cLlmKvSwigluClampExp, "%s.swiglu_clamp_exp"},
    {cLlmKvSwigluClampShexp, "%s.swiglu_clamp_shexp"},
    {cLlmKvUseParallelResidual, "%s.use_parallel_residual"},
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
    {cLlmKvExpertGroupScale, "%s.expert_group_scale"},
    {cLlmKvExpertsPerGroup, "%s.experts_per_group"},
    {cLlmKvMoeEveryNLayers, "%s.moe_every_n_layers"},
    {cLlmKvMoeLatentSize, "%s.moe_latent_size"},
    {cLlmKvNextnPredictLayers, "%s.nextn_predict_layers"},
    {cLlmKvNumDeepstackLayers, "%s.n_deepstack_layers"},
    {cLlmKvDeepstackMapping, "%s.deepstack_mapping"},
    {cLlmKvHiddenAct, "%s.hidden_activation"},
    {cLlmKvPoolingType, "%s.pooling_type"},
    {cLlmKvLogitScale, "%s.logit_scale"},
    {cLlmKvDecoderStartTokenId, "%s.decoder_start_token_id"},
    {cLlmKvDecoderBlockCount, "%s.decoder_block_count"},
    {cLlmKvAttnLogitSoftcapping, "%s.attn_logit_softcapping"},
    {cLlmKvRouterLogitSoftcapping, "%s.router_logit_softcapping"},
    {cLlmKvFinalLogitSoftcapping, "%s.final_logit_softcapping"},
    {cLlmKvSwinNorm, "%s.swin_norm"},
    {cLlmKvRescaleEveryNLayers, "%s.rescale_every_n_layers"},
    {cLlmKvTimeMixExtraDim, "%s.time_mix_extra_dim"},
    {cLlmKvTimeDecayExtraDim, "%s.time_decay_extra_dim"},
    {cLlmKvResidualScale, "%s.residual_scale"},
    {cLlmKvEmbeddingScale, "%s.embedding_scale"},
    {cLlmKvTokenShiftCount, "%s.token_shift_count"},
    {cLlmKvInterleaveMoeLayerStep, "%s.interleave_moe_layer_step"},
    {cLlmKvFullAttentionInterval, "%s.full_attention_interval"},
    {cLlmKvNumLoops, "%s.num_loops"},
    {cLlmKvSkipLoopFinalNorm, "%s.skip_loop_final_norm"},

    {cLlmKvAttentionHeadCount, "%s.attention.head_count"},
    {cLlmKvAttentionHeadCountKv, "%s.attention.head_count_kv"},
    {cLlmKvAttentionMaxAlibiBias, "%s.attention.max_alibi_bias"},
    {cLlmKvAttentionClampKqv, "%s.attention.clamp_kqv"},
    {cLlmKvAttentionKeyLength, "%s.attention.key_length"},
    {cLlmKvAttentionValueLength, "%s.attention.value_length"},
    {cLlmKvAttentionLayernormEps, "%s.attention.layer_norm_epsilon"},
    {cLlmKvAttentionLayernormRmsEps, "%s.attention.layer_norm_rms_epsilon"},
    {cLlmKvAttentionGroupnormEps, "%s.attention.group_norm_epsilon"},
    {cLlmKvAttentionGroupnormGroups, "%s.attention.group_norm_groups"},
    {cLlmKvAttentionCausal, "%s.attention.causal"},
    {cLlmKvAttentionQLoraRank, "%s.attention.q_lora_rank"},
    {cLlmKvAttentionKvLoraRank, "%s.attention.kv_lora_rank"},
    {cLlmKvAttentionDecayLoraRank, "%s.attention.decay_lora_rank"},
    {cLlmKvAttentionIclrLoraRank, "%s.attention.iclr_lora_rank"},
    {cLlmKvAttentionValueResidualMixLoraRank, "%s.attention.value_residual_mix_lora_rank"},
    {cLlmKvAttentionGateLoraRank, "%s.attention.gate_lora_rank"},
    {cLlmKvAttentionRelativeBucketsCount, "%s.attention.relative_buckets_count"},
    {cLlmKvAttentionSlidingWindow, "%s.attention.sliding_window"},
    {cLlmKvAttentionSlidingWindowPattern, "%s.attention.sliding_window_pattern"},
    {cLlmKvAttentionScale, "%s.attention.scale"},
    {cLlmKvAttentionOutputScale, "%s.attention.output_scale"},
    {cLlmKvAttentionValueScale, "%s.attention.value_scale"},
    {cLlmKvAttentionTemperatureLength, "%s.attention.temperature_length"},
    {cLlmKvAttentionTemperatureScale, "%s.attention.temperature_scale"},
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
    {cLlmKvRopeScalingAlpha, "%s.rope.scaling.alpha"},
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

    {cLlmKvWkvHeadSize, "%s.wkv.head_size"},

    {cLlmKvPosnetEmbeddingLength, "%s.posnet.embedding_length"},
    {cLlmKvPosnetBlockCount, "%s.posnet.block_count"},

    {cLlmKvConvnextEmbeddingLength, "%s.convnext.embedding_length"},
    {cLlmKvConvnextBlockCount, "%s.convnext.block_count"},

    {cLlmKvClassifierOutputLabels, "%s.classifier.output_labels"},

    {cLlmKvTargetLayers, "%s.target_layers"},
    {cLlmKvTargetHiddenSize, "%s.target_hidden_size"},
    {cLlmKvNormBeforeResidual, "%s.norm_before_residual"},
    {cLlmKvNormBeforeFc, "%s.norm_before_fc"},

    {cLlmKvShortconvLCache, "%s.shortconv.l_cache"},
    // sentence-transformers dense modules feature dims
    {cLlmKvDense2FeatIn, "%s.dense_2_feat_in"},
    {cLlmKvDense2FeatOut, "%s.dense_2_feat_out"},
    {cLlmKvDense3FeatIn, "%s.dense_3_feat_in"},
    {cLlmKvDense3FeatOut, "%s.dense_3_feat_out"},

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
    {cLlmKvTokenizerClsId, "tokenizer.ggml.cls_token_id"},
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

    {cLlmKvXieluAlphaN, "xielu.alpha_n"},
    {cLlmKvXieluAlphaP, "xielu.alpha_p"},
    {cLlmKvXieluBeta, "xielu.beta"},
    {cLlmKvXieluEps, "xielu.eps"},
};

static const std::map<llm_tensor, const char *> dLlmTensorNames = {
    {cLlmTensorTokenEmbd, "token_embd"},
    {cLlmTensorOutputNorm, "output_norm"},
    {cLlmTensorOutputNormLfm2, "token_embd_norm"}, // fix for wrong tensor name
    {cLlmTensorOutput, "output"},
    {cLlmTensorRopeFreqs, "rope_freqs"},
    {cLlmTensorAttnNorm, "blk.%d.attn_norm"},
    {cLlmTensorAttnQ, "blk.%d.attn_q"},
    {cLlmTensorAttnK, "blk.%d.attn_k"},
    {cLlmTensorAttnV, "blk.%d.attn_v"},
    {cLlmTensorAttnOut, "blk.%d.attn_output"},
    {cLlmTensorAttnRotEmbd, "blk.%d.attn_rot_embd"},
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
    {cLlmTensorFfnPostNorm, "blk.%d.post_ffw_norm"},
    {cLlmTensorFfnGateShexp, "blk.%d.ffn_gate_shexp"},
    {cLlmTensorFfnUpShexp, "blk.%d.ffn_up_shexp"},
    {cLlmTensorFfnDownShexp, "blk.%d.ffn_down_shexp"},
    {cLlmTensorFfnExpProbsB, "blk.%d.exp_probs_b"},
    {cLlmTensorFfnLatentDown, "blk.%d.ffn_latent_down"},
    {cLlmTensorFfnLatentUp, "blk.%d.ffn_latent_up"},
    {cLlmTensorAttnNorm2, "blk.%d.attn_norm_2"},
    {cLlmTensorAttnQkv, "blk.%d.attn_qkv"},
    {cLlmTensorLayerOutNorm, "blk.%d.layer_output_norm"},
    {cLlmTensorAttnOutNorm, "blk.%d.attn_output_norm"},
    {cLlmTensorPosEmbd, "position_embd"},
    {cLlmTensorFfnAct, "blk.%d.ffn.act"},
    {cLlmTensorTokenEmbdNorm, "token_embd_norm"},
    {cLlmTensorTokenTypes, "token_types"},
    {cLlmTensorCls, "cls"},
    {cLlmTensorClsOut, "cls.output"},
    {cLlmTensorClsNorm, "cls.norm"},
    {cLlmTensorEncOutputNorm, "enc.output_norm"},
    {cLlmTensorFfnGateInpShexp, "blk.%d.ffn_gate_inp_shexp"},
    {cLlmTensorSsmANoscan, "blk.%d.ssm_a"},
    {cLlmTensorSsmConv1d, "blk.%d.ssm_conv1d"},
    {cLlmTensorSsmDt, "blk.%d.ssm_dt"},
    {cLlmTensorSsmBetaAlpha, "blk.%d.ssm_ba"},
    {cLlmTensorSsmAlpha, "blk.%d.ssm_alpha"},
    {cLlmTensorSsmIn, "blk.%d.ssm_in"},
    {cLlmTensorSsmNorm, "blk.%d.ssm_norm"},
    {cLlmTensorSsmOut, "blk.%d.ssm_out"},
    {cLlmTensorRopeFactorsLong, "rope_factors_long"},
    {cLlmTensorRopeFactorsShort, "rope_factors_short"},
    {cLlmTensorSsmX, "blk.%d.ssm_x"},
    {cLlmTensorSsmA, "blk.%d.ssm_a"},
    {cLlmTensorSsmD, "blk.%d.ssm_d"},
    {cLlmTensorSsmDtNorm, "blk.%d.ssm_dt_norm"},
    {cLlmTensorSsmBNorm, "blk.%d.ssm_b_norm"},
    {cLlmTensorSsmCNorm, "blk.%d.ssm_c_norm"},
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
    {cLlmTensorAltupUnembdProj, "altup_unembd_proj"},
    {cLlmTensorAltupProj, "altup_proj"},
    {cLlmTensorAltupCorrectCoef, "blk.%d.altup_correct_coef"},
    {cLlmTensorAltupCorrectScale, "blk.%d.altup_correct_scale"},
    {cLlmTensorAltupPredictCoef, "blk.%d.altup_predict_coef"},
    {cLlmTensorAltupRouter, "blk.%d.altup_router"},
    {cLlmTensorAltupRouterNorm, "blk.%d.altup_router_norm"},
    {cLlmTensorLaurelL, "blk.%d.laurel_l"},
    {cLlmTensorLaurelR, "blk.%d.laurel_r"},
    {cLlmTensorLaurelPostNorm, "blk.%d.laurel_post_norm"},
    {cLlmTensorDense2Out, "dense_2"},
    {cLlmTensorDense3Out, "dense_3"},
    {cLlmTensorFfnNormExps, "blk.%d.ffn_norm_exps"},
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
    {cLlmTensorAttnSubNorm, "blk.%d.attn_sub_norm"},
    {cLlmTensorFfnSubNorm, "blk.%d.ffn_sub_norm"},
    {cLlmTensorDecOutputNorm, "dec.output_norm"},
    {cLlmTensorDecAttnNorm, "dec.blk.%d.attn_norm"},
    {cLlmTensorDecAttnQ, "dec.blk.%d.attn_q"},
    {cLlmTensorDecAttnK, "dec.blk.%d.attn_k"},
    {cLlmTensorDecAttnV, "dec.blk.%d.attn_v"},
    {cLlmTensorDecAttnOut, "dec.blk.%d.attn_o"},
    {cLlmTensorDecAttnRelB, "dec.blk.%d.attn_rel_b"},
    {cLlmTensorDecCrossAttnNorm, "dec.blk.%d.cross_attn_norm"},
    {cLlmTensorDecCrossAttnQ, "dec.blk.%d.cross_attn_q"},
    {cLlmTensorDecCrossAttnK, "dec.blk.%d.cross_attn_k"},
    {cLlmTensorDecCrossAttnV, "dec.blk.%d.cross_attn_v"},
    {cLlmTensorDecCrossAttnOut, "dec.blk.%d.cross_attn_o"},
    {cLlmTensorDecCrossAttnRelB, "dec.blk.%d.cross_attn_rel_b"},
    {cLlmTensorDecFfnNorm, "dec.blk.%d.ffn_norm"},
    {cLlmTensorDecFfnGate, "dec.blk.%d.ffn_gate"},
    {cLlmTensorDecFfnDown, "dec.blk.%d.ffn_down"},
    {cLlmTensorDecFfnUp, "dec.blk.%d.ffn_up"},
    {cLlmTensorEncAttnNorm, "enc.blk.%d.attn_norm"},
    {cLlmTensorEncAttnQ, "enc.blk.%d.attn_q"},
    {cLlmTensorEncAttnK, "enc.blk.%d.attn_k"},
    {cLlmTensorEncAttnV, "enc.blk.%d.attn_v"},
    {cLlmTensorEncAttnOut, "enc.blk.%d.attn_o"},
    {cLlmTensorEncAttnRelB, "enc.blk.%d.attn_rel_b"},
    {cLlmTensorEncFfnNorm, "enc.blk.%d.ffn_norm"},
    {cLlmTensorEncFfnGate, "enc.blk.%d.ffn_gate"},
    {cLlmTensorEncFfnDown, "enc.blk.%d.ffn_down"},
    {cLlmTensorEncFfnUp, "enc.blk.%d.ffn_up"},
    {cLlmTensorTimeMixW1, "blk.%d.time_mix_w1"},
    {cLlmTensorTimeMixW2, "blk.%d.time_mix_w2"},
    {cLlmTensorTimeMixLerpX, "blk.%d.time_mix_lerp_x"},
    {cLlmTensorTimeMixLerpW, "blk.%d.time_mix_lerp_w"},
    {cLlmTensorTimeMixLerpK, "blk.%d.time_mix_lerp_k"},
    {cLlmTensorTimeMixLerpV, "blk.%d.time_mix_lerp_v"},
    {cLlmTensorTimeMixLerpR, "blk.%d.time_mix_lerp_r"},
    {cLlmTensorTimeMixLerpG, "blk.%d.time_mix_lerp_g"},
    {cLlmTensorTimeMixLerpFused, "blk.%d.time_mix_lerp_fused"},
    {cLlmTensorTimeMixFirst, "blk.%d.time_mix_first"},
    {cLlmTensorTimeMixDecay, "blk.%d.time_mix_decay"},
    {cLlmTensorTimeMixDecayW1, "blk.%d.time_mix_decay_w1"},
    {cLlmTensorTimeMixDecayW2, "blk.%d.time_mix_decay_w2"},
    {cLlmTensorTimeMixKey, "blk.%d.time_mix_key"},
    {cLlmTensorTimeMixValue, "blk.%d.time_mix_value"},
    {cLlmTensorTimeMixReceptance, "blk.%d.time_mix_receptance"},
    {cLlmTensorTimeMixGate, "blk.%d.time_mix_gate"},
    {cLlmTensorTimeMixLn, "blk.%d.time_mix_ln"},
    {cLlmTensorTimeMixOutput, "blk.%d.time_mix_output"},
    {cLlmTensorChannelMixLerpK, "blk.%d.channel_mix_lerp_k"},
    {cLlmTensorChannelMixLerpR, "blk.%d.channel_mix_lerp_r"},
    {cLlmTensorChannelMixKey, "blk.%d.channel_mix_key"},
    {cLlmTensorChannelMixValue, "blk.%d.channel_mix_value"},
    {cLlmTensorChannelMixReceptance, "blk.%d.channel_mix_receptance"},
    {cLlmTensorTimeMixW0, "blk.%d.time_mix_w0"},
    {cLlmTensorTimeMixA0, "blk.%d.time_mix_a0"},
    {cLlmTensorTimeMixA1, "blk.%d.time_mix_a1"},
    {cLlmTensorTimeMixA2, "blk.%d.time_mix_a2"},
    {cLlmTensorTimeMixV0, "blk.%d.time_mix_v0"},
    {cLlmTensorTimeMixV1, "blk.%d.time_mix_v1"},
    {cLlmTensorTimeMixV2, "blk.%d.time_mix_v2"},
    {cLlmTensorTimeMixG1, "blk.%d.time_mix_g1"},
    {cLlmTensorTimeMixG2, "blk.%d.time_mix_g2"},
    {cLlmTensorTimeMixKK, "blk.%d.time_mix_k_k"},
    {cLlmTensorTimeMixKA, "blk.%d.time_mix_k_a"},
    {cLlmTensorTimeMixRK, "blk.%d.time_mix_r_k"},
    {cLlmTensorConv1d, "conv1d"},
    {cLlmTensorConvnextDw, "convnext.%d.dw"},
    {cLlmTensorConvnextNorm, "convnext.%d.norm"},
    {cLlmTensorConvnextPw1, "convnext.%d.pw1"},
    {cLlmTensorConvnextPw2, "convnext.%d.pw2"},
    {cLlmTensorConvnextGamma, "convnext.%d.gamma"},
    {cLlmTensorPosNetConv1, "posnet.%d.conv1"},
    {cLlmTensorPosNetConv2, "posnet.%d.conv2"},
    {cLlmTensorPosNetNorm, "posnet.%d.norm"},
    {cLlmTensorPosNetNorm1, "posnet.%d.norm1"},
    {cLlmTensorPosNetNorm2, "posnet.%d.norm2"},
    {cLlmTensorPosNetAttnNorm, "posnet.%d.attn_norm"},
    {cLlmTensorPosNetAttnQ, "posnet.%d.attn_q"},
    {cLlmTensorPosNetAttnK, "posnet.%d.attn_k"},
    {cLlmTensorPosNetAttnV, "posnet.%d.attn_v"},
    {cLlmTensorPosNetAttnOut, "posnet.%d.attn_output"},
    {cLlmTensorAttnSinks, "blk.%d.attn_sinks"},
    {cLlmTensorShortconvConv, "blk.%d.shortconv.conv"},
    {cLlmTensorShortconvInproj, "blk.%d.shortconv.in_proj"},
    {cLlmTensorShortconvOutproj, "blk.%d.shortconv.out_proj"},
    {cLlmTensorFfnGateChexps, "blk.%d.ffn_gate_chexps"},
    {cLlmTensorFfnDownChexps, "blk.%d.ffn_down_chexps"},
    {cLlmTensorFfnUpChexps, "blk.%d.ffn_up_chexps"},
    {cLlmTensorVisexpAttnQkv, "blk.%d.vis_attn_qkv"},
    {cLlmTensorVisexpAttnOut, "blk.%d.vis_attn_output"},
    {cLlmTensorVisexpFfnGate, "blk.%d.vis_gate"},
    {cLlmTensorVisexpFfnDown, "blk.%d.vis_down"},
    {cLlmTensorVisexpFfnUp, "blk.%d.vis_up"},
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
    {cLlmTensorFc, "fc"},
    {cLlmTensorD2t, "d2t"},
    {cLlmTensorDsparkMarkovW1, "markov_w1"},
    {cLlmTensorDsparkMarkovW2, "markov_w2"},
    {cLlmTensorDsparkConfProj, "conf_proj"},
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
    {cLlmTensorTokenEmbdNorm,
     {cLlmTensorLayerRepeating, GGML_OP_MUL}}, // do the norms on the first layer (not the input layer)
    {cLlmTensorOutput, {cLlmTensorLayerOutput, GGML_OP_MUL_MAT}},
    {cLlmTensorCls, {cLlmTensorLayerOutput, GGML_OP_MUL_MAT}},
    {cLlmTensorClsOut, {cLlmTensorLayerOutput, GGML_OP_MUL_MAT}},
    {cLlmTensorClsNorm, {cLlmTensorLayerOutput, GGML_OP_MUL}},
    {cLlmTensorDense2Out, {cLlmTensorLayerOutput, GGML_OP_MUL_MAT}}, // Dense layer output
    {cLlmTensorDense3Out, {cLlmTensorLayerOutput, GGML_OP_MUL_MAT}}, // Dense layer output
    {cLlmTensorOutputNorm, {cLlmTensorLayerOutput, GGML_OP_MUL}},
    {cLlmTensorOutputNormLfm2, {cLlmTensorLayerOutput, GGML_OP_MUL}},
    {cLlmTensorDecOutputNorm, {cLlmTensorLayerOutput, GGML_OP_MUL}},
    {cLlmTensorEncOutputNorm, {cLlmTensorLayerOutput, GGML_OP_MUL}},
    {cLlmTensorRopeFreqs, {cLlmTensorLayerRepeating, GGML_OP_ROPE}},
    {cLlmTensorRopeFactorsLong, {cLlmTensorLayerRepeating, GGML_OP_ROPE}},
    {cLlmTensorRopeFactorsShort, {cLlmTensorLayerRepeating, GGML_OP_ROPE}},
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
    {cLlmTensorDecAttnQ, {cLlmTensorLayerRepeating, GGML_OP_MUL_MAT}},
    {cLlmTensorDecAttnK, {cLlmTensorLayerRepeating, GGML_OP_MUL_MAT}},
    {cLlmTensorDecAttnV, {cLlmTensorLayerRepeating, GGML_OP_MUL_MAT}},
    {cLlmTensorDecAttnOut, {cLlmTensorLayerRepeating, GGML_OP_MUL_MAT}},
    {cLlmTensorDecCrossAttnQ, {cLlmTensorLayerRepeating, GGML_OP_MUL_MAT}},
    {cLlmTensorDecCrossAttnK, {cLlmTensorLayerRepeating, GGML_OP_MUL_MAT}},
    {cLlmTensorDecCrossAttnV, {cLlmTensorLayerRepeating, GGML_OP_MUL_MAT}},
    {cLlmTensorDecCrossAttnOut, {cLlmTensorLayerRepeating, GGML_OP_MUL_MAT}},
    {cLlmTensorDecFfnGate, {cLlmTensorLayerRepeating, GGML_OP_MUL_MAT}},
    {cLlmTensorDecFfnDown, {cLlmTensorLayerRepeating, GGML_OP_MUL_MAT}},
    {cLlmTensorDecFfnUp, {cLlmTensorLayerRepeating, GGML_OP_MUL_MAT}},
    {cLlmTensorEncAttnQ, {cLlmTensorLayerRepeating, GGML_OP_MUL_MAT}},
    {cLlmTensorEncAttnK, {cLlmTensorLayerRepeating, GGML_OP_MUL_MAT}},
    {cLlmTensorEncAttnV, {cLlmTensorLayerRepeating, GGML_OP_MUL_MAT}},
    {cLlmTensorEncAttnOut, {cLlmTensorLayerRepeating, GGML_OP_MUL_MAT}},
    {cLlmTensorEncFfnGate, {cLlmTensorLayerRepeating, GGML_OP_MUL_MAT}},
    {cLlmTensorEncFfnDown, {cLlmTensorLayerRepeating, GGML_OP_MUL_MAT}},
    {cLlmTensorEncFfnUp, {cLlmTensorLayerRepeating, GGML_OP_MUL_MAT}},
    {cLlmTensorFfnGateInpShexp, {cLlmTensorLayerRepeating, GGML_OP_MUL_MAT}},
    {cLlmTensorFfnGateInp, {cLlmTensorLayerRepeating, GGML_OP_MUL_MAT}},
    {cLlmTensorSsmIn, {cLlmTensorLayerRepeating, GGML_OP_MUL_MAT}},
    {cLlmTensorSsmX, {cLlmTensorLayerRepeating, GGML_OP_MUL_MAT}},
    {cLlmTensorSsmDt, {cLlmTensorLayerRepeating, GGML_OP_MUL_MAT}},
    {cLlmTensorSsmOut, {cLlmTensorLayerRepeating, GGML_OP_MUL_MAT}},
    {cLlmTensorSsmAlpha, {cLlmTensorLayerRepeating, GGML_OP_MUL_MAT}},
    {cLlmTensorSsmBetaAlpha, {cLlmTensorLayerRepeating, GGML_OP_MUL_MAT}},
    {cLlmTensorTimeMixW1, {cLlmTensorLayerRepeating, GGML_OP_MUL_MAT}},
    {cLlmTensorTimeMixW2, {cLlmTensorLayerRepeating, GGML_OP_MUL_MAT}},
    {cLlmTensorTimeMixA1, {cLlmTensorLayerRepeating, GGML_OP_MUL_MAT}},
    {cLlmTensorTimeMixA2, {cLlmTensorLayerRepeating, GGML_OP_MUL_MAT}},
    {cLlmTensorTimeMixV1, {cLlmTensorLayerRepeating, GGML_OP_MUL_MAT}},
    {cLlmTensorTimeMixV2, {cLlmTensorLayerRepeating, GGML_OP_MUL_MAT}},
    {cLlmTensorTimeMixG1, {cLlmTensorLayerRepeating, GGML_OP_MUL_MAT}},
    {cLlmTensorTimeMixG2, {cLlmTensorLayerRepeating, GGML_OP_MUL_MAT}},
    {cLlmTensorTimeMixDecayW1, {cLlmTensorLayerRepeating, GGML_OP_MUL_MAT}},
    {cLlmTensorTimeMixDecayW2, {cLlmTensorLayerRepeating, GGML_OP_MUL_MAT}},
    {cLlmTensorTimeMixKey, {cLlmTensorLayerRepeating, GGML_OP_MUL_MAT}},
    {cLlmTensorTimeMixValue, {cLlmTensorLayerRepeating, GGML_OP_MUL_MAT}},
    {cLlmTensorTimeMixReceptance, {cLlmTensorLayerRepeating, GGML_OP_MUL_MAT}},
    {cLlmTensorTimeMixGate, {cLlmTensorLayerRepeating, GGML_OP_MUL_MAT}},
    {cLlmTensorTimeMixOutput, {cLlmTensorLayerRepeating, GGML_OP_MUL_MAT}},
    {cLlmTensorChannelMixKey, {cLlmTensorLayerRepeating, GGML_OP_MUL_MAT}},
    {cLlmTensorChannelMixReceptance, {cLlmTensorLayerRepeating, GGML_OP_MUL_MAT}},
    {cLlmTensorChannelMixValue, {cLlmTensorLayerRepeating, GGML_OP_MUL_MAT}},
    {cLlmTensorFfnAct, {cLlmTensorLayerRepeating, GGML_OP_DIV}},
    {cLlmTensorSsmConv1d, {cLlmTensorLayerRepeating, GGML_OP_SSM_CONV}},
    {cLlmTensorSsmA, {cLlmTensorLayerRepeating, GGML_OP_SSM_SCAN}},
    {cLlmTensorSsmANoscan,
     {cLlmTensorLayerRepeating, GGML_OP_MUL}}, // a version of SSM_A used for MUL instead of SSM_SCAN
    {cLlmTensorSsmDtNorm, {cLlmTensorLayerRepeating, GGML_OP_MUL}},
    {cLlmTensorSsmBNorm, {cLlmTensorLayerRepeating, GGML_OP_MUL}},
    {cLlmTensorSsmCNorm, {cLlmTensorLayerRepeating, GGML_OP_MUL}},
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
    {cLlmTensorTimeMixLerpX, {cLlmTensorLayerRepeating, GGML_OP_MUL}},
    {cLlmTensorTimeMixLn, {cLlmTensorLayerRepeating, GGML_OP_MUL}},
    {cLlmTensorChannelMixLerpK, {cLlmTensorLayerRepeating, GGML_OP_MUL}},
    {cLlmTensorChannelMixLerpR, {cLlmTensorLayerRepeating, GGML_OP_MUL}},
    {cLlmTensorTimeMixKK, {cLlmTensorLayerRepeating, GGML_OP_MUL}},
    {cLlmTensorTimeMixKA, {cLlmTensorLayerRepeating, GGML_OP_MUL}},
    {cLlmTensorTimeMixRK, {cLlmTensorLayerRepeating, GGML_OP_MUL}},
    {cLlmTensorTimeMixLerpW, {cLlmTensorLayerRepeating, GGML_OP_ADD}},
    {cLlmTensorTimeMixLerpK, {cLlmTensorLayerRepeating, GGML_OP_ADD}},
    {cLlmTensorTimeMixLerpV, {cLlmTensorLayerRepeating, GGML_OP_ADD}},
    {cLlmTensorTimeMixLerpR, {cLlmTensorLayerRepeating, GGML_OP_ADD}},
    {cLlmTensorTimeMixLerpG, {cLlmTensorLayerRepeating, GGML_OP_ADD}},
    {cLlmTensorTimeMixLerpFused, {cLlmTensorLayerRepeating, GGML_OP_ADD}},
    {cLlmTensorTimeMixDecay, {cLlmTensorLayerRepeating, GGML_OP_ADD}},
    {cLlmTensorTimeMixW0, {cLlmTensorLayerRepeating, GGML_OP_ADD}},
    {cLlmTensorTimeMixA0, {cLlmTensorLayerRepeating, GGML_OP_ADD}},
    {cLlmTensorTimeMixV0, {cLlmTensorLayerRepeating, GGML_OP_ADD}},
    {cLlmTensorTimeMixFirst, {cLlmTensorLayerRepeating, GGML_OP_RWKV_WKV6}},
    {cLlmTensorAttnNorm, {cLlmTensorLayerRepeating, GGML_OP_MUL}},
    {cLlmTensorAttnNorm2, {cLlmTensorLayerRepeating, GGML_OP_MUL}},
    {cLlmTensorAttnOutNorm, {cLlmTensorLayerRepeating, GGML_OP_MUL}},
    {cLlmTensorAttnPostNorm, {cLlmTensorLayerRepeating, GGML_OP_MUL}},
    {cLlmTensorFfnNorm, {cLlmTensorLayerRepeating, GGML_OP_MUL}},
    {cLlmTensorFfnPostNorm, {cLlmTensorLayerRepeating, GGML_OP_MUL}},
    {cLlmTensorFfnNormExps, {cLlmTensorLayerRepeating, GGML_OP_MUL}},
    {cLlmTensorAttnQNorm, {cLlmTensorLayerRepeating, GGML_OP_MUL}},
    {cLlmTensorAttnKNorm, {cLlmTensorLayerRepeating, GGML_OP_MUL}},
    {cLlmTensorLayerOutNorm, {cLlmTensorLayerRepeating, GGML_OP_MUL}},
    {cLlmTensorAttnQANorm, {cLlmTensorLayerRepeating, GGML_OP_MUL}},
    {cLlmTensorAttnKvANorm, {cLlmTensorLayerRepeating, GGML_OP_MUL}},
    {cLlmTensorAttnSubNorm, {cLlmTensorLayerRepeating, GGML_OP_MUL}},
    {cLlmTensorFfnSubNorm, {cLlmTensorLayerRepeating, GGML_OP_MUL}},
    {cLlmTensorDecAttnNorm, {cLlmTensorLayerRepeating, GGML_OP_MUL}},
    {cLlmTensorDecCrossAttnNorm, {cLlmTensorLayerRepeating, GGML_OP_MUL}},
    {cLlmTensorDecFfnNorm, {cLlmTensorLayerRepeating, GGML_OP_MUL}},
    {cLlmTensorEncAttnNorm, {cLlmTensorLayerRepeating, GGML_OP_MUL}},
    {cLlmTensorEncFfnNorm, {cLlmTensorLayerRepeating, GGML_OP_MUL}},
    {cLlmTensorDecAttnRelB, {cLlmTensorLayerRepeating, GGML_OP_GET_ROWS}},
    {cLlmTensorEncAttnRelB, {cLlmTensorLayerRepeating, GGML_OP_GET_ROWS}},
    {cLlmTensorFfnDownExps, {cLlmTensorLayerRepeating, GGML_OP_MUL_MAT_ID}},
    {cLlmTensorFfnGateExps, {cLlmTensorLayerRepeating, GGML_OP_MUL_MAT_ID}},
    {cLlmTensorFfnUpExps, {cLlmTensorLayerRepeating, GGML_OP_MUL_MAT_ID}},
    {cLlmTensorFfnGateUpExps, {cLlmTensorLayerRepeating, GGML_OP_MUL_MAT_ID}},
    {cLlmTensorFfnDownChexps, {cLlmTensorLayerRepeating, GGML_OP_MUL_MAT_ID}},
    {cLlmTensorFfnGateChexps, {cLlmTensorLayerRepeating, GGML_OP_MUL_MAT_ID}},
    {cLlmTensorFfnUpChexps, {cLlmTensorLayerRepeating, GGML_OP_MUL_MAT_ID}},
    {cLlmTensorFfnExpProbsB, {cLlmTensorLayerRepeating, GGML_OP_ADD}},
    // per-layer token embeddings (qwen4exp)
    {cLlmTensorPerLayerTokenEmbd, {cLlmTensorLayerInput, GGML_OP_GET_ROWS}},
    // altup / laurel (gemma 3n)
    {cLlmTensorAltupProj, {cLlmTensorLayerOutput, GGML_OP_MUL_MAT}},
    {cLlmTensorAltupUnembdProj, {cLlmTensorLayerOutput, GGML_OP_MUL_MAT}},
    {cLlmTensorAltupCorrectCoef, {cLlmTensorLayerRepeating, GGML_OP_MUL_MAT}},
    {cLlmTensorAltupCorrectScale, {cLlmTensorLayerRepeating, GGML_OP_MUL}},
    {cLlmTensorAltupPredictCoef, {cLlmTensorLayerRepeating, GGML_OP_MUL_MAT}},
    {cLlmTensorAltupRouter, {cLlmTensorLayerRepeating, GGML_OP_MUL_MAT}},
    {cLlmTensorAltupRouterNorm, {cLlmTensorLayerRepeating, GGML_OP_MUL}},
    {cLlmTensorLaurelL, {cLlmTensorLayerRepeating, GGML_OP_MUL_MAT}},
    {cLlmTensorLaurelR, {cLlmTensorLayerRepeating, GGML_OP_MUL_MAT}},
    {cLlmTensorLaurelPostNorm, {cLlmTensorLayerRepeating, GGML_OP_MUL}},
    // this tensor is loaded for T5, but never used
    {cLlmTensorDecCrossAttnRelB, {cLlmTensorLayerRepeating, GGML_OP_NONE}},
    {cLlmTensorConv1d, {cLlmTensorLayerRepeating, GGML_OP_IM2COL}},
    {cLlmTensorPosNetNorm, {cLlmTensorLayerRepeating, GGML_OP_MUL}},
    {cLlmTensorPosNetNorm1, {cLlmTensorLayerRepeating, GGML_OP_MUL}},
    {cLlmTensorPosNetNorm2, {cLlmTensorLayerRepeating, GGML_OP_MUL}},
    {cLlmTensorPosNetConv1, {cLlmTensorLayerRepeating, GGML_OP_IM2COL}},
    {cLlmTensorPosNetConv2, {cLlmTensorLayerRepeating, GGML_OP_IM2COL}},
    {cLlmTensorPosNetAttnNorm, {cLlmTensorLayerRepeating, GGML_OP_MUL}},
    {cLlmTensorPosNetAttnQ, {cLlmTensorLayerRepeating, GGML_OP_MUL_MAT}},
    {cLlmTensorPosNetAttnK, {cLlmTensorLayerRepeating, GGML_OP_MUL_MAT}},
    {cLlmTensorPosNetAttnV, {cLlmTensorLayerRepeating, GGML_OP_MUL_MAT}},
    {cLlmTensorPosNetAttnOut, {cLlmTensorLayerRepeating, GGML_OP_MUL_MAT}},
    {cLlmTensorConvnextDw, {cLlmTensorLayerRepeating, GGML_OP_IM2COL}},
    {cLlmTensorConvnextNorm, {cLlmTensorLayerRepeating, GGML_OP_MUL}},
    {cLlmTensorConvnextPw1, {cLlmTensorLayerRepeating, GGML_OP_MUL_MAT}},
    {cLlmTensorConvnextPw2, {cLlmTensorLayerRepeating, GGML_OP_MUL_MAT}},
    {cLlmTensorConvnextGamma, {cLlmTensorLayerRepeating, GGML_OP_MUL}},
    {cLlmTensorShortconvConv, {cLlmTensorLayerRepeating, GGML_OP_SSM_CONV}},
    {cLlmTensorShortconvInproj, {cLlmTensorLayerRepeating, GGML_OP_MUL_MAT}},
    {cLlmTensorShortconvOutproj, {cLlmTensorLayerRepeating, GGML_OP_MUL_MAT}},
    {cLlmTensorVisexpAttnQkv, {cLlmTensorLayerRepeating, GGML_OP_MUL_MAT}},
    {cLlmTensorVisexpAttnOut, {cLlmTensorLayerRepeating, GGML_OP_MUL_MAT}},
    {cLlmTensorVisexpFfnGate, {cLlmTensorLayerRepeating, GGML_OP_MUL_MAT}},
    {cLlmTensorVisexpFfnDown, {cLlmTensorLayerRepeating, GGML_OP_MUL_MAT}},
    {cLlmTensorVisexpFfnUp, {cLlmTensorLayerRepeating, GGML_OP_MUL_MAT}},
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
    // eagle3
    {cLlmTensorFc, {cLlmTensorLayerOutput, GGML_OP_MUL_MAT}},
    {cLlmTensorD2t, {cLlmTensorLayerOutput, GGML_OP_GET_ROWS}},
    // dspark
    {cLlmTensorDsparkMarkovW1, {cLlmTensorLayerOutput, GGML_OP_GET_ROWS}},
    {cLlmTensorDsparkMarkovW2, {cLlmTensorLayerOutput, GGML_OP_MUL_MAT}},
    {cLlmTensorDsparkConfProj, {cLlmTensorLayerOutput, GGML_OP_MUL_MAT}},
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

bool fLlmArchIsRecurrent(const llm_arch &pArch) {
  switch (pArch) {
  case cLlmArchMamba:
  case cLlmArchMamba2:
  case cLlmArchRwkv6:
  case cLlmArchRwkv6qwen2:
  case cLlmArchRwkv7:
  case cLlmArchArwkv7:
    return true;
  default:
    return false;
  }
}

bool fLlmArchIsHybrid(const llm_arch &pArch) {
  switch (pArch) {
  case cLlmArchJamba:
  case cLlmArchFalconH1:
  case cLlmArchPlamo2:
  case cLlmArchGraniteHybrid:
  case cLlmArchLfm2:
  case cLlmArchLfm2moe:
  case cLlmArchNemotronH:
  case cLlmArchNemotronHMoe:
  case cLlmArchQwen3next:
  case cLlmArchKimiLinear:
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

bool fLlmArchIsDiffusion(const llm_arch &pArch) {
  switch (pArch) {
  case cLlmArchDream:
  case cLlmArchLlada:
  case cLlmArchLladaMoe:
  case cLlmArchRnd1:
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
  case cLlmArchGrok:
  case cLlmArchMpt:
  case cLlmArchPlamo2:
  case cLlmArchMinicpm3:
  case cLlmArchGemma3n:
  case cLlmArchMamba:
  case cLlmArchMamba2:
  case cLlmArchJamba:
  case cLlmArchFalconH1:
  case cLlmArchOlmo2:
  case cLlmArchOlmoe:
  case cLlmArchGlmDsa:
  case cLlmArchBitnet:
  case cLlmArchT5:
  case cLlmArchNemotronH:
  case cLlmArchNemotronHMoe:
  case cLlmArchGraniteHybrid:
  case cLlmArchLfm2:
  case cLlmArchLfm2moe:
  case cLlmArchMinimaxM2:
  case cLlmArchMinimaxM3:
  case cLlmArchKimiLinear:
  case cLlmArchKimiK3:
  case cLlmArchGlm5Next:
    return false;
  default:
    return true;
  }
}
