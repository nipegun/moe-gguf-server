#pragma once

#include "ggml.h"
#include "gguf.h"
#include "clip.h"

#include <array>
#include <climits>
#include <cstdarg>
#include <cinttypes>
#include <string>
#include <map>
#include <sstream>
#include <vector>
#include <memory>
#include <fstream>

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#endif

// Internal header for clip.cpp

#define MTMD_INTERNAL_HEADER

#define KEY_FTYPE "general.file_type"
#define KEY_NAME "general.name"
#define KEY_DESCRIPTION "general.description"
#define KEY_PROJ_TYPE "clip.projector_type"
#define KEY_HAS_AUDIO_ENC "clip.has_audio_encoder"
#define KEY_HAS_VISION_ENC "clip.has_vision_encoder"
#define KEY_USE_GELU "clip.use_gelu"
#define KEY_USE_SILU "clip.use_silu"

#define KEY_N_EMBD "clip.%s.embedding_length"
#define KEY_N_FF "clip.%s.feed_forward_length"
#define KEY_N_BLOCK "clip.%s.block_count"
#define KEY_PROJ_DIM "clip.%s.projection_dim"
#define KEY_N_HEAD "clip.%s.attention.head_count"
#define KEY_N_HEAD_KV "clip.%s.attention.head_count_kv"
#define KEY_N_EMBD_HEAD "clip.%s.attention.head_dim"
#define KEY_LAYER_NORM_EPS "clip.%s.attention.layer_norm_epsilon"
#define KEY_FEATURE_LAYERS "clip.%s.feature_layer"

// vision-specific
#define KEY_VISION_PROJ_TYPE "clip.vision.projector_type" // for models with mixed modalities
#define KEY_IMAGE_SIZE "clip.vision.image_size"
#define KEY_IMAGE_MIN_PIXELS "clip.vision.image_min_pixels"
#define KEY_IMAGE_MAX_PIXELS "clip.vision.image_max_pixels"
#define KEY_PREPROC_MIN_TILES "clip.vision.preproc_min_tiles"
#define KEY_PREPROC_MAX_TILES "clip.vision.preproc_max_tiles"
#define KEY_PREPROC_IMAGE_SIZE "clip.vision.preproc_image_size"
#define KEY_PATCH_SIZE "clip.vision.patch_size"
#define KEY_IMAGE_MEAN "clip.vision.image_mean"
#define KEY_IMAGE_STD "clip.vision.image_std"
#define KEY_PROJ_SCALE_FACTOR "clip.vision.projector.scale_factor"
#define KEY_PROJ_SAMPLE_QUERY_SIDE "clip.vision.projector.query_side"
#define KEY_PROJ_SAMPLE_WINDOW_SIDE "clip.vision.projector.window_side"
#define KEY_PROJ_SPATIAL_OFFSETS "clip.vision.projector.spatial_offsets"
#define KEY_SPATIAL_MERGE_SIZE "clip.vision.spatial_merge_size"
#define KEY_SWIGLU_CLAMP "clip.vision.swiglu_clamp"

#define KEY_IMAGE_GRID_PINPOINTS "clip.vision.image_grid_pinpoints"
#define KEY_WIN_ATTN_PATTERN "clip.vision.n_wa_pattern"
#define KEY_WIN_ATTN_LAYER_INDEXES "clip.vision.wa_layer_indexes"
#define KEY_WA_PATTERN_MODE "clip.vision.wa_pattern_mode"
#define KEY_ATTN_WINDOW_SIZE "clip.vision.window_size"
// audio-specific
#define KEY_AUDIO_PROJ_TYPE "clip.audio.projector_type" // for models with mixed modalities
#define KEY_A_NUM_MEL_BINS "clip.audio.num_mel_bins"
#define KEY_A_PROJ_STACK_FACTOR "clip.audio.projector.stack_factor"
#define KEY_A_CHUNK_SIZE "clip.audio.chunk_size"
#define KEY_A_CONV_KERNEL_SIZE "clip.audio.conv_kernel_size"
#define KEY_A_MAX_POS_EMB "clip.audio.max_pos_emb"
#define KEY_A_PROJ_WINDOW_SIZE "clip.audio.projector.window_size"
#define KEY_A_PROJ_DOWNSAMPLE_RATE "clip.audio.projector.downsample_rate"
#define KEY_A_PROJ_HEAD_COUNT "clip.audio.projector.head_count"
#define KEY_A_RVQ_NUM_QUANTIZERS "clip.audio.rvq.num_quantizers" // mimo-audio-tokenizer
#define KEY_A_RVQ_CODEBOOK_SIZE "clip.audio.rvq.codebook_size"   // mimo-audio-tokenizer: per-quantizer bin count
#define KEY_A_WA_PATTERN_MODE "clip.audio.wa_pattern_mode" // mimo-audio-tokenizer, per-layer -1 (full) / 0 (windowed)
#define KEY_A_ATTN_WINDOW_SIZE "clip.audio.window_size"    // mimo-audio-tokenizer: sliding-window radius
#define KEY_A_LOCAL_BLOCK_COUNT "clip.audio.local_block_count" // mimo-v2.5: input_local_transformer layer count
#define KEY_A_LOCAL_GROUP_SIZE "clip.audio.local_group_size"   // mimo-v2.5: input_local_transformer grouping size
#define KEY_AUDIO_SUBSAMPLING_FACTOR "clip.audio.subsampling_factor"

//
// tensor name constants
//

#define TN_POS_EMBD "%s.position_embd.weight"
#define TN_CLASS_EMBD "v.class_embd"
#define TN_PATCH_EMBD "v.patch_embd.weight"     // first (or only) patch embedding kernel, without a ".0" suffix
#define TN_PATCH_EMBD_1 "v.patch_embd.weight.1" // second temporal kernel of a split Conv3D (Qwen2-VL family)
#define TN_PATCH_BIAS "v.patch_embd.bias"
#define TN_NORM_EMBD "v.norm_embd.%s"
#define TN_ATTN_QKV "%s.blk.%d.attn_qkv.%s"
#define TN_ATTN_K "%s.blk.%d.attn_k.%s"
#define TN_ATTN_Q "%s.blk.%d.attn_q.%s"
#define TN_ATTN_V "%s.blk.%d.attn_v.%s"
#define TN_ATTN_OUTPUT "%s.blk.%d.attn_out.%s"
#define TN_ATTN_SINKS "%s.blk.%d.attn_sinks"
#define TN_ATTN_K_NORM "%s.blk.%d.attn_k_norm.%s"
#define TN_ATTN_Q_NORM "%s.blk.%d.attn_q_norm.%s"
#define TN_FFN_DOWN "%s.blk.%d.ffn_down.%s"
#define TN_FFN_GATE "%s.blk.%d.ffn_gate.%s"
#define TN_FFN_UP "%s.blk.%d.ffn_up.%s"
#define TN_FFN_GATE "%s.blk.%d.ffn_gate.%s"
#define TN_LN_1 "%s.blk.%d.ln1.%s"                      // layer norm
#define TN_LN_2 "%s.blk.%d.ln2.%s"                      // layer norm
#define TN_LS_1 "%s.blk.%d.ls1.%s"                      // layer scale
#define TN_LS_2 "%s.blk.%d.ls2.%s"                      // layer scale
#define TN_LS_OUT "%s.blk.%d.out_scale.%s"              // layer out scale
#define TN_ATTN_POST_NORM "%s.blk.%d.attn_post_norm.%s" // post-attn norm
#define TN_FFN_POST_NORM "%s.blk.%d.ffn_post_norm.%s"   // post-FFN norm
#define TN_LN_PRE "%s.pre_ln.%s"
#define TN_LN_POST "%s.post_ln.%s"
#define TN_LLAVA_PROJ "mm.%d.%s"
#define TN_MM_UP "mm.up.%s"
#define TN_MM_GATE "mm.gate.%s"
#define TN_MM_DOWN "mm.down.%s"
#define TN_MM_POST_NORM "mm.post_norm.%s"
#define TN_MVLM_PROJ_MLP "mm.model.mlp.%d.%s"
#define TN_IMAGE_NEWLINE "v.image_newline"
#define TN_IMAGE_SEPERATOR "v.view_seperator"
#define TN_MM_INP_NORM "mm.input_norm.weight"
#define TN_MM_INP_NORM_B "mm.input_norm.bias"
#define TN_MM_INP_PROJ "mm.input_projection.weight" // gemma3
#define TN_MM_SOFT_EMB_N "mm.soft_emb_norm.weight"  // gemma3
#define TN_MM_PROJECTOR "mm.model.fc.%s"            // idefics3
#define TN_MM_PATCH_MERGER "mm.patch_merger.%s"     // mistral small 3.1, glm4v
#define TN_MM_MERGER_FC1 "mm.merger.fc1.%s"         // minimax-m3 patch-merge MLP
#define TN_MM_MERGER_FC2 "mm.merger.fc2.%s"
#define TN_TOK_IMG_BREAK "v.token_embd.img_break"  // pixtral
#define TN_DEEPSTACK_NORM "v.deepstack.%d.norm.%s" // qwen3vl deepstack
#define TN_DEEPSTACK_FC1 "v.deepstack.%d.fc1.%s"   // qwen3vl deepstack
#define TN_DEEPSTACK_FC2 "v.deepstack.%d.fc2.%s"   // qwen3vl deepstack

// MiniCPM-V 4.6 ViT merger (window attention + MLP downsample),
// matching the upstream `vit_merger` module name in transformers.
#define TN_VIT_MERGER_LN1 "v.vit_merger.ln1.%s"
#define TN_VIT_MERGER_ATTN_Q "v.vit_merger.attn_q.%s"
#define TN_VIT_MERGER_ATTN_K "v.vit_merger.attn_k.%s"
#define TN_VIT_MERGER_ATTN_V "v.vit_merger.attn_v.%s"
#define TN_VIT_MERGER_ATTN_O "v.vit_merger.attn_out.%s"
#define TN_VIT_MERGER_DS_LN "v.vit_merger.ds_ln.%s"
#define TN_VIT_MERGER_DS_UP "v.vit_merger.ds_ffn_up.%s"
#define TN_VIT_MERGER_DS_DOWN "v.vit_merger.ds_ffn_down.%s"

// ultravox
#define TN_CONV1D "a.conv1d.%d.%s"
#define TN_CONV2D "a.conv2d.%d.%s"
#define TN_CONV_OUT "a.conv_out.%s"
#define TN_MM_AUDIO_MLP "mm.a.mlp.%d.%s"
#define TN_MM_AUDIO_FC "mm.a.fc.%s" // fully connected layer
#define TN_MM_NORM_PRE "mm.a.norm_pre.%s"
#define TN_MM_NORM_MID "mm.a.norm_mid.%s"

// mimo-audio-tokenizer
#define TN_A_DOWNSAMPLE_CONV "a.downsample.conv.%s"
#define TN_A_DOWNSAMPLE_NORM "a.downsample.norm.%s"
#define TN_A_RVQ_CODEBOOK "a.rvq.codebook.%s"
// mimo-v2.5: text-side RVQ code embedding ("text codebook")
#define TN_MM_A_CODE_EMBD "mm.a.code_embd.%s"
// mimo-v2.5: LLM-side connector (input_local_transformer)
#define TN_MM_A_LOCAL_ATTN_Q "mm.a.local_blk.%d.attn_q.%s"
#define TN_MM_A_LOCAL_ATTN_K "mm.a.local_blk.%d.attn_k.%s"
#define TN_MM_A_LOCAL_ATTN_V "mm.a.local_blk.%d.attn_v.%s"
#define TN_MM_A_LOCAL_ATTN_OUT "mm.a.local_blk.%d.attn_out.%s"
#define TN_MM_A_LOCAL_FFN_GATE "mm.a.local_blk.%d.ffn_gate.%s"
#define TN_MM_A_LOCAL_FFN_UP "mm.a.local_blk.%d.ffn_up.%s"
#define TN_MM_A_LOCAL_FFN_DOWN "mm.a.local_blk.%d.ffn_down.%s"
#define TN_MM_A_LOCAL_LN1 "mm.a.local_blk.%d.ln1.%s"
#define TN_MM_A_LOCAL_LN2 "mm.a.local_blk.%d.ln2.%s"
#define TN_MM_A_LOCAL_NORM "mm.a.local_norm.%s"

// cogvlm
#define TN_MM_POST_FC_NORM "mm.post_fc_norm.%s"
#define TN_MM_H_TO_4H "mm.up.%s"
#define TN_MM_GATE "mm.gate.%s"
#define TN_MM_4H_TO_H "mm.down.%s"
#define TN_TOK_BOI "v.boi"
#define TN_TOK_EOI "v.eoi"

// hunyuanvl (shared GGUF tensor names)
#define TN_MM_PRE_NORM "mm.pre_norm.%s"
#define TN_TOK_IMG_BEGIN "mm.image_begin"
#define TN_TOK_IMG_END "mm.image_end"

// (conformer) lfm2
#define TN_PRE_ENCODE_OUT "a.pre_encode.out.%s"
#define TN_FFN_NORM "%s.blk.%d.ffn_norm.%s"
#define TN_FFN_NORM_1 "%s.blk.%d.ffn_norm_1.%s"
#define TN_FFN_UP_1 "%s.blk.%d.ffn_up_1.%s"
#define TN_FFN_DOWN_1 "%s.blk.%d.ffn_down_1.%s"
#define TN_POS_BIAS_U "%s.blk.%d.pos_bias_u"
#define TN_POS_BIAS_V "%s.blk.%d.pos_bias_v"
#define TN_NORM_CONV "%s.blk.%d.norm_conv.%s"
#define TN_LINEAR_POS "%s.blk.%d.linear_pos.%s"
#define TN_CONV_DW "%s.blk.%d.conv_dw.%s"
#define TN_CONV_NORM "%s.blk.%d.conv_norm.%s"
#define TN_CONV_PW1 "%s.blk.%d.conv_pw1.%s"
#define TN_CONV_PW2 "%s.blk.%d.conv_pw2.%s"
#define TN_INP_PROJ "a.input_projection.%s"
#define TN_CTC_OUT "a.enc_ctc_out.%s"
#define TN_CTC_OUT_MID "a.enc_ctc_out_mid.%s"
#define TN_ATTN_REL_POS_EMB "%s.blk.%d.attn_rel_pos_emb"
// qformer projector
#define TN_QF_PROJ_QUERY "%s.proj_query"
#define TN_QF_PROJ_NORM "%s.proj_norm.%s"
#define TN_QF_PROJ_LINEAR "%s.proj_linear.%s"
#define TN_QF_SELF_ATTN_Q "%s.proj_blk.%d.self_attn_q.%s"
#define TN_QF_SELF_ATTN_K "%s.proj_blk.%d.self_attn_k.%s"
#define TN_QF_SELF_ATTN_V "%s.proj_blk.%d.self_attn_v.%s"
#define TN_QF_SELF_ATTN_O "%s.proj_blk.%d.self_attn_out.%s"
#define TN_QF_SELF_ATTN_N "%s.proj_blk.%d.self_attn_norm.%s"
#define TN_QF_CROSS_ATTN_Q "%s.proj_blk.%d.cross_attn_q.%s"
#define TN_QF_CROSS_ATTN_K "%s.proj_blk.%d.cross_attn_k.%s"
#define TN_QF_CROSS_ATTN_V "%s.proj_blk.%d.cross_attn_v.%s"
#define TN_QF_CROSS_ATTN_O "%s.proj_blk.%d.cross_attn_out.%s"
#define TN_QF_CROSS_ATTN_N "%s.proj_blk.%d.cross_attn_norm.%s"
#define TN_QF_FFN_UP "%s.proj_blk.%d.ffn_up.%s"
#define TN_QF_FFN_DOWN "%s.proj_blk.%d.ffn_down.%s"
#define TN_QF_FFN_NORM "%s.proj_blk.%d.ffn_norm.%s"
// multi-projector qformer (bid => projector ID)
#define TN_MULTI_PROJ_IMG_POS "v.proj_blk.%d.img_pos"
#define TN_MULTI_PROJ_QUERY "%s.proj_blk.%d.query"
#define TN_MULTI_PROJ_LINEAR "%s.proj_blk.%d.linear.%s"
#define TN_MULTI_PROJ_NORM "%s.proj_blk.%d.norm.%s"
#define TN_MULTI_PROJ_POST_NORM "%s.proj_blk.%d.post_norm.%s"

// mobilenetv5 (gemma3n) definitions
#define TN_MNV5_STEM_CONV "v.conv_stem.conv.weight"
#define TN_MNV5_STEM_BIAS "v.conv_stem.conv.bias"
#define TN_MNV5_STEM_BN "v.conv_stem.bn.weight"

// Stage 0 Block (Edge Residual)
#define TN_MNV5_BLK_S0_EXP_W "v.blk.%d.%d.conv_exp.weight"
#define TN_MNV5_BLK_S0_BN1_W "v.blk.%d.%d.bn1.weight"
#define TN_MNV5_BLK_S0_PWL_W "v.blk.%d.%d.conv_pwl.weight"
#define TN_MNV5_BLK_S0_BN2_W "v.blk.%d.%d.bn2.weight"

// Stage 1+ Block (Universal Inverted Residual)
#define TN_MNV5_BLK_DW_START_W "v.blk.%d.%d.dw_start.conv.weight"
#define TN_MNV5_BLK_DW_START_BN "v.blk.%d.%d.dw_start.bn.weight"
#define TN_MNV5_BLK_DW_MID_W "v.blk.%d.%d.dw_mid.conv.weight"
#define TN_MNV5_BLK_DW_MID_BN "v.blk.%d.%d.dw_mid.bn.weight"
#define TN_MNV5_BLK_PW_EXP_W "v.blk.%d.%d.pw_exp.conv.weight"
#define TN_MNV5_BLK_PW_EXP_BN "v.blk.%d.%d.pw_exp.bn.weight"
#define TN_MNV5_BLK_PW_PROJ_W "v.blk.%d.%d.pw_proj.conv.weight"
#define TN_MNV5_BLK_PW_PROJ_BN "v.blk.%d.%d.pw_proj.bn.weight"
#define TN_MNV5_BLK_LAYER_SCALE "v.blk.%d.%d.layer_scale.gamma"

// Attention Components
#define TN_MNV5_ATTN_Q_W "v.blk.%d.%d.attn.query.proj.weight"
#define TN_MNV5_ATTN_K_W "v.blk.%d.%d.attn.key.proj.weight"
#define TN_MNV5_ATTN_V_W "v.blk.%d.%d.attn.value.proj.weight"
#define TN_MNV5_ATTN_O_W "v.blk.%d.%d.attn.output.proj.weight"
#define TN_MNV5_ATTN_K_DW "v.blk.%d.%d.attn.key.down_conv.weight"
#define TN_MNV5_ATTN_K_NORM "v.blk.%d.%d.attn.key.norm.weight"
#define TN_MNV5_ATTN_V_DW "v.blk.%d.%d.attn.value.down_conv.weight"
#define TN_MNV5_ATTN_V_NORM "v.blk.%d.%d.attn.value.norm.weight"
#define TN_MNV5_ATTN_NORM "v.blk.%d.%d.norm.weight" // Block norm used in attn blocks

// MSFA
#define TN_MNV5_MSFA_FFN_EXP_W "v.msfa.ffn.pw_exp.conv.weight"
#define TN_MNV5_MSFA_FFN_EXP_BN "v.msfa.ffn.pw_exp.bn.weight"
#define TN_MNV5_MSFA_FFN_PROJ_W "v.msfa.ffn.pw_proj.conv.weight"
#define TN_MNV5_MSFA_FFN_PROJ_BN "v.msfa.ffn.pw_proj.bn.weight"
#define TN_MNV5_MSFA_NORM "v.msfa.norm.weight"

// yasa2
#define TN_YASA_PATCH_LN_W "v.patch_ln.weight"
#define TN_YASA_PATCH_LN_B "v.patch_ln.bias"
#define TN_YASA_BACKBONE_LN_W "v.backbone_ln.weight"
#define TN_YASA_BACKBONE_LN_B "v.backbone_ln.bias"
#define TN_YASA_POS_EMBD "v.vision_pos_embed"
#define TN_YASA_STAGE_DOWN_LN "v.stage.%d.down.ln.%s"
#define TN_YASA_STAGE_DOWN_CONV "v.stage.%d.down.conv.%s"
#define TN_YASA_STAGE_BLK "v.stage.%d.blk.%d.%s.%s"

// parakeet
#define TN_MEL_FILTERS "a.mel_filters"
#define TN_WINDOW "a.window"
#define TN_CONV_NORM_MEAN "%s.blk.%d.conv_norm_mean"
#define TN_CONV_NORM_VAR "%s.blk.%d.conv_norm_var"

// align x to upper multiple of n
#define CLIP_ALIGN(x, n) ((((x) + (n) - 1) / (n)) * (n))

// forward declaration
// TODO: improve this later
struct clip_ctx;

enum projector_type {
  cProjectorTypeQwen2vl,
  cProjectorTypeQwen3vl,
  cProjectorTypeStep3vl,
  cProjectorTypeGemma3,
  cProjectorTypeGemma3nv,
  cProjectorTypeGemma3na,
  cProjectorTypePhi4,
  cProjectorTypeIdefics3,
  cProjectorTypePixtral,
  cProjectorTypeQwen25vl,
  cProjectorTypeUltravox,
  cProjectorTypeInternvl,
  cProjectorTypeQwen2a,
  cProjectorTypeQwen3a,
  cProjectorTypeGlma,
  cProjectorTypeQwen25o, // will be replaced by QWEN2A or QWEN25VL depending on clip_ctx
  cProjectorTypeVoxtral,
  cProjectorTypeMeralion,
  cProjectorTypeMusicFlamingo,
  cProjectorTypeLfm2,
  cProjectorTypeKimivl,
  cProjectorTypePaddleocr,
  cProjectorTypeLightonocr,
  cProjectorTypeCogvlm,
  cProjectorTypeJanusPro,
  cProjectorTypeDotsOcr,
  cProjectorTypeLfm2a,
  cProjectorTypeGlm4v,
  cProjectorTypeGlm5v,
  cProjectorTypeYoutuvl,
  cProjectorTypeYasa2,
  cProjectorTypeKimik25,
  cProjectorTypeNemotronV2Vl,
  cProjectorTypeHunyuanvl,
  cProjectorTypeParakeet,
  cProjectorTypeEXAONE4Point5,
  cProjectorTypeMINICPMV4Point6,
  cProjectorTypeGraniteSpeech,
  cProjectorTypeMimovl,
  cProjectorTypeMinimaxM3,
  cProjectorTypeGranite4Vision,
  cProjectorTypeMimoAudio,
  cProjectorTypeUnknown,
};

static std::map<projector_type, std::string> dProjectorTypeNames = {
    {cProjectorTypeQwen2vl, "qwen2vl_merger"},
    {cProjectorTypeQwen25vl, "qwen2.5vl_merger"},
    {cProjectorTypeQwen3vl, "qwen3vl_merger"},
    {cProjectorTypeStep3vl, "step3vl"},
    {cProjectorTypeGemma3, "gemma3"},
    {cProjectorTypeGemma3nv, "gemma3nv"},
    {cProjectorTypeGemma3na, "gemma3na"},
    {cProjectorTypePhi4, "phi4"},
    {cProjectorTypeIdefics3, "idefics3"},
    {cProjectorTypePixtral, "pixtral"},
    {cProjectorTypeUltravox, "ultravox"},
    {cProjectorTypeInternvl, "internvl"},
    {cProjectorTypeQwen2a, "qwen2a"},
    {cProjectorTypeQwen3a, "qwen3a"},
    {cProjectorTypeGlma, "glma"},
    {cProjectorTypeQwen25o, "qwen2.5o"},
    {cProjectorTypeVoxtral, "voxtral"},
    {cProjectorTypeMeralion, "meralion"},
    {cProjectorTypeMusicFlamingo, "musicflamingo"},
    {cProjectorTypeLfm2, "lfm2"},
    {cProjectorTypeKimivl, "kimivl"},
    {cProjectorTypePaddleocr, "paddleocr"},
    {cProjectorTypeLightonocr, "lightonocr"},
    {cProjectorTypeCogvlm, "cogvlm"},
    {cProjectorTypeJanusPro, "janus_pro"},
    {cProjectorTypeDotsOcr, "dots_ocr"},
    {cProjectorTypeLfm2a, "lfm2a"},
    {cProjectorTypeGlm4v, "glm4v"},
    {cProjectorTypeGlm5v, "glm5v"},
    {cProjectorTypeYoutuvl, "youtuvl"},
    {cProjectorTypeYasa2, "yasa2"},
    {cProjectorTypeKimik25, "kimik25"},
    {cProjectorTypeNemotronV2Vl, "nemotron_v2_vl"},
    {cProjectorTypeEXAONE4Point5, "exaone4_5"},
    {cProjectorTypeHunyuanvl, "hunyuanvl"},
    {cProjectorTypeMINICPMV4Point6, "minicpmv4_6"},
    {cProjectorTypeGraniteSpeech, "granite_speech"},
    {cProjectorTypeMimovl, "mimovl"},
    {cProjectorTypeMinimaxM3, "minimax_m3"},
    {cProjectorTypeGranite4Vision, "granite4_vision"},
    {cProjectorTypeMimoAudio, "mimo_audio"},
    {cProjectorTypeParakeet, "parakeet"},
};

static projector_type fClipProjectorTypeFromString(const std::string &pStr) {
  for (const auto &vPair : dProjectorTypeNames) {
    if (vPair.second == pStr) {
      return vPair.first;
    }
  }
  return cProjectorTypeUnknown;
}

// RGB uint8 image
struct clip_image_u8 {
  clip_image_size mGetSize() const { return {vNx, vNy}; }

  void mSetSize(clip_image_size pSize, bool pIsPlaceholder) {
    vNx = pSize.vWidth;
    vNy = pSize.vHeight;
    if (pIsPlaceholder) {
      lBuf.clear();
    } else {
      lBuf.resize((size_t)vNx * (size_t)vNy * 3);
    }
  }

  void mCpyBuf(const std::vector<uint8_t> &pNewBuf) { lBuf = pNewBuf; }

  const std::vector<uint8_t> &mGetRoBuf() const {
    if (mIsPlaceholder()) {
      throw std::runtime_error("this clip_image_u8 is a placeholder");
    }
    return lBuf;
  }

  // note to contributors: NEVER add a get_rw_buf(), it is a DANGEROUS pattern. always use get_pixel / set_pixel for
  // buffer manipulation

  bool mIsPlaceholder() const { return lBuf.empty(); }

  std::array<uint8_t, 3> mGetPixel(int pX, int pY) const {
    if (mIsPlaceholder()) {
      // a placeholder only carries its size: callers must check mIsPlaceholder() before reading pixels
      GGML_ABORT("cannot read a pixel of a placeholder clip_image_u8");
    }
    int vIdx = (pY * vNx + pX) * 3;
    return {lBuf[vIdx], lBuf[vIdx + 1], lBuf[vIdx + 2]};
  }

  void mSetPixel(int pX, int pY, const std::array<uint8_t, 3> &pRgb) {
    if (mIsPlaceholder()) {
      return; // no-op
    }
    int vIdx = (pY * vNx + pX) * 3;
    lBuf[vIdx] = pRgb[0];
    lBuf[vIdx + 1] = pRgb[1];
    lBuf[vIdx + 2] = pRgb[2];
  }

  size_t mNElements() const { return mNPixels() * 3; }

private:
  std::vector<uint8_t> lBuf;
  int vNx = 0;
  int vNy = 0;

  size_t mNPixels() const { return (size_t)vNx * (size_t)vNy; }
};

// For images, buf.size() == nx*ny*3
//     Memory layout: RGBRGBRGB...
// For seq, buf.size() == nx*ny*3*nt
//     Memory layout: RGBRGB...RGBRGB... (nt times)
// For audio, only one channel is used, buf.size() == nx*ny
//     nx will be n_frames and ny will be n_mel
struct clip_image_f32 {
  // whether a learned newline (or EOI) token should be appended after the image (eg Granite4 Vision)
  bool vAddNewline = false;

  clip_image_size mGetSize() const { return {vNx, vNy}; }

  int mNx() const { return vNx; }
  int mNy() const { return vNy; }

  void mSetSize(clip_image_size pSize, bool pIsPlaceholder, bool pIsAudio) {
    vNx = pSize.vWidth;
    vNy = pSize.vHeight;
    if (pIsPlaceholder) {
      lBuf.clear();
    } else {
      if (pIsAudio) {
        lBuf.resize((size_t)vNx * (size_t)vNy);
      } else {
        lBuf.resize((size_t)vNx * (size_t)vNy * 3);
      }
    }
  }

  void mCpyBuf(const std::vector<float> &pNewBuf) { lBuf = pNewBuf; }

  void mFromU8(const clip_image_u8 &pImg) {
    auto vSize = pImg.mGetSize();
    vNx = vSize.vWidth;
    vNy = vSize.vHeight;
    if (pImg.mIsPlaceholder()) {
      lBuf.clear();
      return; // no-op
    }
    lBuf.resize(pImg.mNElements());
    const auto &vU8Buf = pImg.mGetRoBuf();
    for (size_t vI = 0; vI < pImg.mNElements(); ++vI) {
      lBuf[vI] = (float)vU8Buf[vI] / 255.0f;
    }
  }

  size_t mNElements() const { return mNPixels() * 3; }

  void mNormalize(const float pMean[3], const float pStd[3]) {
    if (mIsPlaceholder()) {
      return; // no-op
    }
    for (size_t vI = 0; vI < mNPixels(); ++vI) {
      lBuf[vI * 3 + 0] = (lBuf[vI * 3 + 0] - pMean[0]) / pStd[0];
      lBuf[vI * 3 + 1] = (lBuf[vI * 3 + 1] - pMean[1]) / pStd[1];
      lBuf[vI * 3 + 2] = (lBuf[vI * 3 + 2] - pMean[2]) / pStd[2];
    }
  }

  const std::vector<float> &mGetRoBuf() const {
    if (mIsPlaceholder()) {
      throw std::runtime_error("this clip_image_f32 is a placeholder");
    }
    return lBuf;
  }

  // note to contributors: NEVER add a get_rw_buf(), it is a DANGEROUS pattern

  bool mIsPlaceholder() const { return lBuf.empty(); }

private:
  std::vector<float> lBuf;
  int vNx = 0;
  int vNy = 0;

  size_t mNPixels() const { return (size_t)vNx * (size_t)vNy; }
};

//
// logging
//

static void fClipLogCallbackDefault(enum ggml_log_level pLevel, const char *pText, void *pUserData) {
  (void)pLevel;
  (void)pUserData;
  fputs(pText, stderr);
  fflush(stderr);
}

struct clip_logger_state {
  ggml_log_callback vLogCallback;
  void *vLogCallbackUserData;
};

extern struct clip_logger_state vGLoggerState;

static void fClipLogInternalV(enum ggml_log_level pLevel, const char *pFormat, va_list pArgs) {
  if (pFormat == NULL) {
    return;
  }
  va_list vArgsCopy;
  va_copy(vArgsCopy, pArgs);
  char aBuffer[128];
  int vLen = vsnprintf(aBuffer, 128, pFormat, pArgs);
  if (vLen < 128) {
    vGLoggerState.vLogCallback(pLevel, aBuffer, vGLoggerState.vLogCallbackUserData);
  } else {
    char *vBuffer2 = (char *)calloc(vLen + 1, sizeof(char));
    vsnprintf(vBuffer2, vLen + 1, pFormat, vArgsCopy);
    vBuffer2[vLen] = 0;
    vGLoggerState.vLogCallback(pLevel, vBuffer2, vGLoggerState.vLogCallbackUserData);
    free(vBuffer2);
  }
  va_end(vArgsCopy);
}

static void fClipLogInternal(enum ggml_log_level pLevel, const char *pFormat, ...) {
  va_list vArgs;
  va_start(vArgs, pFormat);
  fClipLogInternalV(pLevel, pFormat, vArgs);
  va_end(vArgs);
}

#define LOG_TRC(...) fClipLogInternal(GGML_LOG_LEVEL_DEBUG, __VA_ARGS__)
#define LOG_DBG(...) fClipLogInternal(GGML_LOG_LEVEL_DEBUG, __VA_ARGS__)
#define LOG_INF(...) fClipLogInternal(GGML_LOG_LEVEL_INFO, __VA_ARGS__)
#define LOG_WRN(...) fClipLogInternal(GGML_LOG_LEVEL_WARN, __VA_ARGS__)
#define LOG_ERR(...) fClipLogInternal(GGML_LOG_LEVEL_ERROR, __VA_ARGS__)
#define LOG_CNT(...) fClipLogInternal(GGML_LOG_LEVEL_CONT, __VA_ARGS__)

//
// cpp wrappers
//

struct clip_image_f32_batch {
  std::vector<clip_image_f32> lEntries;
  bool vIsAudio = false;

  clip_image_f32_batch mClone() const {
    clip_image_f32_batch vNewBatch{
        /* entries  */ {},
        /* is_audio */ vIsAudio,
    };
    vNewBatch.lEntries.reserve(lEntries.size());
    for (const auto &vEntry : lEntries) {
      vNewBatch.lEntries.emplace_back(vEntry); // copy
    }
    return vNewBatch;
  }
};

//
// common utils
//

#ifdef _WIN32
static std::ifstream fOpenIfstreamBinary(const std::string &pFname) {
  int vWlen = MultiByteToWideChar(CP_UTF8, 0, pFname.c_str(), -1, NULL, 0);
  if (!vWlen) {
    throw std::runtime_error("failed to convert filename to UTF-16: " + pFname);
  }
  std::vector<wchar_t> lWfname(vWlen);
  (void)MultiByteToWideChar(CP_UTF8, 0, pFname.c_str(), -1, lWfname.data(), vWlen);
  return std::ifstream(lWfname.data(), std::ios::binary);
}
#else
static std::ifstream fOpenIfstreamBinary(const std::string &pFname) { return std::ifstream(pFname, std::ios::binary); }
#endif

static std::string fStringFormat(const char *pFmt, ...) {
  va_list vAp;
  va_list vAp2;
  va_start(vAp, pFmt);
  va_copy(vAp2, vAp);
  int vSize = vsnprintf(NULL, 0, pFmt, vAp);
  GGML_ASSERT(vSize >= 0 && vSize < INT_MAX); // NOLINT
  std::vector<char> lBuf(vSize + 1);
  int vSize2 = vsnprintf(lBuf.data(), vSize + 1, pFmt, vAp2);
  GGML_ASSERT(vSize2 == vSize);
  va_end(vAp2);
  va_end(vAp);
  return std::string(lBuf.data(), lBuf.size());
}

static void fStringReplaceAll(std::string &pS, const std::string &pSearch, const std::string &pReplace) {
  if (pSearch.empty()) {
    return;
  }
  std::string vBuilder;
  vBuilder.reserve(pS.length());
  size_t vPos = 0;
  size_t vLastPos = 0;
  while ((vPos = pS.find(pSearch, vLastPos)) != std::string::npos) {
    vBuilder.append(pS, vLastPos, vPos - vLastPos);
    vBuilder.append(pReplace);
    vLastPos = vPos + pSearch.length();
  }
  vBuilder.append(pS, vLastPos, std::string::npos);
  pS = std::move(vBuilder);
}

// split string by a `std::string delim` instead of `char delim`
static std::vector<std::string> fStringSplitStr(std::string pS, const std::string &pDelimiter) {
  std::vector<std::string> lTokens;
  size_t vPos = 0;
  std::string vToken;
  while ((vPos = pS.find(pDelimiter)) != std::string::npos) {
    vToken = pS.substr(0, vPos);
    lTokens.push_back(vToken);
    pS.erase(0, vPos + pDelimiter.length());
  }
  lTokens.push_back(pS);
  return lTokens;
}

// remove when moving to c++20
inline bool fStringStartsWith(std::string_view pStr, std::string_view pPrefix) {
  return pStr.size() >= pPrefix.size() && pStr.compare(0, pPrefix.size(), pPrefix) == 0;
}

// remove when moving to c++20
inline bool fStringEndsWith(std::string_view pStr, std::string_view pSuffix) {
  return pStr.size() >= pSuffix.size() && pStr.compare(pStr.size() - pSuffix.size(), pSuffix.size(), pSuffix) == 0;
}

//
// gguf utils
//

static std::string fGgufDataToStr(enum gguf_type pType, const void *pData, int pI) {
  switch (pType) {
  case GGUF_TYPE_UINT8:
    return std::to_string(((const uint8_t *)pData)[pI]);
  case GGUF_TYPE_INT8:
    return std::to_string(((const int8_t *)pData)[pI]);
  case GGUF_TYPE_UINT16:
    return std::to_string(((const uint16_t *)pData)[pI]);
  case GGUF_TYPE_INT16:
    return std::to_string(((const int16_t *)pData)[pI]);
  case GGUF_TYPE_UINT32:
    return std::to_string(((const uint32_t *)pData)[pI]);
  case GGUF_TYPE_INT32:
    return std::to_string(((const int32_t *)pData)[pI]);
  case GGUF_TYPE_UINT64:
    return std::to_string(((const uint64_t *)pData)[pI]);
  case GGUF_TYPE_INT64:
    return std::to_string(((const int64_t *)pData)[pI]);
  case GGUF_TYPE_FLOAT32:
    return std::to_string(((const float *)pData)[pI]);
  case GGUF_TYPE_FLOAT64:
    return std::to_string(((const double *)pData)[pI]);
  case GGUF_TYPE_BOOL:
    return ((const int8_t *)pData)[pI] != 0 ? "true" : "false";
  default:
    return fStringFormat("unknown type %d", pType);
  }
}

static std::string fGgufKvToStr(const struct gguf_context *pCtxGguf, int pI) {
  const enum gguf_type cType = gguf_get_kv_type(pCtxGguf, pI);

  switch (cType) {
  case GGUF_TYPE_STRING:
    return gguf_get_val_str(pCtxGguf, pI);
  case GGUF_TYPE_ARRAY: {
    const enum gguf_type cArrType = gguf_get_arr_type(pCtxGguf, pI);
    int vArrN = gguf_get_arr_n(pCtxGguf, pI);
    const void *vData = cArrType == GGUF_TYPE_STRING ? nullptr : gguf_get_arr_data(pCtxGguf, pI);
    std::stringstream vSs;
    vSs << "[";
    for (int vJ = 0; vJ < vArrN; vJ++) {
      if (cArrType == GGUF_TYPE_STRING) {
        std::string vVal = gguf_get_arr_str(pCtxGguf, pI, vJ);
        // escape quotes
        fStringReplaceAll(vVal, "\\", "\\\\");
        fStringReplaceAll(vVal, "\"", "\\\"");
        vSs << '"' << vVal << '"';
      } else if (cArrType == GGUF_TYPE_ARRAY) {
        vSs << "???";
      } else {
        vSs << fGgufDataToStr(cArrType, vData, vJ);
      }
      if (vJ < vArrN - 1) {
        vSs << ", ";
      }
    }
    vSs << "]";
    return vSs.str();
  }
  default:
    return fGgufDataToStr(cType, gguf_get_val_data(pCtxGguf, pI), 0);
  }
}

//
// debugging
//

static void fPrintTensorShape(ggml_tensor *pT) {
  printf("%s.shape = [", pT->name);
  for (int vI = 0; vI < ggml_n_dims(pT); ++vI) {
    printf("%" PRId64, pT->ne[vI]);
    if (vI < ggml_n_dims(pT) - 1) {
      printf(", ");
    }
  }
  printf("]\n");
}

static void fPrintTensorData(ggml_tensor *pT, uint8_t *pData, int64_t pN) {
  ggml_type vType = pT->type;
  int64_t *vNe = pT->ne;
  size_t *vNb = pT->nb;
  for (int64_t vI3 = 0; vI3 < vNe[3]; vI3++) {
    printf("%s.data: [\n", pT->name);
    for (int64_t vI2 = 0; vI2 < vNe[2]; vI2++) {
      if (vI2 == pN && vNe[2] > 2 * pN) {
        printf("     ..., \n");
        vI2 = vNe[2] - pN;
      }
      printf("     [\n");
      for (int64_t vI1 = 0; vI1 < vNe[1]; vI1++) {
        if (vI1 == pN && vNe[1] > 2 * pN) {
          printf("      ..., \n");
          vI1 = vNe[1] - pN;
        }
        printf("      [");
        for (int64_t vI0 = 0; vI0 < vNe[0]; vI0++) {
          if (vI0 == pN && vNe[0] > 2 * pN) {
            printf("..., ");
            vI0 = vNe[0] - pN;
          }
          size_t vI = vI3 * vNb[3] + vI2 * vNb[2] + vI1 * vNb[1] + vI0 * vNb[0];
          float vV;
          if (vType == GGML_TYPE_F16) {
            vV = ggml_fp16_to_fp32(*(ggml_fp16_t *)&pData[vI]);
          } else if (vType == GGML_TYPE_F32) {
            vV = *(float *)&pData[vI];
          } else if (vType == GGML_TYPE_I32) {
            vV = (float)*(int32_t *)&pData[vI];
          } else if (vType == GGML_TYPE_I16) {
            vV = (float)*(int16_t *)&pData[vI];
          } else if (vType == GGML_TYPE_I8) {
            vV = (float)*(int8_t *)&pData[vI];
          } else {
            GGML_ABORT("fatal error");
          }
          printf("%8.4f", vV);
          if (vI0 < vNe[0] - 1)
            printf(", ");
        }
        printf("],\n");
      }
      printf("     ],\n");
    }
    printf("    ]\n");
  }
}

//
// API used internally with mtmd
//

projector_type fClipGetProjectorType(const struct clip_ctx *pCtx);
void fClipSetDebugOutputEmbeddings(struct clip_ctx *pCtx, bool pDebug);
