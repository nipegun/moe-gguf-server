import type {
  ContentPartType,
  FileTypeAudio,
  ServerModelStatus,
  ServerModelsSseEventType,
  ServerRole
} from '$lib/enums';
import type { ChatMessagePromptProgress, ChatRole } from './chat';

export type AudioInputFormat = FileTypeAudio.WAV | FileTypeAudio.MP3;

export interface ApiChatCompletionToolFunction {
  name: string;
  description?: string;
  parameters: Record<string, unknown>;
}

export interface ApiChatCompletionTool {
  type: 'function';
  function: ApiChatCompletionToolFunction;
}

export interface ApiChatMessageContentPart {
  type: ContentPartType;
  text?: string;
  image_url?: {
    url: string;
  };
  input_audio?: {
    data: string;
    format: AudioInputFormat;
  };
  input_video?: {
    data: string;
    format: 'mp4' | 'ogg' | 'auto';
  };
}

export interface ApiContextSizeError {
  code: number;
  message: string;
  type: 'exceed_context_size_error';
  n_prompt_tokens: number;
  n_ctx: number;
}

export interface ApiErrorResponse {
  error:
    | ApiContextSizeError
    | {
        code: number;
        message: string;
        type?: string;
      };
}

export interface ApiChatMessageData {
  role: ChatRole;
  content: string | ApiChatMessageContentPart[];
  reasoning_content?: string;
  tool_calls?: ApiChatCompletionToolCall[];
  tool_call_id?: string;
  timestamp?: number;
}

/**
 * Model status object from /models endpoint
 */
export interface ApiModelStatus {
  /** Status value: loaded, unloaded, loading, sleeping, failed */
  value: ServerModelStatus;
  /** Command line arguments used when loading (only for loaded models) */
  args?: string[];
  /** True when the last load failed; exit_code holds the exit code of the child process */
  failed?: boolean;
  exit_code?: number;
}

/**
 * Model entry from /models endpoint (ROUTER mode)
 * Based on actual API response structure
 */
export interface ApiModelDataEntry {
  /** Model identifier (e.g., "ggml-org/Qwen2.5-Omni-7B-GGUF:latest") */
  id: string;
  /** Model name (optional, usually same as id - not always returned by API) */
  name?: string;
  /** Object type, always "model" */
  object: string;
  /** Owner, usually "mgs" */
  owned_by: string;
  /** Creation timestamp */
  created: number;
  /** Whether model files are in HuggingFace cache */
  in_cache: boolean;
  /** GGUF file of the model, the first part of a split model (router mode) */
  path?: string;
  /** Size on disk in bytes, every part of a split model included (router mode) */
  file_size?: number;
  /** Multimodal projector GGUF, when the model has one (router mode) */
  mmproj?: string;
  /** Where the router found the model: models_dir, cache or preset */
  source?: string;
  /** Current status of the model */
  status: ApiModelStatus;
  /** Alternative names that resolve to this model */
  aliases?: string[];
  /** Informational tags for this model */
  tags?: string[];
  /** Model metadata (vocabulary, context, embedding size, parameters, file size, type) */
  meta?: Record<string, unknown> | null;
  /** Configuration the model was loaded with (only while it runs) */
  load?: ApiModelLoadInfo;
}

/**
 * Configuration a model was loaded with, after --fit and the automatic choices of mgs,
 * as published by the "load" object of /v1/models.
 */
export interface ApiModelLoadInfo {
  architecture?: string;
  description?: string;
  n_layer?: number;
  n_layer_mtp?: number;
  n_ctx?: number;
  n_ctx_slot?: number;
  n_parallel?: number;
  n_batch?: number;
  n_ubatch?: number;
  n_threads?: number;
  n_threads_batch?: number;
  flash_attn?: string;
  /** Whether the context uses flash attention, also when flash_attn is "auto" */
  flash_attn_enabled?: boolean;
  cache_type_k?: string;
  cache_type_v?: string;
  fit?: boolean;
  mtp?: boolean;
  speculative?: string[];
  draft_n_max?: number;
  /** MiB of the MoE expert cache: -1 automatic, 0 disabled */
  moe_cache_mib?: number;
  devices?: { name: string; description: string }[];
  mmproj?: string;
}

/**
 * Folder of local models in router mode (GET and POST /models/folder).
 */
export interface ApiModelsFolderInfo {
  /** Folder scanned for GGUF models, empty when there is none */
  folder: string;
  /** cli: set with --models-dir; settings: saved in ~/.mgs/config/ui-settings.json; none */
  source: 'cli' | 'settings' | 'none';
  exists: boolean;
  /** Models of the list that come from the folder */
  model_count: number;
  /** File where the server keeps the folder (~/.mgs/config/ui-settings.json) */
  settings_file: string;
  /** Download cache, whose models are listed as well */
  cache_dir: string;
}

/**
 * One folder of the server's computer, for the folder browser of Settings > Models
 * (GET /models/folder/browse).
 */
export interface ApiModelsFolderBrowse {
  /** Absolute path of the folder */
  path: string;
  /** Folder above it, empty at the root */
  parent: string;
  /** Home folder of the user that runs mgs, empty when it is unknown */
  home: string;
  /** Subfolders, sorted by name */
  folders: { name: string; path: string; hidden: boolean }[];
  /** .gguf files right inside the folder */
  gguf_files: number;
}

/**
 * Load stage reported by the /models/sse feed, in load order.
 */
export type ApiModelLoadStage = 'text_model' | 'spec_model' | 'mmproj_model';

/**
 * Load progress snapshot: the full ordered stage plan, the active stage,
 * and its fractional value (0.0 -> 1.0).
 */
export interface ApiModelsSseProgress {
  stages: ApiModelLoadStage[];
  current: ApiModelLoadStage;
  value: number;
}

/**
 * Status payload carried by a /models/sse envelope.
 * exit_code appears on unload.
 */
export interface ApiModelsSseData {
  status: ServerModelStatus;
  progress?: ApiModelsSseProgress;
  exit_code?: number;
}

/**
 * Event kind multiplexed on the /models/sse feed.
 * Only the status_* events carry a status payload, models_reload signals a
 * full list refresh, model_remove drops a row, download_* drive download UI.
 */
/**
 * One /models/sse record. event discriminates the kind, model names the
 * target instance, data carries the status payload when present.
 */
export interface ApiModelsSseEvent {
  model: string;
  event: ServerModelsSseEventType;
  data: ApiModelsSseData;
}

export interface ApiModelDetails {
  name: string;
  model: string;
  modified_at?: string;
  size?: string | number;
  digest?: string;
  type?: string;
  description?: string;
  tags?: string[];
  capabilities?: string[];
  parameters?: string;
  details?: {
    parent_model?: string;
    format?: string;
    family?: string;
    families?: string[];
    parameter_size?: string;
    quantization_level?: string;
  };
}

export interface ApiModelListResponse {
  object: string;
  data: ApiModelDataEntry[];
  models?: ApiModelDetails[];
}

export interface ApiMgsServerProps {
  default_generation_settings: {
    id: number;
    id_task: number;
    n_ctx: number;
    speculative: boolean;
    is_processing: boolean;
    params: {
      n_predict: number;
      seed: number;
      temperature: number;
      dynatemp_range: number;
      dynatemp_exponent: number;
      top_k: number;
      top_p: number;
      min_p: number;
      top_n_sigma: number;
      xtc_probability: number;
      xtc_threshold: number;
      typ_p: number;
      repeat_last_n: number;
      repeat_penalty: number;
      presence_penalty: number;
      frequency_penalty: number;
      dry_multiplier: number;
      dry_base: number;
      dry_allowed_length: number;
      dry_penalty_last_n: number;
      dry_sequence_breakers: string[];
      mirostat: number;
      mirostat_tau: number;
      mirostat_eta: number;
      stop: string[];
      max_tokens: number;
      n_keep: number;
      n_discard: number;
      ignore_eos: boolean;
      stream: boolean;
      logit_bias: Array<[number, number]>;
      n_probs: number;
      min_keep: number;
      grammar: string;
      grammar_lazy: boolean;
      grammar_triggers: string[];
      preserved_tokens: number[];
      chat_format: string;
      reasoning_format: string;
      generation_prompt: string;
      samplers: string[];
      backend_sampling: boolean;
      'speculative.n_max': number;
      'speculative.n_min': number;
      'speculative.p_min': number;
      timings_per_token: boolean;
      post_sampling_probs: boolean;
      lora: Array<{ name: string; scale: number }>;
    };
    prompt: string;
    next_token: {
      has_next_token: boolean;
      has_new_line: boolean;
      n_remain: number;
      n_decoded: number;
      stopping_word: string;
    };
  };
  total_slots: number;
  model_path: string;
  role: ServerRole;
  modalities: {
    vision: boolean;
    audio: boolean;
    video: boolean;
  };
  chat_template: string;
  bos_token: string;
  eos_token: string;
  build_info: string;
  ui_settings?: Record<string, string | number | boolean>;
  cors_proxy_enabled?: boolean;
  /** true while the local shell MCP server answers at <api prefix>/mcp/shell */
  shell_mcp_server?: boolean;
  /** true when --local-shell-mcp-server keeps it on, so the web UI cannot turn it off */
  shell_mcp_server_forced?: boolean;
  /** true when this browser may turn it on or off (an API key is set, or it runs on the server's computer) */
  shell_mcp_server_toggle?: boolean;
}

export interface ApiChatCompletionRequest {
  messages: Array<{
    role: ChatRole;
    content: string | ApiChatMessageContentPart[];
    reasoning_content?: string;
    tool_calls?: ApiChatCompletionToolCall[];
    tool_call_id?: string;
  }>;
  stream?: boolean;
  model?: string;
  /** Marks the request as sent by the interface, which receives queue status events */
  mgs_client?: 'webui';
  return_progress?: boolean;
  sse_ping_interval?: number;
  tools?: ApiChatCompletionTool[];
  // Reasoning parameters
  reasoning_format?: string;
  // Generation parameters
  temperature?: number;
  max_tokens?: number;
  // Sampling parameters
  dynatemp_range?: number;
  dynatemp_exponent?: number;
  top_k?: number;
  top_p?: number;
  min_p?: number;
  xtc_probability?: number;
  xtc_threshold?: number;
  typ_p?: number;
  // Penalty parameters
  repeat_last_n?: number;
  repeat_penalty?: number;
  presence_penalty?: number;
  frequency_penalty?: number;
  dry_multiplier?: number;
  dry_base?: number;
  dry_allowed_length?: number;
  dry_penalty_last_n?: number;
  // Sampler configuration
  samplers?: string[];
  backend_sampling?: boolean;
  // Custom parameters (JSON string)
  custom?: Record<string, unknown>;
  timings_per_token?: boolean;
  // Continuation control (vLLM compat)
  add_generation_prompt?: boolean;
  continue_final_message?: boolean;
}

export interface ApiChatCompletionToolCallFunctionDelta {
  name?: string;
  arguments?: string;
}

export interface ApiChatCompletionToolCallDelta {
  index?: number;
  id?: string;
  type?: string;
  function?: ApiChatCompletionToolCallFunctionDelta;
}

export interface ApiChatCompletionToolCall extends ApiChatCompletionToolCallDelta {
  function?: ApiChatCompletionToolCallFunctionDelta & { arguments?: string };
}

/**
 * Queue status that the server streams ({"mgs_queue": {...}}) while a request of the interface waits
 * for a free slot; every count is 0 when the request starts.
 */
export interface ApiQueueStatus {
  /** Requests resolved before this one, running or waiting: the sum of the four counts below */
  ahead: number;
  /** Of them, sent directly to the API and running now */
  running_api: number;
  /** Of them, sent by the interface and running now */
  running_webui: number;
  /** Of them, sent directly to the API and waiting ahead in the queue */
  waiting_api: number;
  /** Of them, sent by the interface and waiting ahead in the queue */
  waiting_webui: number;
}

export interface ApiChatCompletionStreamChunk {
  /** Queue status event: the only field of the chunk */
  mgs_queue?: ApiQueueStatus;
  id?: string;
  object?: string;
  model?: string;
  choices: Array<{
    model?: string;
    metadata?: { model?: string };
    delta: {
      content?: string;
      reasoning_content?: string;
      model?: string;
      tool_calls?: ApiChatCompletionToolCallDelta[];
    };
    finish_reason?: string | null;
  }>;
  timings?: {
    prompt_n?: number;
    prompt_ms?: number;
    predicted_n?: number;
    predicted_ms?: number;
    cache_n?: number;
  };
  prompt_progress?: ChatMessagePromptProgress;
}

export interface ApiChatCompletionResponse {
  model?: string;
  choices: Array<{
    model?: string;
    metadata?: { model?: string };
    message: {
      content: string;
      reasoning_content?: string;
      model?: string;
      tool_calls?: ApiChatCompletionToolCall[];
    };
    finish_reason?: string | null;
  }>;
}

export interface ApiSlotData {
  id: number;
  id_task: number;
  n_ctx: number;
  speculative: boolean;
  is_processing: boolean;
  params: {
    n_predict: number;
    seed: number;
    temperature: number;
    dynatemp_range: number;
    dynatemp_exponent: number;
    top_k: number;
    top_p: number;
    min_p: number;
    top_n_sigma: number;
    xtc_probability: number;
    xtc_threshold: number;
    typical_p: number;
    repeat_last_n: number;
    repeat_penalty: number;
    presence_penalty: number;
    frequency_penalty: number;
    dry_multiplier: number;
    dry_base: number;
    dry_allowed_length: number;
    dry_penalty_last_n: number;
    mirostat: number;
    mirostat_tau: number;
    mirostat_eta: number;
    max_tokens: number;
    n_keep: number;
    n_discard: number;
    ignore_eos: boolean;
    stream: boolean;
    n_probs: number;
    min_keep: number;
    chat_format: string;
    reasoning_format: string;
    generation_prompt: string;
    samplers: string[];
    backend_sampling: boolean;
    'speculative.n_max': number;
    'speculative.n_min': number;
    'speculative.p_min': number;
    timings_per_token: boolean;
    post_sampling_probs: boolean;
    lora: Array<{ name: string; scale: number }>;
  };
  next_token: {
    has_next_token: boolean;
    has_new_line: boolean;
    n_remain: number;
    n_decoded: number;
  };
}

export interface ApiProcessingState {
  status: 'initializing' | 'generating' | 'preparing' | 'idle';
  tokensDecoded: number;
  tokensRemaining: number;
  contextUsed: number;
  contextTotal: number | null;
  outputTokensUsed: number; // Total output tokens (thinking + regular content)
  outputTokensMax: number; // Max output tokens allowed
  temperature: number;
  topP: number;
  speculative: boolean;
  hasNextToken: boolean;
  tokensPerSecond?: number;
  // Progress information from prompt_progress
  progressPercent?: number;
  promptProgress?: ChatMessagePromptProgress;
  promptTokens?: number;
  promptMs?: number;
  cacheTokens?: number;
}

/**
 * Request to load a model
 */
export interface ApiRouterModelsLoadRequest {
  model: string;
}

/**
 * Response from loading a model
 */
export interface ApiRouterModelsLoadResponse {
  success: boolean;
  error?: string;
}

/**
 * Request to check model status
 */
export interface ApiRouterModelsStatusRequest {
  model: string;
}

/**
 * Response with model status
 */
export interface ApiRouterModelsStatusResponse {
  model: string;
  status: ModelStatus;
  port?: number;
  error?: string;
}

/**
 * Response with list of all models from /models endpoint
 * Note: This is the same as ApiModelListResponse - the endpoint returns the same structure
 * regardless of server mode (MODEL or ROUTER)
 */
export interface ApiRouterModelsListResponse {
  object: string;
  data: ApiModelDataEntry[];
}

/**
 * Request to unload a model
 */
export interface ApiRouterModelsUnloadRequest {
  model: string;
}

/**
 * Response from unloading a model
 */
export interface ApiRouterModelsUnloadResponse {
  success: boolean;
  error?: string;
}

/**
 * Entry returned by POST /v1/streams/lookup. The client passes the conv ids it owns in the body
 * and the server returns one entry per matching live or recently completed background streaming
 * session, keyed by conversation_id. The WebUI uses this at mount and on visibilitychange to
 * populate sidebar spinners and to reattach to an ongoing inference for the active conversation.
 * The server never lists ids the client did not ask about, so foreign random UUIDs stay private.
 */
export interface ApiStreamSession {
  conversation_id: string;
  is_done: boolean;
  total_bytes: number;
  started_at: number;
  completed_at: number;
}
