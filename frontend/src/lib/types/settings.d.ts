import type { dSettingConfigDefault, cSettingsSectionTitles } from '$lib/constants';
import type { ChatMessagePromptProgress, ChatMessageTimings } from './chat';
import type { OpenAIToolDefinition } from './mcp';
import type { ApiQueueStatus } from './api';
import type { DatabaseMessageExtra } from './database';
import type {
  ParameterSource,
  SyncableParameterType,
  SettingsFieldType,
  StreamConnectionState,
  ReasoningEffort
} from '$lib/enums';
import type { Icon } from '@lucide/svelte';
import type { Component } from 'svelte';

export type SettingsConfigValue = string | number | boolean | undefined;

/** Section title type derived from registry section titles. */
export type SettingsSectionTitle =
  (typeof cSettingsSectionTitles)[keyof typeof cSettingsSectionTitles];

/** Per-setting metadata — one entry per setting. */
export interface SettingsEntry {
  key: string;
  label: string;
  help: string;
  defaultValue: SettingsConfigValue;
  type: SettingsFieldType;
  section?: string;
  options?: Array<{ value: string; label: string; icon: Component }>;
  /** Options rendered for RADIO fields. Each entry maps a `value` (the radio's selected value) to the underlying config `key` whose boolean state mirrors it. */
  radioOptions?: Array<{ value: string; label: string; key: string; isExperimental?: boolean }>;
  isExperimental?: boolean;
  isPositiveInteger?: boolean;
  dependsOn?: string;
  sync?: {
    serverKey: string;
    paramType: SyncableParameterType;
  };
}

/** A settings section with its icon, slug, title, and ordered settings. */
export interface SettingsSectionEntry {
  title: SettingsSectionTitle;
  slug: string;
  icon: Component;
  settings: SettingsEntry[];
}

export interface SettingsFieldConfig {
  key: string;
  label: string;
  type: SettingsFieldType;
  isExperimental?: boolean;
  isPositiveInteger?: boolean;
  dependsOn?: string;
  help?: string;
  options?: Array<{ value: string; label: string; icon?: typeof Icon }>;
  /** Options rendered for RADIO fields. Each entry maps a `value` (the radio's selected value) to the underlying config `key` whose boolean state mirrors it. */
  radioOptions?: Array<{ value: string; label: string; key: string; isExperimental?: boolean }>;
}

/** A settings section: its fields, icon, URL slug and title. */
export interface SettingsSection {
  fields?: SettingsFieldConfig[];
  icon: Component;
  slug: string;
  title: SettingsSectionTitle;
}

export interface SettingsChatServiceOptions {
  stream?: boolean;
  // Model (required in ROUTER mode, optional in MODEL mode)
  model?: string;
  // System message to inject
  systemMessage?: string;
  // Disable reasoning parsing (use 'none' instead of 'auto')
  disableReasoningParsing?: boolean;
  // Strip reasoning content from context before sending
  excludeReasoningFromContext?: boolean;
  // Enable model thinking/reasoning via chat_template_kwargs
  enableThinking?: boolean;
  // Reasoning effort level (low/medium/high/max) for thinking models
  reasoningEffort?: ReasoningEffort;
  tools?: OpenAIToolDefinition[];
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
  samplers?: string | string[];
  backend_sampling?: boolean;
  // Custom JSON parameters
  customJson?: string;
  timings_per_token?: boolean;
  // Continuation control (vLLM compat), opt in to the explicit continue final message flag
  continueFinalMessage?: boolean;
  // Callbacks
  onChunk?: (chunk: string) => void;
  onReasoningChunk?: (chunk: string) => void;
  onToolCallChunk?: (chunk: string) => void;
  onAttachments?: (extras: DatabaseMessageExtra[]) => void;
  onModel?: (model: string) => void;
  onCompletionId?: (id: string) => void;
  onTimings?: (timings?: ChatMessageTimings, promptProgress?: ChatMessagePromptProgress) => void;
  /** Queue status while the request waits for a free slot; null when it starts or ends */
  onQueue?: (status: ApiQueueStatus | null) => void;
  onComplete?: (
    response: string,
    reasoningContent?: string,
    timings?: ChatMessageTimings,
    toolCalls?: string
  ) => void;
  onError?: (error: Error) => void;
  onConnectionState?: (state: StreamConnectionState) => void;
}

export type SettingsConfigType = typeof dSettingConfigDefault & {
  [key: string]: SettingsConfigValue;
};

/**
 * Parameter synchronization types for server defaults and user overrides
 * Note: ParameterSource and SyncableParameterType enums are imported from '$lib/enums'
 */
export type ParameterValue = string | number | boolean;
export type ParameterRecord = Record<string, ParameterValue>;

export interface ParameterInfo {
  value: string | number | boolean;
  source: ParameterSource;
  serverDefault?: string | number | boolean;
  userOverride?: string | number | boolean;
}

export interface SyncableParameter {
  key: string;
  serverKey: string;
  type: SyncableParameterType;
  canSync: boolean;
}

/**
 * Settings document of the Web UI, kept by the server in ~/.mgs/config/ui-settings.json of the user
 * that runs mgs and shared by every browser. Every section is optional: a missing one keeps its default.
 * The API key that authenticates a browser is never part of it.
 */
export interface UiSettingsDocument {
  /** Settings config without the API key (includes theme as a config key) */
  config?: Record<string, SettingsConfigValue>;
  /** Keys that differ from server defaults */
  userOverrides?: string[];
  /** Interface language, such as "es-ES" */
  language?: string;
  favoriteModels?: string[];
  disabledToolKeys?: string[];
  alwaysAllowedTools?: string[];
  reasoningEffortDefault?: string;
  dismissedRecommendedMcpServers?: boolean;
}

/**
 * Shape of the settings JSON export file.
 * Versioned to allow future schema evolution.
 */
export interface SettingsExportType {
  /** Export format version — bumped on breaking changes */
  version: number;
  /** Unix timestamp of export */
  timestamp: number;
  /** Full settings config (includes theme as a config key) */
  config: SettingsConfigType;
  /** Keys that differ from server defaults (derived, but persisted for fidelity) */
  userOverrides: string[];
}
