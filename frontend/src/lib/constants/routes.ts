export const cNewChatParam = 'new_chat';

/** Settings section slugs — used for routes and navigation. */
export const cSettingsSectionSlugs = {
  GENERAL: 'general',
  MODELS: 'models',
  DISPLAY: 'display',
  SAMPLING: 'sampling',
  PENALTIES: 'penalties',
  AGENTIC: 'agentic',
  DEVELOPER: 'developer',
  TOOLS: 'tools',
  MCP_SERVERS: 'mcp-servers',
  IMPORT_EXPORT: 'import-export'
} as const;

export const cRoutes = {
  /** Root — start of the app. */
  START: '#/',
  /** New chat — root with new chat query param. */
  NEW_CHAT: `?${cNewChatParam}=true#/`,
  /** Chat base — for dynamic chat URLs use RouterService. */
  CHAT: '#/chat',
  /** Monitor: CPU, GPU and memory usage. */
  MONITOR: '#/monitor',
  /** Model: information and load configuration of the loaded models. */
  MODEL: '#/model',
  /** Settings base — for dynamic settings URLs use RouterService. */
  SETTINGS: '#/settings'
} as const;
