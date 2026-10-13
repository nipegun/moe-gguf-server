export const cNewChatParam = 'new_chat';

/** Settings section slugs — used for routes and navigation. */
export const cSettingsSectionSlugs = {
  GENERAL: 'general',
  DISPLAY: 'display',
  SAMPLING: 'sampling',
  PENALTIES: 'penalties',
  AGENTIC: 'agentic',
  DEVELOPER: 'developer',
  TOOLS: 'tools',
  IMPORT_EXPORT: 'import-export'
} as const;

export const cRoutes = {
  /** Root — start of the app. */
  START: '#/',
  /** New chat — root with new chat query param. */
  NEW_CHAT: `?${cNewChatParam}=true#/`,
  /** Chat base — for dynamic chat URLs use RouterService. */
  CHAT: '#/chat',
  /** MCP servers. */
  MCP_SERVERS: '#/mcp-servers',
  /** Monitor: CPU, GPU and memory usage. */
  MONITOR: '#/monitor',
  /** Settings base — for dynamic settings URLs use RouterService. */
  SETTINGS: '#/settings',
  /** Search — mobile-only full-page conversation search. */
  SEARCH: '#/search'
} as const;
