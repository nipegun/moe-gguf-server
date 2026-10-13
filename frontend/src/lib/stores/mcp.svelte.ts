import { fTranslate } from '../i18n';
/**
 * mcpStore - Reactive State Store for MCP Operations
 *
 * Implements the "Host" role in MCP architecture, coordinating multiple server
 * connections and providing a unified interface for tool operations.
 *
 * **Architecture & Relationships:**
 * - **MCPService**: Stateless protocol layer (transport, connect, callTool)
 * - **mcpStore** (this): Reactive state + business logic
 *
 * **Key Responsibilities:**
 * - Lifecycle management (initialize, shutdown)
 * - Multi-server coordination
 * - Tool name conflict detection and resolution
 * - Automatic tool-to-server routing
 * - Health checks
 *
 * MCP connection state and raw `Tool[]` per server are owned here; the
 * OpenAI-compatible wire format for those tools is built in `toolsStore`
 * (see {@link toolsStore.mcpEntries} / {@link toolsStore.getEnabledToolsForLLM}).
 *
 * @see MCPService in services/mcp.service.ts for protocol operations
 */

import { browser } from '$app/environment';
import { cSettingsKeys } from '$lib/constants';
import { MCPService } from '$lib/services/mcp.service';
import { fConfig, cSettingsStore } from '$lib/stores/settings.svelte';
import { cMcpResourceStore } from '$lib/stores/mcp-resources.svelte';
import { cServerStore } from '$lib/stores/server.svelte';
import { mode } from 'mode-watcher';
import { SvelteMap } from 'svelte/reactivity';
import {
  parseMcpServerSettings,
  detectMcpTransportFromUrl,
  uuid,
  extractRootDomain
} from '$lib/utils';
import {
  MCPConnectionPhase,
  MCPLogLevel,
  HealthCheckStatus,
  MCPRefType,
  ColorMode,
  UrlProtocol
} from '$lib/enums';
import {
  cDefaultCacheTtlMs,
  cDefaultMcpConfig,
  cExpectedThemedIconPairCount,
  sMcpAllowedIconMimeTypes,
  cMcpServerIdPrefix,
  cMcpReconnectBackoffMultiplier,
  cMcpReconnectInitialDelay,
  cMcpReconnectMaxDelay,
  cMcpReconnectAttemptTimeoutMs,
  cShellMcpCallIdMeta,
  cShellMcpEndpoint,
  cShellMcpServerId,
  cShellMcpServerName
} from '$lib/constants';
import type {
  MCPToolCall,
  ServerStatus,
  ToolExecutionResult,
  MCPClientConfig,
  MCPConnection,
  HealthCheckParams,
  ServerCapabilities,
  ClientCapabilities,
  MCPCapabilitiesInfo,
  MCPConnectionLog,
  MCPPromptInfo,
  GetPromptResult,
  Tool,
  HealthCheckState,
  MCPServerSettingsEntry,
  MCPServerDisplayInfo,
  MCPServerConfig,
  MCPResourceIcon,
  MCPResourceAttachment,
  MCPResourceContent,
  LocalShellActiveCall
} from '$lib/types';
import type { ListChangedHandlers } from '@modelcontextprotocol/sdk/types.js';
import type { DatabaseMessageExtraMcpResource, McpServerOverride } from '$lib/types/database';
import type { SettingsConfigType } from '$lib/types/settings';

class MCPStore {
  private vIsInitializing = $state(false);
  private vError = $state<string | null>(null);
  private vToolCount = $state(0);
  private vConnectedServers = $state<string[]>([]);
  private vHealthChecks = $state<Record<string, HealthCheckState>>({});

  private dConnections = new Map<string, MCPConnection>();
  // tool name -> ids of the connected servers that list it; a name listed by more than
  // one server is ambiguous and only runs through an explicit server id
  private dToolsIndex = new Map<string, Set<string>>();
  // id sent in the _meta of tools/call -> running call of the local shell MCP server
  private dShellCalls = new SvelteMap<string, LocalShellActiveCall>();
  private dServerConfigs = new Map<string, MCPServerConfig>(); // Store configs for reconnection
  private sReconnectingServers = new Set<string>(); // Guard against concurrent reconnections
  private vConfigSignature: string | null = null;
  private vInitPromise: Promise<boolean> | null = null;
  private vActiveFlowCount = 0;

  get mIsProxyAvailable(): boolean {
    return cServerStore.vProps?.cors_proxy_enabled ?? false;
  }

  /**
   * Generates a unique server ID from an optional ID string or index.
   */
  #generateServerId(pId: unknown, pIndex: number): string {
    if (typeof pId === 'string' && pId.trim()) {
      return pId.trim();
    }

    return `${cMcpServerIdPrefix}-${pIndex + 1}`;
  }

  /**
   * Parses raw server settings from config into MCPServerSettingsEntry array.
   */
  #parseServerSettings(pRawServers: unknown): MCPServerSettingsEntry[] {
    if (!pRawServers) {
      return [];
    }

    let vParsed: unknown;
    if (typeof pRawServers === 'string') {
      const cTrimmed = pRawServers.trim();
      if (!cTrimmed) {
        return [];
      }

      try {
        vParsed = JSON.parse(cTrimmed);
      } catch (vError) {
        console.warn('[MCP] Failed to parse mcpServers JSON:', vError);

        return [];
      }
    } else {
      vParsed = pRawServers;
    }
    if (!Array.isArray(vParsed)) {
      return [];
    }

    return vParsed.map((pEntry, pIndex) => {
      const cUrl = typeof pEntry?.url === 'string' ? pEntry.url.trim() : '';
      const cHeaders = typeof pEntry?.headers === 'string' ? pEntry.headers.trim() : undefined;

      return {
        id: this.#generateServerId((pEntry as { id?: unknown })?.id, pIndex),
        enabled: Boolean((pEntry as { enabled?: unknown })?.enabled),
        url: cUrl,
        name: (pEntry as { name?: string })?.name,
        displayName: (pEntry as { displayName?: string })?.displayName,
        headers: cHeaders || undefined,
        useProxy: Boolean((pEntry as { useProxy?: unknown })?.useProxy)
      } satisfies MCPServerSettingsEntry;
    });
  }

  /**
   * Request timeout in milliseconds, read live from the global setting
   * so a change in Settings applies to every server immediately.
   */
  #requestTimeoutMs(): number {
    const cSeconds =
      Number(fConfig().mcpRequestTimeoutSeconds) || cDefaultMcpConfig.requestTimeoutSeconds;
    return Math.round(cSeconds * 1000);
  }

  /**
   * Builds server configuration from a settings entry.
   */
  #buildServerConfig(
    pEntry: MCPServerSettingsEntry,
    pConnectionTimeoutMs = cDefaultMcpConfig.connectionTimeoutMs
  ): MCPServerConfig | undefined {
    if (!pEntry?.url) {
      return undefined;
    }

    let dHeaders: Record<string, string> | undefined;
    if (pEntry.headers) {
      try {
        const cParsed = JSON.parse(pEntry.headers);
        if (typeof cParsed === 'object' && cParsed !== null && !Array.isArray(cParsed))
          dHeaders = cParsed as Record<string, string>;
      } catch {
        console.warn('[MCP] Failed to parse custom headers JSON:', pEntry.headers);
      }
    }

    return {
      url: pEntry.url,
      transport: detectMcpTransportFromUrl(pEntry.url),
      handshakeTimeoutMs: pConnectionTimeoutMs,
      requestTimeoutMs: this.#requestTimeoutMs(),
      headers: dHeaders,
      useProxy: pEntry.useProxy
    };
  }

  /**
   * Checks if a server is enabled for a given chat.
   * A per-chat override wins when present; a server without one resolves
   * to its own `enabled` flag in `mcpServers`.
   */
  #checkServerEnabled(
    pServer: MCPServerSettingsEntry,
    pPerChatOverrides?: McpServerOverride[]
  ): boolean {
    // Per-chat overrides win when present; missing entries inherit the
    // server's own `enabled` flag so partial override lists are not all
    // treated as disabled.
    const cOverride = pPerChatOverrides?.find((pO) => pO.serverId === pServer.id);
    return cOverride?.enabled ?? pServer.enabled;
  }

  /**
   * Builds MCP client configuration from settings.
   */
  #buildMcpClientConfig(
    pCfg: SettingsConfigType,
    pPerChatOverrides?: McpServerOverride[]
  ): MCPClientConfig | undefined {
    const cRawServers = this.#parseServerSettings(pCfg.mcpServers);
    if (!cRawServers.length) {
      return undefined;
    }

    const dServers: Record<string, MCPServerConfig> = {};

    for (const [cIndex, cEntry] of cRawServers.entries()) {
      if (!this.#checkServerEnabled(cEntry, pPerChatOverrides)) continue;
      const cNormalized = this.#buildServerConfig(cEntry);
      if (cNormalized) dServers[this.#generateServerId(cEntry.id, cIndex)] = cNormalized;
    }

    if (Object.keys(dServers).length === 0) {
      return undefined;
    }

    return {
      protocolVersion: cDefaultMcpConfig.protocolVersion,
      capabilities: cDefaultMcpConfig.capabilities,
      clientInfo: cDefaultMcpConfig.clientInfo,
      requestTimeoutMs: this.#requestTimeoutMs(),
      servers: dServers
    };
  }

  /**
   * Builds capabilities info from server and client capabilities.
   */
  #buildCapabilitiesInfo(
    pServerCaps?: ServerCapabilities,
    pClientCaps?: ClientCapabilities
  ): MCPCapabilitiesInfo {
    return {
      server: {
        tools: pServerCaps?.tools ? { listChanged: pServerCaps.tools.listChanged } : undefined,
        prompts: pServerCaps?.prompts
          ? { listChanged: pServerCaps.prompts.listChanged }
          : undefined,
        resources: pServerCaps?.resources
          ? {
              subscribe: pServerCaps.resources.subscribe,
              listChanged: pServerCaps.resources.listChanged
            }
          : undefined,
        logging: !!pServerCaps?.logging,
        completions: !!pServerCaps?.completions,
        tasks: !!pServerCaps?.tasks
      },
      client: {
        roots: pClientCaps?.roots ? { listChanged: pClientCaps.roots.listChanged } : undefined,
        sampling: !!pClientCaps?.sampling,
        elicitation: pClientCaps?.elicitation
          ? { form: !!pClientCaps.elicitation.form, url: !!pClientCaps.elicitation.url }
          : undefined,
        tasks: !!pClientCaps?.tasks
      }
    };
  }

  get mIsInitializing(): boolean {
    return this.vIsInitializing;
  }

  get mIsInitialized(): boolean {
    return this.dConnections.size > 0;
  }

  get mError(): string | null {
    return this.vError;
  }

  get mToolCount(): number {
    return this.vToolCount;
  }

  get mConnectedServerCount(): number {
    return this.vConnectedServers.length;
  }

  get mConnectedServerNames(): string[] {
    return this.vConnectedServers;
  }

  get mIsEnabled(): boolean {
    const cMcpConfig = this.#buildMcpClientConfig(fConfig());
    return (
      cMcpConfig !== null && cMcpConfig !== undefined && Object.keys(cMcpConfig.servers).length > 0
    );
  }

  get mAvailableTools(): string[] {
    return Array.from(this.dToolsIndex.keys());
  }

  private mUpdateState(pState: {
    isInitializing?: boolean;
    error?: string | null;
    toolCount?: number;
    connectedServers?: string[];
  }): void {
    if (pState.isInitializing !== undefined) {
      this.vIsInitializing = pState.isInitializing;
    }

    if (pState.error !== undefined) {
      this.vError = pState.error;
    }

    if (pState.toolCount !== undefined) {
      this.vToolCount = pState.toolCount;
    }

    if (pState.connectedServers !== undefined) {
      this.vConnectedServers = pState.connectedServers;
    }
  }

  mUpdateHealthCheck(pServerId: string, pState: HealthCheckState): void {
    this.vHealthChecks = { ...this.vHealthChecks, [pServerId]: pState };
  }

  mGetHealthCheckState(pServerId: string): HealthCheckState {
    return this.vHealthChecks[pServerId] ?? { status: HealthCheckStatus.IDLE };
  }

  mHasHealthCheck(pServerId: string): boolean {
    return (
      pServerId in this.vHealthChecks &&
      this.vHealthChecks[pServerId].status !== HealthCheckStatus.IDLE
    );
  }

  mClearHealthCheck(pServerId: string): void {
    const { [pServerId]: cRemoved, ...cRest } = this.vHealthChecks;
    this.vHealthChecks = cRest;
  }

  mClearAllHealthChecks(): void {
    this.vHealthChecks = {};
  }

  mClearError(): void {
    this.vError = null;
  }

  mGetServers(): MCPServerSettingsEntry[] {
    return parseMcpServerSettings(fConfig().mcpServers);
  }

  /**
   * Get all active MCP connections.
   * @returns Map of server names to connections
   */
  mGetConnections(): Map<string, MCPConnection> {
    return this.dConnections;
  }

  /**
   * Resolves the raw label for a server: user-defined display name first,
   * then server-reported title or name when the health check succeeded,
   * then the configured name (admin baseline or the local shell MCP server), then URL.
   */
  #serverBaseLabel(pServer: MCPServerDisplayInfo): string {
    if (pServer.displayName) return pServer.displayName;

    const cHealthState = this.mGetHealthCheckState(pServer.id);

    if (cHealthState?.status === HealthCheckStatus.SUCCESS)
      return (
        cHealthState.serverInfo?.title ||
        cHealthState.serverInfo?.name ||
        pServer.name ||
        pServer.url
      );
    return pServer.name || pServer.url;
  }

  /**
   * Returns the display label for a server, suffixed with a positional
   * counter when several configured servers resolve to the same base label
   * (e.g. two endpoints of the same host reporting an identical name).
   * Numbering follows config order, so it is stable across renders.
   */
  mGetServerLabel(pServer: MCPServerDisplayInfo): string {
    const cLabel = this.#serverBaseLabel(pServer);
    const cTwins = this.mGetServers().filter((pS) => this.#serverBaseLabel(pS) === cLabel);

    if (cTwins.length < 2) return cLabel;

    const cPosition = cTwins.findIndex((pS) => pS.id === pServer.id);

    return cPosition < 0 ? cLabel : `${cLabel} (${cPosition + 1})`;
  }

  mGetServerById(pServerId: string): MCPServerSettingsEntry | undefined {
    return this.mGetServers().find((pS) => pS.id === pServerId);
  }

  /**
   * Get display name for an MCP server by its ID.
   * Falls back to the server ID if server is not found.
   */
  mGetServerDisplayName(pServerId: string): string {
    const cServer = this.mGetServerById(pServerId);
    return cServer ? this.mGetServerLabel(cServer) : pServerId;
  }

  /**
   * Validates that an icon URI uses a safe scheme (https: or data:).
   */
  #isValidIconUri(pSrc: string): boolean {
    try {
      if (pSrc.startsWith(UrlProtocol.DATA)) return true;

      const cUrl = new URL(pSrc);

      return cUrl.protocol === UrlProtocol.HTTPS;
    } catch {
      return false;
    }
  }

  /**
   * Selects the best icon URL from an MCP icons array.
   * Follows security guidelines from the MCP specification:
   * - Only allows https: and data: URIs
   * - Filters to supported MIME types
   *
   * Selection priority:
   * 1. Icon matching the current color scheme (dark/light)
   * 2. Universal icon (no theme specified); if exactly 2, assumes [0]=light, [1]=dark
   * 3. First valid icon as last resort
   */
  #getMcpIconUrl(pIcons: MCPResourceIcon[] | undefined, pIsDark = false): string | null {
    if (!pIcons?.length) return null;

    const cValidIcons = pIcons.filter((pIcon) => {
      if (!pIcon.src || !this.#isValidIconUri(pIcon.src)) return false;
      if (pIcon.mimeType && !sMcpAllowedIconMimeTypes.has(pIcon.mimeType)) return false;
      return true;
    });

    if (cValidIcons.length === 0) return null;

    const cPreferredTheme = pIsDark ? ColorMode.DARK : ColorMode.LIGHT;

    // 1. Prefer icon explicitly matching the current color scheme
    const cThemedIcon = cValidIcons.find((pIcon) => pIcon.theme === cPreferredTheme);
    if (cThemedIcon) return cThemedIcon.src;

    // 2. Handle universal icons (no theme specified)
    const cUniversalIcons = cValidIcons.filter((pIcon) => !pIcon.theme);

    if (cUniversalIcons.length === cExpectedThemedIconPairCount) {
      // Heuristic: two theme-less icons → assume [0] = light, [1] = dark
      return cUniversalIcons[pIsDark ? 1 : 0].src;
    }

    if (cUniversalIcons.length > 0) {
      return cUniversalIcons[0].src;
    }

    // 3. Last resort: use opposite-theme icon
    return cValidIcons[0].src;
  }

  /**
   * Get icon URL for an MCP server by its ID.
   * Returns the best icon from the MCP server's `icons` array
   * (see MCP spec: spec.modelcontextprotocol.io).
   * Returns null if no icon is available.
   */
  mGetServerFavicon(pServerId: string): string | null {
    const cServer = this.mGetServerById(pServerId);
    if (!cServer) {
      return null;
    }

    const cIsDark = mode.current === ColorMode.DARK;
    const cHealthState = this.mGetHealthCheckState(pServerId);
    if (cHealthState.status === HealthCheckStatus.SUCCESS && cHealthState.serverInfo?.icons) {
      const cMcpIconUrl = this.#getMcpIconUrl(cHealthState.serverInfo.icons, cIsDark);

      if (cMcpIconUrl) {
        return cMcpIconUrl;
      }
    }

    return this.#getServerFaviconFallback(cServer.url);
  }

  /**
   * Construct a fallback favicon URL from the MCP server URL.
   * e.g. https://mcp.example.com/sse -> https://example.com/favicon.ico
   */
  #getServerFaviconFallback(pServerUrl: string): string | null {
    try {
      const cUrl = new URL(pServerUrl);
      const cRootDomain = extractRootDomain(cUrl);
      if (!cRootDomain) return null;

      const cOrigin = `${cUrl.protocol}//${cRootDomain}`;
      const aCandidates = ['favicon.ico', 'favicon.png'];

      for (const cPath of aCandidates) {
        const cFaviconUrl = `${cOrigin}/${cPath}`;
        if (this.#isValidIconUri(cFaviconUrl)) {
          return cFaviconUrl;
        }
      }
    } catch {
      // Invalid URL, return null
    }

    return null;
  }

  mAddServer(
    pServerData: Omit<MCPServerSettingsEntry, 'id'> & { id?: string }
  ): MCPServerSettingsEntry {
    const cServers = this.mGetServers();
    const dNewServer: MCPServerSettingsEntry = {
      id: pServerData.id || (uuid() ?? `server-${Date.now()}`),
      enabled: pServerData.enabled,
      url: pServerData.url.trim(),
      name: pServerData.name,
      displayName: pServerData.displayName,
      headers: pServerData.headers?.trim() || undefined,
      useProxy: pServerData.useProxy
    };
    cSettingsStore.mUpdateConfig(
      cSettingsKeys.MCP_SERVERS,
      JSON.stringify([...cServers, dNewServer])
    );
    return dNewServer;
  }

  mUpdateServer(pId: string, pUpdates: Partial<MCPServerSettingsEntry>): void {
    const cServers = this.mGetServers();
    cSettingsStore.mUpdateConfig(
      cSettingsKeys.MCP_SERVERS,
      JSON.stringify(
        cServers.map((pServer) => (pServer.id === pId ? { ...pServer, ...pUpdates } : pServer))
      )
    );
  }

  mRemoveServer(pId: string): void {
    const cServers = this.mGetServers();
    cSettingsStore.mUpdateConfig(
      cSettingsKeys.MCP_SERVERS,
      JSON.stringify(cServers.filter((pS) => pS.id !== pId))
    );
    this.mClearHealthCheck(pId);
  }

  /**
   * Mirrors the server's local shell MCP server (--local-shell-mcp-server) in the server list.
   * The entry is added enabled the first time the server offers it, its URL and name follow
   * the current origin, API prefix and SHELL_MCP_SERVER_NAME, and it is removed when the
   * server no longer offers it. While it is offered, the user's enabled/disabled choice and
   * any display name the user gives it are kept.
   */
  mSyncShellServer(pOffered: boolean): void {
    if (!browser) return;

    const vExisting = this.mGetServerById(cShellMcpServerId);
    if (!pOffered) {
      if (vExisting) this.mRemoveServer(cShellMcpServerId);
      return;
    }

    const vUrl = new URL(cShellMcpEndpoint, window.location.origin).href;
    if (!vExisting) {
      this.mAddServer({
        id: cShellMcpServerId,
        enabled: true,
        url: vUrl,
        name: cShellMcpServerName,
        useProxy: false
      });
      return;
    }

    if (vExisting.url !== vUrl || vExisting.useProxy) {
      this.mUpdateServer(cShellMcpServerId, {
        url: vUrl,
        name: cShellMcpServerName,
        useProxy: false
      });
      this.mClearHealthCheck(cShellMcpServerId);
    } else if (vExisting.name !== cShellMcpServerName) {
      this.mUpdateServer(cShellMcpServerId, { name: cShellMcpServerName });
    }
  }

  mHasAvailableServers(): boolean {
    return parseMcpServerSettings(fConfig().mcpServers).some((pS) => pS.enabled && pS.url.trim());
  }
  mHasEnabledServers(pPerChatOverrides?: McpServerOverride[]): boolean {
    return Boolean(this.#buildMcpClientConfig(fConfig(), pPerChatOverrides));
  }

  mGetEnabledServersForConversation(
    pPerChatOverrides?: McpServerOverride[]
  ): MCPServerSettingsEntry[] {
    return this.mGetServers().filter((pServer) => {
      return this.#checkServerEnabled(pServer, pPerChatOverrides);
    });
  }

  async mEnsureInitialized(pPerChatOverrides?: McpServerOverride[]): Promise<boolean> {
    if (!browser) {
      return false;
    }

    const cMcpConfig = this.#buildMcpClientConfig(fConfig(), pPerChatOverrides);
    const cSignature = cMcpConfig ? JSON.stringify(cMcpConfig) : null;
    if (!cSignature) {
      await this.mShutdown();

      return false;
    }
    if (this.mIsInitialized && this.vConfigSignature === cSignature) {
      return true;
    }

    if (this.vInitPromise && this.vConfigSignature === cSignature) {
      return this.vInitPromise;
    }

    if (this.dConnections.size > 0 || this.vInitPromise) await this.mShutdown();
    return this.mInitialize(cSignature, cMcpConfig!);
  }

  private async mInitialize(pSignature: string, pMcpConfig: MCPClientConfig): Promise<boolean> {
    this.mUpdateState({ isInitializing: true, error: null });
    this.vConfigSignature = pSignature;

    const cServerEntries = Object.entries(pMcpConfig.servers);

    if (cServerEntries.length === 0) {
      this.mUpdateState({ isInitializing: false, toolCount: 0, connectedServers: [] });

      return false;
    }
    this.vInitPromise = this.mDoInitialize(pSignature, pMcpConfig, cServerEntries);

    return this.vInitPromise;
  }

  private async mDoInitialize(
    pSignature: string,
    pMcpConfig: MCPClientConfig,
    pServerEntries: [string, MCPClientConfig['servers'][string]][]
  ): Promise<boolean> {
    const cClientInfo = pMcpConfig.clientInfo ?? cDefaultMcpConfig.clientInfo;
    const cCapabilities = pMcpConfig.capabilities ?? cDefaultMcpConfig.capabilities;
    const cResults = await Promise.allSettled(
      pServerEntries.map(async ([pName, pServerConfig]) => {
        // Store config for reconnection
        this.dServerConfigs.set(pName, pServerConfig);

        const cListChangedHandlers = this.mCreateListChangedHandlers(pName);
        const cConnection = await MCPService.mConnect(
          pName,
          pServerConfig,
          cClientInfo,
          cCapabilities,
          (pPhase) => {
            // Handle WebSocket disconnection
            if (pPhase === MCPConnectionPhase.DISCONNECTED) {
              console.log(`[MCPStore][${pName}] Connection lost, starting auto-reconnect`);
              this.mAutoReconnect(pName);
            }
          },
          cListChangedHandlers
        );

        return { name: pName, connection: cConnection };
      })
    );
    if (this.vConfigSignature !== pSignature) {
      for (const cResult of cResults) {
        if (cResult.status === 'fulfilled')
          await MCPService.mDisconnect(cResult.value.connection).catch(console.warn);
      }

      return false;
    }
    for (const cResult of cResults) {
      if (cResult.status === 'fulfilled') {
        const { name: cName, connection: cConnection } = cResult.value;

        this.dConnections.set(cName, cConnection);
        this.mIndexServerTools(cName, cConnection.tools);
      } else {
        console.error(`[MCPStore] Failed to connect:`, cResult.reason);
      }
    }

    const cSuccessCount = this.dConnections.size;
    if (cSuccessCount === 0 && pServerEntries.length > 0) {
      this.mUpdateState({
        isInitializing: false,
        error: fTranslate('text_5ea1d4562751'),
        toolCount: 0,
        connectedServers: []
      });
      this.vInitPromise = null;

      return false;
    }

    this.mUpdateState({
      isInitializing: false,
      error: null,
      toolCount: this.dToolsIndex.size,
      connectedServers: Array.from(this.dConnections.keys())
    });
    this.vInitPromise = null;

    return true;
  }

  private mCreateListChangedHandlers(pServerName: string): ListChangedHandlers {
    return {
      tools: {
        onChanged: (pError: Error | null, pTools: Tool[] | null) => {
          if (pError) {
            console.warn(`[MCPStore][${pServerName}] Tools list changed error:`, pError);
            return;
          }
          this.mHandleToolsListChanged(pServerName, pTools ?? []);
        }
      },
      prompts: {
        onChanged: (pError: Error | null) => {
          if (pError) {
            console.warn(`[MCPStore][${pServerName}] Prompts list changed error:`, pError);
            return;
          }
        }
      }
    };
  }

  private mHandleToolsListChanged(pServerName: string, pTools: Tool[]): void {
    const cConnection = this.dConnections.get(pServerName);
    if (!cConnection) {
      return;
    }

    cConnection.tools = pTools;
    this.mIndexServerTools(pServerName, pTools);
    this.mUpdateState({ toolCount: this.dToolsIndex.size });
  }

  /**
   * Replaces the tools a server contributes to the name index. A name that another
   * connected server also lists is kept for both and reported as a conflict: tool
   * calls always carry the id of the server they were approved for, and the tools
   * store withholds ambiguous names from the model.
   */
  private mIndexServerTools(pServerId: string, pTools: { name: string }[]): void {
    for (const [cToolName, sOwners] of this.dToolsIndex) {
      sOwners.delete(pServerId);
      if (sOwners.size === 0) this.dToolsIndex.delete(cToolName);
    }

    for (const cTool of pTools) {
      let sOwners = this.dToolsIndex.get(cTool.name);
      if (!sOwners) {
        sOwners = new Set<string>();
        this.dToolsIndex.set(cTool.name, sOwners);
      }
      sOwners.add(pServerId);
      if (sOwners.size > 1)
        console.warn(
          `[MCPStore] Tool name conflict: "${cTool.name}" is listed by ${[...sOwners].map((pId) => `"${pId}"`).join(', ')}. It is withheld from the model while more than one of them is enabled.`
        );
    }
  }

  /** The only connected server listing a tool name, or undefined when none or several do */
  private mGetUniqueToolOwner(pToolName: string): string | undefined {
    const sOwners = this.dToolsIndex.get(pToolName);
    if (!sOwners || sOwners.size !== 1) return undefined;
    return sOwners.values().next().value;
  }

  mAcquireConnection(): void {
    this.vActiveFlowCount++;
  }

  /**
   * Release a connection reference.
   * By default, keeps connections alive for reuse (shutdownIfUnused=false).
   * MCP spec encourages long-lived sessions to avoid reconnection overhead.
   */
  async mReleaseConnection(pShutdownIfUnused = false): Promise<void> {
    this.vActiveFlowCount = Math.max(0, this.vActiveFlowCount - 1);
    if (pShutdownIfUnused && this.vActiveFlowCount === 0) {
      await this.mShutdown();
    }
  }

  mGetActiveFlowCount(): number {
    return this.vActiveFlowCount;
  }

  async mShutdown(): Promise<void> {
    if (this.vInitPromise) {
      await this.vInitPromise.catch(() => {});
      this.vInitPromise = null;
    }

    if (this.dConnections.size === 0) {
      return;
    }

    await Promise.all(
      Array.from(this.dConnections.values()).map((pConn) =>
        MCPService.mDisconnect(pConn).catch((pError) =>
          console.warn(`[MCPStore] Error disconnecting ${pConn.serverName}:`, pError)
        )
      )
    );

    this.dConnections.clear();
    this.dToolsIndex.clear();
    this.dServerConfigs.clear();
    this.vConfigSignature = null;
    this.mUpdateState({
      isInitializing: false,
      error: null,
      toolCount: 0,
      connectedServers: []
    });
  }

  /**
   * Immediately reconnect to a server by creating a fresh transport and session.
   * Used when a session-expired error (HTTP 404) is detected during tool execution.
   * Per MCP spec 2025-11-25: client MUST discard session ID and re-initialize.
   *
   * Unlike autoReconnect (which uses exponential backoff for connectivity issues),
   * this performs a single immediate reconnection attempt since the server is known
   * to be reachable (it responded with 404).
   */
  private async mReconnectServer(pServerName: string): Promise<void> {
    const cServerConfig = this.dServerConfigs.get(pServerName);
    if (!cServerConfig) {
      throw new Error(`[MCPStore] No config found for ${pServerName}, cannot reconnect`);
    }

    // Disconnect stale connection (clears old transport + session ID)
    const cOldConnection = this.dConnections.get(pServerName);
    if (cOldConnection) {
      await MCPService.mDisconnect(cOldConnection).catch(console.warn);
      this.dConnections.delete(pServerName);
    }

    console.log(`[MCPStore][${pServerName}] Session expired, reconnecting with fresh session...`);

    const cListChangedHandlers = this.mCreateListChangedHandlers(pServerName);
    const cConnection = await MCPService.mConnect(
      pServerName,
      cServerConfig,
      cDefaultMcpConfig.clientInfo,
      cDefaultMcpConfig.capabilities,
      (pPhase) => {
        if (pPhase === MCPConnectionPhase.DISCONNECTED) {
          console.log(`[MCPStore][${pServerName}] Connection lost, starting auto-reconnect`);
          this.mAutoReconnect(pServerName);
        }
      },
      cListChangedHandlers
    );

    // Replace connection and rebuild tool index for this server
    this.dConnections.set(pServerName, cConnection);
    this.mIndexServerTools(pServerName, cConnection.tools);

    console.log(`[MCPStore][${pServerName}] Session recovered successfully`);
  }

  /**
   * Auto-reconnect to a server with exponential backoff.
   * Continues indefinitely until successful.
   *
   * Race-condition safety: when the phase callback fires a DISCONNECTED event
   * while we are still inside this function (e.g., the server drops right after
   * a successful connect()), a naive inner `autoReconnect()` call would be
   * swallowed by the `reconnectingServers` guard, leaving the server
   * permanently disconnected once the outer call exits. We solve this by
   * deferring the new reconnection via the `needsReconnect` flag: the flag is
   * set inside the phase callback and honoured in the `finally` block after
   * the guard entry has been removed.
   */
  private async mAutoReconnect(pServerName: string): Promise<void> {
    // Guard against concurrent reconnections
    if (this.sReconnectingServers.has(pServerName)) {
      console.log(`[MCPStore][${pServerName}] Reconnection already in progress, skipping`);

      return;
    }

    const cServerConfig = this.dServerConfigs.get(pServerName);
    if (!cServerConfig) {
      console.error(`[MCPStore] No config found for ${pServerName}, cannot reconnect`);

      return;
    }

    this.sReconnectingServers.add(pServerName);
    let vBackoff = cMcpReconnectInitialDelay;
    // Flag set by the phase callback when a DISCONNECTED event fires while
    // reconnectingServers still holds this server (see JSDoc above).
    let vNeedsReconnect = false;

    try {
      while (true) {
        await new Promise((pResolve) => setTimeout(pResolve, vBackoff));

        console.log(`[MCPStore][${pServerName}] Auto-reconnecting...`);

        try {
          // Per-attempt timeout: reject if the server doesn't respond in time,
          // then fall through to backoff logic as with any other failure.
          const cTimeoutPromise = new Promise<never>((pIgnored, pReject) =>
            setTimeout(
              () =>
                pReject(
                  new Error(fTranslate('text_001e61139ccb', { p0: cMcpReconnectAttemptTimeoutMs }))
                ),
              cMcpReconnectAttemptTimeoutMs
            )
          );

          vNeedsReconnect = false;
          const cListChangedHandlers = this.mCreateListChangedHandlers(pServerName);
          const cConnectPromise = MCPService.mConnect(
            pServerName,
            cServerConfig,
            cDefaultMcpConfig.clientInfo,
            cDefaultMcpConfig.capabilities,
            (pPhase) => {
              if (pPhase === MCPConnectionPhase.DISCONNECTED) {
                if (this.sReconnectingServers.has(pServerName)) {
                  // Reconnect loop is active; defer to after it exits.
                  vNeedsReconnect = true;
                } else {
                  console.log(
                    `[MCPStore][${pServerName}] Connection lost, restarting auto-reconnect`
                  );
                  this.mAutoReconnect(pServerName);
                }
              }
            },
            cListChangedHandlers
          );

          const cConnection = await Promise.race([cConnectPromise, cTimeoutPromise]);

          // Replace old connection with new one
          this.dConnections.set(pServerName, cConnection);

          // Rebuild tool index for this server
          this.mIndexServerTools(pServerName, cConnection.tools);

          console.log(`[MCPStore][${pServerName}] Reconnected successfully`);
          break;
        } catch (vError) {
          console.warn(`[MCPStore][${pServerName}] Reconnection failed:`, vError);
          vBackoff = Math.min(vBackoff * cMcpReconnectBackoffMultiplier, cMcpReconnectMaxDelay);
        }
      }
    } finally {
      this.sReconnectingServers.delete(pServerName);
      // If the phase callback signalled a disconnect while this function held
      // the guard, kick off a fresh reconnect now that the guard is released.
      if (vNeedsReconnect) {
        console.log(
          `[MCPStore][${pServerName}] Deferred disconnect detected, restarting auto-reconnect`
        );
        this.mAutoReconnect(pServerName);
      }
    }
  }

  mGetToolNames(): string[] {
    return Array.from(this.dToolsIndex.keys());
  }

  mHasTool(pToolName: string): boolean {
    return this.dToolsIndex.has(pToolName);
  }

  mGetToolServer(pToolName: string): string | undefined {
    return this.mGetUniqueToolOwner(pToolName);
  }

  /**
   * Resolve which configured MCP server owns a given tool name. Looks at
   * active connections first (fast path), then falls back to per-server
   * health-check data so server-side MCP proxies (where mgs
   * executes MCP tools but the browser does not hold a direct connection)
   * still resolve tool names to their owning server.
   */
  mFindServerForTool(pToolName: string): string | undefined {
    // A name listed by several connected servers has no single owner
    if (this.dToolsIndex.has(pToolName)) return this.mGetUniqueToolOwner(pToolName);

    let vOwner: string | undefined;
    for (const cServer of this.mGetServers()) {
      const cHealth = this.vHealthChecks[cServer.id];
      if (!cHealth || cHealth.status !== HealthCheckStatus.SUCCESS) continue;
      if (cHealth.tools.some((pTool) => pTool.name === pToolName)) {
        if (vOwner !== undefined) return undefined;
        vOwner = cServer.id;
      }
    }

    return vOwner;
  }

  /**
   * Resolve the favicon URL for an MCP server by one of its tool names.
   * Returns `null` if the tool is not provided by any configured MCP server,
   * or if the owning server has no icon to show.
   * Pair with {@link mGetServerFavicon} for direct server-id lookup.
   */
  mGetServerFaviconForTool(pToolName: string | undefined): string | null {
    if (!pToolName) return null;
    const cServerId = this.mFindServerForTool(pToolName);
    if (!cServerId) return null;
    return this.mGetServerFavicon(cServerId);
  }

  mHasPromptsSupport(): boolean {
    for (const cConnection of this.dConnections.values()) {
      if (cConnection.serverCapabilities?.prompts) {
        return true;
      }
    }

    return false;
  }

  /**
   * Check if any enabled server with successful health check supports prompts.
   * Uses health check state since servers may not have active connections until
   * the user actually sends a message or uses prompts.
   * @param pPerChatOverrides - Per-chat server overrides to filter by enabled servers.
   *                          If provided (even empty array), only checks enabled servers.
   *                          If undefined, falls back to each server's own `enabled` flag.
   */
  mHasPromptsCapability(pPerChatOverrides?: McpServerOverride[]): boolean {
    let sEnabledServerIds: Set<string>;

    if (pPerChatOverrides !== undefined) {
      sEnabledServerIds = new Set(
        pPerChatOverrides.filter((pO) => pO.enabled).map((pO) => pO.serverId)
      );
    } else {
      sEnabledServerIds = new Set(
        this.mGetServers()
          .filter((pS) => pS.enabled)
          .map((pS) => pS.id)
      );
    }

    if (sEnabledServerIds.size === 0) {
      return false;
    }

    for (const [cServerId, cState] of Object.entries(this.vHealthChecks)) {
      if (!sEnabledServerIds.has(cServerId)) continue;
      if (
        cState.status === HealthCheckStatus.SUCCESS &&
        cState.capabilities?.server?.prompts !== undefined
      ) {
        return true;
      }
    }

    for (const [cServerName, cConnection] of this.dConnections) {
      if (!sEnabledServerIds.has(cServerName)) continue;
      if (cConnection.serverCapabilities?.prompts) {
        return true;
      }
    }

    return false;
  }

  async mGetAllPrompts(): Promise<MCPPromptInfo[]> {
    const aResults: MCPPromptInfo[] = [];

    for (const [cServerName, cConnection] of this.dConnections) {
      if (!cConnection.serverCapabilities?.prompts) continue;

      const cPrompts = await MCPService.mListPrompts(cConnection);

      for (const cPrompt of cPrompts) {
        aResults.push({
          name: cPrompt.name,
          description: cPrompt.description,
          title: cPrompt.title,
          serverName: cServerName,
          arguments: cPrompt.arguments?.map((pArg) => ({
            name: pArg.name,
            description: pArg.description,
            required: pArg.required
          }))
        });
      }
    }

    return aResults;
  }

  async mGetPrompt(
    pServerName: string,
    pPromptName: string,
    pArgs?: Record<string, string>
  ): Promise<GetPromptResult> {
    const cConnection = this.dConnections.get(pServerName);
    if (!cConnection)
      throw new Error(fTranslate('text_7c335ada7906', { p0: pServerName, p1: pPromptName }));

    return MCPService.mGetPrompt(cConnection, pPromptName, pArgs);
  }

  /**
   * Runs a tool on the exact server it was resolved and approved for. The server id
   * is never looked up again by tool name, so another server listing the same name
   * can not receive the call.
   */
  async mExecuteTool(
    pToolCall: MCPToolCall,
    pServerId: string,
    pSignal?: AbortSignal
  ): Promise<ToolExecutionResult> {
    const cToolName = pToolCall.function.name;
    const cServerName = pServerId;

    const cConnection = this.dConnections.get(cServerName);
    if (!cConnection) throw new Error(fTranslate('text_d407790407b6', { p0: cServerName }));
    if (!cConnection.tools.some((pTool) => pTool.name === cToolName))
      throw new Error(fTranslate('text_7a0a26cba180', { p0: cToolName }));

    const cArgs = this.mParseToolArguments(pToolCall.function.arguments);

    // Calls of the local shell MCP server carry an id, so the password dialog can find and
    // answer the sudo password prompts of the command while it runs.
    const vShellCallId = cServerName === cShellMcpServerId ? uuid() : undefined;
    const dMeta = vShellCallId ? { [cShellMcpCallIdMeta]: vShellCallId } : undefined;
    if (vShellCallId) {
      this.dShellCalls.set(vShellCallId, {
        id: vShellCallId,
        toolName: cToolName,
        command: typeof cArgs.command === 'string' ? cArgs.command : ''
      });
    }

    try {
      return await MCPService.mCallTool(
        cConnection,
        { name: cToolName, arguments: cArgs, _meta: dMeta },
        pSignal
      );
    } catch (vError) {
      // Session expired (server restarted) - reconnect and retry once
      if (MCPService.mIsSessionExpiredError(vError)) {
        await this.mReconnectServer(cServerName);

        const cNewConnection = this.dConnections.get(cServerName);
        if (!cNewConnection) throw new Error(fTranslate('text_40846dfb3adb', { p0: cServerName }));

        return MCPService.mCallTool(
          cNewConnection,
          { name: cToolName, arguments: cArgs, _meta: dMeta },
          pSignal
        );
      }

      throw vError;
    } finally {
      if (vShellCallId) this.dShellCalls.delete(vShellCallId);
    }
  }

  /**
   * Calls of the local shell MCP server that are running; the server lists the sudo password
   * prompts of each command under its id.
   */
  mActiveShellCalls(): LocalShellActiveCall[] {
    return [...this.dShellCalls.values()];
  }

  async mExecuteToolByName(
    pToolName: string,
    pArgs: Record<string, unknown>,
    pSignal?: AbortSignal
  ): Promise<ToolExecutionResult> {
    // Only an unambiguous name runs without an explicit server id
    const cServerName = this.mGetUniqueToolOwner(pToolName);
    if (!cServerName) throw new Error(fTranslate('text_7a0a26cba180', { p0: pToolName }));
    const cConnection = this.dConnections.get(cServerName);
    if (!cConnection) throw new Error(fTranslate('text_d407790407b6', { p0: cServerName }));

    try {
      return await MCPService.mCallTool(
        cConnection,
        { name: pToolName, arguments: pArgs },
        pSignal
      );
    } catch (vError) {
      if (MCPService.mIsSessionExpiredError(vError)) {
        await this.mReconnectServer(cServerName);

        const cNewConnection = this.dConnections.get(cServerName);
        if (!cNewConnection) throw new Error(fTranslate('text_40846dfb3adb', { p0: cServerName }));

        return MCPService.mCallTool(cNewConnection, { name: pToolName, arguments: pArgs }, pSignal);
      }

      throw vError;
    }
  }

  private mParseToolArguments(pArgs: string | Record<string, unknown>): Record<string, unknown> {
    if (typeof pArgs === 'string') {
      const cTrimmed = pArgs.trim();
      if (cTrimmed === '') {
        return {};
      }

      try {
        const cParsed = JSON.parse(cTrimmed);
        if (typeof cParsed !== 'object' || cParsed === null || Array.isArray(cParsed))
          throw new Error(
            fTranslate('text_e663dc47f845', {
              p0: Array.isArray(cParsed) ? 'array' : typeof cParsed
            })
          );

        return cParsed as Record<string, unknown>;
      } catch (vError) {
        throw new Error(fTranslate('text_a7283305ef7e', { p0: (vError as Error).message }));
      }
    }

    if (typeof pArgs === 'object' && pArgs !== null && !Array.isArray(pArgs)) {
      return pArgs;
    }

    throw new Error(fTranslate('text_d27a35ba5f4a', { p0: typeof pArgs }));
  }

  async mGetPromptCompletions(
    pServerName: string,
    pPromptName: string,
    pArgumentName: string,
    pArgumentValue: string
  ): Promise<{ values: string[]; total?: number; hasMore?: boolean } | null> {
    const cConnection = this.dConnections.get(pServerName);
    if (!cConnection) {
      console.warn(`[MCPStore] Server "${pServerName}" is not connected`);
      return null;
    }
    if (!cConnection.serverCapabilities?.completions) {
      return null;
    }

    return MCPService.mComplete(
      cConnection,
      { type: MCPRefType.PROMPT, name: pPromptName },
      { name: pArgumentName, value: pArgumentValue }
    );
  }

  /**
   * Get completions for a resource template argument.
   * Uses the MCP Completion API with ref/resource.
   */
  async mGetResourceCompletions(
    pServerName: string,
    pUriTemplate: string,
    pArgumentName: string,
    pArgumentValue: string
  ): Promise<{ values: string[]; total?: number; hasMore?: boolean } | null> {
    const cConnection = this.dConnections.get(pServerName);

    if (!cConnection) {
      console.warn(`[MCPStore] Server "${pServerName}" is not connected`);
      return null;
    }

    if (!cConnection.serverCapabilities?.completions) {
      return null;
    }

    return MCPService.mComplete(
      cConnection,
      { type: MCPRefType.RESOURCE, uri: pUriTemplate },
      { name: pArgumentName, value: pArgumentValue }
    );
  }

  /**
   * Read a resource by an arbitrary URI (e.g., one expanded from a template).
   * Unlike readResource(), this does not require the URI to be in the resources list.
   */
  async mReadResourceByUri(
    pServerName: string,
    pUri: string
  ): Promise<MCPResourceContent[] | null> {
    const cConnection = this.dConnections.get(pServerName);

    if (!cConnection) {
      console.error(`[MCPStore] No connection found for server: ${pServerName}`);

      return null;
    }

    try {
      const cResult = await MCPService.mReadResource(cConnection, pUri);

      return cResult.contents;
    } catch (vError) {
      console.error(`[MCPStore] Failed to read resource ${pUri}:`, vError);

      return null;
    }
  }

  private mParseHeaders(pHeadersJson?: string): Record<string, string> | undefined {
    if (!pHeadersJson?.trim()) {
      return undefined;
    }

    try {
      const cParsed = JSON.parse(pHeadersJson);
      if (typeof cParsed === 'object' && cParsed !== null && !Array.isArray(cParsed))
        return cParsed as Record<string, string>;
    } catch {
      console.warn('[MCPStore] Failed to parse custom headers JSON:', pHeadersJson);
    }

    return undefined;
  }

  async mRunHealthChecksForServers(
    pServers: {
      id: string;
      enabled: boolean;
      url: string;
      headers?: string;
    }[],
    pSkipIfChecked = true,
    pPromoteToActive = false
  ): Promise<void> {
    const cServersToCheck = pSkipIfChecked
      ? pServers.filter((pS) => !this.mHasHealthCheck(pS.id) && pS.url.trim())
      : pServers.filter((pS) => pS.url.trim());

    if (cServersToCheck.length === 0) {
      return;
    }

    const cBatchSize = 5;
    for (let vI = 0; vI < cServersToCheck.length; vI += cBatchSize) {
      const cBatch = cServersToCheck.slice(vI, vI + cBatchSize);
      await Promise.allSettled(
        cBatch.map((pServer) => this.mRunHealthCheck(pServer, pPromoteToActive))
      );
    }
  }

  /**
   * Check if a server already has an active connection that can be reused.
   * Returns the existing connection if available.
   */
  mGetExistingConnection(pServerId: string): MCPConnection | undefined {
    return this.dConnections.get(pServerId);
  }

  /**
   * Run a health check for a server.
   * If the server already has an active connection, reuses it instead of creating a new one.
   * If promoteToActive is true and server is enabled, the connection will be kept
   * and promoted to an active connection instead of being disconnected.
   */
  async mRunHealthCheck(pServer: HealthCheckParams, pPromoteToActive = false): Promise<void> {
    // Check if we already have an active connection for this server
    const cExistingConnection = this.dConnections.get(pServer.id);
    if (cExistingConnection) {
      // Reuse existing connection - just refresh tools list
      try {
        const cTools = await MCPService.mListTools(cExistingConnection);
        const cCapabilities = this.#buildCapabilitiesInfo(
          cExistingConnection.serverCapabilities,
          cExistingConnection.clientCapabilities
        );
        this.mUpdateHealthCheck(pServer.id, {
          status: HealthCheckStatus.SUCCESS,
          tools: cTools.map((pTool) => ({
            name: pTool.name,
            description: pTool.description,
            title: pTool.title
          })),
          serverInfo: cExistingConnection.serverInfo,
          capabilities: cCapabilities,
          transportType: cExistingConnection.transportType,
          protocolVersion: cExistingConnection.protocolVersion,
          instructions: cExistingConnection.instructions,
          connectionTimeMs: cExistingConnection.connectionTimeMs,
          logs: []
        });
        return;
      } catch (vError) {
        console.warn(
          `[MCPStore] Failed to reuse connection for ${pServer.id}, creating new one:`,
          vError
        );
        // Connection may be stale, remove it and create new one
        this.dConnections.delete(pServer.id);
      }
    }

    const cTrimmedUrl = pServer.url.trim();
    const aLogs: MCPConnectionLog[] = [];
    let vCurrentPhase: MCPConnectionPhase = MCPConnectionPhase.IDLE;

    if (!cTrimmedUrl) {
      this.mUpdateHealthCheck(pServer.id, {
        status: HealthCheckStatus.ERROR,
        message: fTranslate('text_3ae5afc1cd7d'),
        logs: []
      });
      return;
    }

    this.mUpdateHealthCheck(pServer.id, {
      status: HealthCheckStatus.CONNECTING,
      phase: MCPConnectionPhase.TRANSPORT_CREATING,
      logs: []
    });

    const cTimeoutMs = this.#requestTimeoutMs();
    const cHeaders = this.mParseHeaders(pServer.headers);

    try {
      const dServerConfig: MCPServerConfig = {
        url: cTrimmedUrl,
        transport: detectMcpTransportFromUrl(cTrimmedUrl),
        handshakeTimeoutMs: cDefaultMcpConfig.connectionTimeoutMs,
        requestTimeoutMs: cTimeoutMs,
        headers: cHeaders,
        useProxy: pServer.useProxy
      };

      // Store config for reconnection
      this.dServerConfigs.set(pServer.id, dServerConfig);

      const cConnection = await MCPService.mConnect(
        pServer.id,
        dServerConfig,
        cDefaultMcpConfig.clientInfo,
        cDefaultMcpConfig.capabilities,
        (pPhase, pLog) => {
          vCurrentPhase = pPhase;
          aLogs.push(pLog);
          this.mUpdateHealthCheck(pServer.id, {
            status: HealthCheckStatus.CONNECTING,
            phase: pPhase,
            logs: [...aLogs]
          });

          // Handle WebSocket disconnection
          if (pPhase === MCPConnectionPhase.DISCONNECTED && pPromoteToActive) {
            console.log(
              `[MCPStore][${pServer.id}] Connection lost during health check, starting auto-reconnect`
            );
            this.mAutoReconnect(pServer.id);
          }
        }
      );

      const cTools = cConnection.tools.map((pTool) => ({
        name: pTool.name,
        description: pTool.description,
        title: pTool.title
      }));

      const cCapabilities = this.#buildCapabilitiesInfo(
        cConnection.serverCapabilities,
        cConnection.clientCapabilities
      );

      this.mUpdateHealthCheck(pServer.id, {
        status: HealthCheckStatus.SUCCESS,
        tools: cTools,
        serverInfo: cConnection.serverInfo,
        capabilities: cCapabilities,
        transportType: cConnection.transportType,
        protocolVersion: cConnection.protocolVersion,
        instructions: cConnection.instructions,
        connectionTimeMs: cConnection.connectionTimeMs,
        logs: aLogs
      });

      // Promote to active connection or disconnect
      if (pPromoteToActive && pServer.enabled) {
        this.mPromoteHealthCheckToConnection(pServer.id, cConnection);
      } else {
        await MCPService.mDisconnect(cConnection);
      }
    } catch (vError) {
      const cMessage = vError instanceof Error ? vError.message : fTranslate('text_43c4deec555b');

      if (aLogs.at(-1)?.phase !== MCPConnectionPhase.ERROR) {
        aLogs.push({
          timestamp: new Date(),
          phase: MCPConnectionPhase.ERROR,
          message: fTranslate('text_f9b1a54a491b', { p0: cMessage }),
          level: MCPLogLevel.ERROR
        });
      }

      this.mUpdateHealthCheck(pServer.id, {
        status: HealthCheckStatus.ERROR,
        message: cMessage,
        phase: vCurrentPhase,
        logs: aLogs
      });
    }
  }

  /**
   * Promote a health check connection to an active connection.
   * This avoids the need to reconnect when the server is needed for agentic flows.
   */
  private mPromoteHealthCheckToConnection(pServerId: string, pConnection: MCPConnection): void {
    // Register tools from the connection
    this.mIndexServerTools(pServerId, pConnection.tools);

    // Add to active connections
    this.dConnections.set(pServerId, pConnection);

    // Update state
    this.mUpdateState({
      toolCount: this.dToolsIndex.size,
      connectedServers: Array.from(this.dConnections.keys())
    });
  }

  mGetServersStatus(): ServerStatus[] {
    const aStatuses: ServerStatus[] = [];

    for (const [cName, cConnection] of this.dConnections) {
      aStatuses.push({
        name: cName,
        isConnected: true,
        toolCount: cConnection.tools.length,
        error: undefined
      });
    }

    return aStatuses;
  }

  /**
   * Get aggregated server instructions from all connected servers.
   * Returns an array of { serverName, serverTitle, instructions } objects.
   */
  mGetServerInstructions(): Array<{
    serverName: string;
    serverTitle?: string;
    instructions: string;
  }> {
    const aResults: Array<{ serverName: string; serverTitle?: string; instructions: string }> = [];

    for (const [cServerName, cConnection] of this.dConnections) {
      if (cConnection.instructions) {
        aResults.push({
          serverName: cServerName,
          serverTitle: cConnection.serverInfo?.title || cConnection.serverInfo?.name,
          instructions: cConnection.instructions
        });
      }
    }

    return aResults;
  }

  /**
   * Get server instructions from health check results (for display before active connection).
   * Useful for showing instructions in settings UI.
   */
  mGetHealthCheckInstructions(): Array<{
    serverId: string;
    serverTitle?: string;
    instructions: string;
  }> {
    const aResults: Array<{ serverId: string; serverTitle?: string; instructions: string }> = [];

    for (const [cServerId, cState] of Object.entries(this.vHealthChecks)) {
      if (cState.status === HealthCheckStatus.SUCCESS && cState.instructions) {
        aResults.push({
          serverId: cServerId,
          serverTitle: cState.serverInfo?.title || cState.serverInfo?.name,
          instructions: cState.instructions
        });
      }
    }

    return aResults;
  }

  /**
   * Check if any connected server has instructions.
   */
  mHasServerInstructions(): boolean {
    for (const cConnection of this.dConnections.values()) {
      if (cConnection.instructions) {
        return true;
      }
    }

    return false;
  }

  /**
   *
   *
   * Resources Operations
   *
   *
   */

  /**
   * Check if any enabled server with successful health check supports resources.
   * Uses health check state since servers may not have active connections until
   * the user actually sends a message or uses prompts.
   * @param pPerChatOverrides - Per-chat server overrides to filter by enabled servers.
   *                          If provided (even empty array), only checks enabled servers.
   *                          If undefined, falls back to each server's own `enabled` flag.
   */
  mHasResourcesCapability(pPerChatOverrides?: McpServerOverride[]): boolean {
    let sEnabledServerIds: Set<string>;

    if (pPerChatOverrides !== undefined) {
      sEnabledServerIds = new Set(
        pPerChatOverrides.filter((pO) => pO.enabled).map((pO) => pO.serverId)
      );
    } else {
      sEnabledServerIds = new Set(
        this.mGetServers()
          .filter((pS) => pS.enabled)
          .map((pS) => pS.id)
      );
    }
    if (sEnabledServerIds.size === 0) {
      return false;
    }

    for (const [cServerId, cState] of Object.entries(this.vHealthChecks)) {
      if (!sEnabledServerIds.has(cServerId)) continue;
      if (
        cState.status === HealthCheckStatus.SUCCESS &&
        cState.capabilities?.server?.resources !== undefined
      ) {
        return true;
      }
    }

    for (const [cServerName, cConnection] of this.dConnections) {
      if (!sEnabledServerIds.has(cServerName)) continue;
      if (MCPService.mSupportsResources(cConnection)) {
        return true;
      }
    }

    return false;
  }

  /**
   * Get list of enabled servers that support resources.
   * Checks active connections first, then health check state as fallback.
   */
  mGetServersWithResources(): string[] {
    const sEnabledServerIds = new Set(
      this.mGetServers()
        .filter((pS) => pS.enabled)
        .map((pS) => pS.id)
    );
    const aServers: string[] = [];

    // Check active connections
    for (const [cName, cConnection] of this.dConnections) {
      if (!sEnabledServerIds.has(cName)) continue;
      if (MCPService.mSupportsResources(cConnection) && !aServers.includes(cName)) {
        aServers.push(cName);
      }
    }

    // Also check health check states for servers not yet connected
    for (const [cServerId, cState] of Object.entries(this.vHealthChecks)) {
      if (!sEnabledServerIds.has(cServerId)) continue;
      if (
        !aServers.includes(cServerId) &&
        cState.status === HealthCheckStatus.SUCCESS &&
        cState.capabilities?.server?.resources !== undefined
      ) {
        aServers.push(cServerId);
      }
    }

    return aServers;
  }

  /**
   * Fetch resources from all connected servers that support them.
   * Updates mcpResourceStore with the results.
   * @param pForceRefresh - If true, bypass cache and fetch fresh data
   */
  async mFetchAllResources(pForceRefresh: boolean = false): Promise<void> {
    const cServersWithResources = this.mGetServersWithResources();
    if (cServersWithResources.length === 0) {
      return;
    }

    // Check if we have cached resources and they're recent (unless force refresh)
    if (!pForceRefresh) {
      const cAllServersCached = cServersWithResources.every((pServerName) => {
        const cServerRes = cMcpResourceStore.mGetServerResources(pServerName);
        if (!cServerRes || !cServerRes.lastFetched) {
          return false;
        }

        // Cache is valid for 5 minutes
        const cAge = Date.now() - cServerRes.lastFetched.getTime();

        return cAge < cDefaultCacheTtlMs;
      });

      if (cAllServersCached) {
        console.log('[MCPStore] Using cached resources');

        return;
      }
    }

    cMcpResourceStore.mSetLoading(true);

    try {
      await Promise.all(
        cServersWithResources.map((pServerName) => this.mFetchServerResources(pServerName))
      );
    } finally {
      cMcpResourceStore.mSetLoading(false);
    }
  }

  /**
   * Fetch resources from a specific server.
   * Updates mcpResourceStore with the results.
   */
  async mFetchServerResources(pServerName: string): Promise<void> {
    const cConnection = this.dConnections.get(pServerName);
    if (!cConnection) {
      console.warn(`[MCPStore] No connection found for server: ${pServerName}`);
      return;
    }

    if (!MCPService.mSupportsResources(cConnection)) {
      return;
    }

    cMcpResourceStore.mSetServerLoading(pServerName, true);

    try {
      const [cResources, cTemplates] = await Promise.all([
        MCPService.mListAllResources(cConnection),
        MCPService.mListAllResourceTemplates(cConnection)
      ]);

      cMcpResourceStore.mSetServerResources(pServerName, cResources, cTemplates);
    } catch (vError) {
      const cMessage = vError instanceof Error ? vError.message : String(vError);
      cMcpResourceStore.mSetServerError(pServerName, cMessage);
      console.error(`[MCPStore][${pServerName}] Failed to fetch resources:`, vError);
    }
  }

  /**
   * Read resource content from a server.
   * Caches the result in mcpResourceStore.
   */
  async mReadResource(pUri: string): Promise<MCPResourceContent[] | null> {
    // Check cache first
    const cCached = cMcpResourceStore.mGetCachedContent(pUri);
    if (cCached) {
      return cCached.content;
    }

    // Find which server has this resource
    const cServerName = cMcpResourceStore.mFindServerForUri(pUri);
    if (!cServerName) {
      console.error(`[MCPStore] No server found for resource URI: ${pUri}`);

      return null;
    }

    const cConnection = this.dConnections.get(cServerName);
    if (!cConnection) {
      console.error(`[MCPStore] No connection found for server: ${cServerName}`);

      return null;
    }

    try {
      const cResult = await MCPService.mReadResource(cConnection, pUri);
      const cResourceInfo = cMcpResourceStore.mFindResourceByUri(pUri);

      if (cResourceInfo) {
        cMcpResourceStore.mCacheResourceContent(cResourceInfo, cResult.contents);
      }

      return cResult.contents;
    } catch (vError) {
      console.error(`[MCPStore] Failed to read resource ${pUri}:`, vError);

      return null;
    }
  }

  /**
   * Subscribe to resource updates.
   */
  async mSubscribeToResource(pUri: string): Promise<boolean> {
    const cServerName = cMcpResourceStore.mFindServerForUri(pUri);
    if (!cServerName) {
      console.error(`[MCPStore] No server found for resource URI: ${pUri}`);

      return false;
    }

    const cConnection = this.dConnections.get(cServerName);
    if (!cConnection) {
      console.error(`[MCPStore] No connection found for server: ${cServerName}`);

      return false;
    }

    if (!MCPService.mSupportsResourceSubscriptions(cConnection)) {
      return false;
    }

    try {
      await MCPService.mSubscribeResource(cConnection, pUri);
      cMcpResourceStore.mAddSubscription(pUri, cServerName);

      return true;
    } catch (vError) {
      console.error(`[MCPStore] Failed to subscribe to resource ${pUri}:`, vError);

      return false;
    }
  }

  /**
   * Unsubscribe from resource updates.
   */
  async mUnsubscribeFromResource(pUri: string): Promise<boolean> {
    const cServerName = cMcpResourceStore.mFindServerForUri(pUri);
    if (!cServerName) {
      console.error(`[MCPStore] No server found for resource URI: ${pUri}`);

      return false;
    }

    const cConnection = this.dConnections.get(cServerName);
    if (!cConnection) {
      console.error(`[MCPStore] No connection found for server: ${cServerName}`);

      return false;
    }

    try {
      await MCPService.mUnsubscribeResource(cConnection, pUri);
      cMcpResourceStore.mRemoveSubscription(pUri);

      return true;
    } catch (vError) {
      console.error(`[MCPStore] Failed to unsubscribe from resource ${pUri}:`, vError);

      return false;
    }
  }

  /**
   * Add a resource as attachment to chat context.
   * Automatically fetches content if not cached.
   */
  async mAttachResource(pUri: string): Promise<MCPResourceAttachment | null> {
    const cResourceInfo = cMcpResourceStore.mFindResourceByUri(pUri);
    if (!cResourceInfo) {
      console.error(`[MCPStore] Resource not found: ${pUri}`);

      return null;
    }

    // Check if already attached
    if (cMcpResourceStore.mIsAttached(pUri)) {
      return null;
    }

    // Add attachment (initially loading)
    const cAttachment = cMcpResourceStore.mAddAttachment(cResourceInfo);

    // Fetch content
    try {
      const cContent = await this.mReadResource(pUri);

      if (cContent) {
        cMcpResourceStore.mUpdateAttachmentContent(cAttachment.id, cContent);
      } else {
        cMcpResourceStore.mUpdateAttachmentError(cAttachment.id, fTranslate('text_61847ecebbd0'));
      }
    } catch (vError) {
      const cMessage = vError instanceof Error ? vError.message : String(vError);
      cMcpResourceStore.mUpdateAttachmentError(cAttachment.id, cMessage);
    }

    return cMcpResourceStore.mGetAttachment(cAttachment.id) ?? null;
  }

  /**
   * Remove a resource attachment from chat context.
   */
  mRemoveResourceAttachment(pAttachmentId: string): void {
    cMcpResourceStore.mRemoveAttachment(pAttachmentId);
  }

  /**
   * Clear all resource attachments.
   */
  mClearResourceAttachments(): void {
    cMcpResourceStore.mClearAttachments();
  }

  /**
   * Get formatted resource context for chat.
   */
  mGetResourceContextForChat(): string {
    return cMcpResourceStore.mFormatAttachmentsForContext();
  }

  /**
   * Convert current resource attachments to DatabaseMessageExtra[] and clear them.
   * Called during message send to persist resources with the user message.
   */
  mConsumeResourceAttachmentsAsExtras(): DatabaseMessageExtraMcpResource[] {
    const cExtras = cMcpResourceStore.mToMessageExtras();
    if (cExtras.length > 0) {
      cMcpResourceStore.mClearAttachments();
    }
    return cExtras;
  }
}

export const cMcpStore = new MCPStore();

export const fMcpIsInitializing = () => cMcpStore.mIsInitializing;
export const fMcpIsInitialized = () => cMcpStore.mIsInitialized;
export const fMcpError = () => cMcpStore.mError;
export const fMcpIsEnabled = () => cMcpStore.mIsEnabled;
export const fMcpIsProxyAvailable = () => cMcpStore.mIsProxyAvailable;
export const fMcpAvailableTools = () => cMcpStore.mAvailableTools;
export const fMcpConnectedServerCount = () => cMcpStore.mConnectedServerCount;
export const fMcpConnectedServerNames = () => cMcpStore.mConnectedServerNames;
export const fMcpToolCount = () => cMcpStore.mToolCount;
export const fMcpServerInstructions = () => cMcpStore.mGetServerInstructions();
export const fMcpHasServerInstructions = () => cMcpStore.mHasServerInstructions();

// Resources exports
export const fMcpHasResourcesCapability = () => cMcpStore.mHasResourcesCapability();
export const fMcpServersWithResources = () => cMcpStore.mGetServersWithResources();
export const fMcpResourceContext = () => cMcpStore.mGetResourceContextForChat();
