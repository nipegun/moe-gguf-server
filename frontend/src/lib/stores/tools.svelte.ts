import type { OpenAIToolDefinition, ToolEntry, ToolGroup } from '$lib/types';
import { ToolsService } from '$lib/services/tools.service';
import { ApiError } from '$lib/utils/api-fetch';
import { cMcpStore } from '$lib/stores/mcp.svelte';
import { HealthCheckStatus, JsonSchemaType, ToolCallType, ToolSource } from '$lib/enums';
import { fConfig } from '$lib/stores/settings.svelte';
import { cUiSettingsStore } from '$lib/stores/ui-settings.svelte';
import {
  fBuildSandboxToolDefinition,
  cToolGroupLabels,
  cToolServerLabels
} from '$lib/constants';

import { SvelteMap, SvelteSet } from 'svelte/reactivity';

/** Stable selection identity for a tool, shared by the disabled set and the permission store */

class ToolsStore {
  private vBuiltinTools = $state<OpenAIToolDefinition[]>([]);
  private vLoading = $state(false);
  private vError = $state<string | null>(null);
  private vDisabledTools = $state(new SvelteSet<string>());
  private vToolsEndpointUnreachable = $state(false);

  constructor() {
    // disabled tools are kept in the server's settings document (section "disabledToolKeys")
    cUiSettingsStore.mRegister('disabledToolKeys', (pSaved) => {
      this.vDisabledTools.clear();
      if (Array.isArray(pSaved)) {
        for (const cKey of pSaved) {
          if (typeof cKey === 'string') this.vDisabledTools.add(cKey);
        }
      }
    });

    this.mFetchBuiltinTools();
  }

  private mPersistDisabledTools(): void {
    cUiSettingsStore.mSet('disabledToolKeys', [...this.vDisabledTools]);
  }

  private mToolKey(pSource: ToolSource, pName: string, pServerId?: string): string {
    switch (pSource) {
      case ToolSource.MCP:
        return pServerId ? `mcp-${pServerId}:${pName}` : `mcp:${pName}`;
      case ToolSource.CUSTOM:
        return `custom:${pName}`;
      case ToolSource.FRONTEND:
        return `frontend:${pName}`;
      default:
        return `builtin:${pName}`;
    }
  }

  private mInferTypeFromDefault(pValue: unknown): string | undefined {
    if (typeof pValue === 'string') return 'string';
    if (typeof pValue === 'boolean') return 'boolean';
    if (typeof pValue === 'number') return Number.isInteger(pValue) ? 'integer' : 'number';
    if (Array.isArray(pValue)) return 'array';
    if (pValue !== null && typeof pValue === 'object') return 'object';
    return undefined;
  }

  /**
   * Recursively normalize a JSON Schema object: infers `type` from `default`
   * for properties / items that omit it, and descends into nested `properties`
   * and `items`. Returns a new object -- does not mutate the input.
   */
  private mNormalizeJsonSchema(pSchema: Record<string, unknown>): Record<string, unknown> {
    if (!pSchema || typeof pSchema !== 'object') return pSchema;

    const dNormalized: Record<string, unknown> = { ...pSchema };

    if (dNormalized.properties && typeof dNormalized.properties === 'object') {
      const cProps = dNormalized.properties as Record<string, Record<string, unknown>>;
      const dNormalizedProps: Record<string, Record<string, unknown>> = {};
      for (const [cKey, cProp] of Object.entries(cProps)) {
        if (!cProp || typeof cProp !== 'object') {
          dNormalizedProps[cKey] = cProp;
          continue;
        }

        const dNormalizedProp: Record<string, unknown> = { ...cProp };

        if (!dNormalizedProp.type && dNormalizedProp.default !== undefined) {
          const cInferred = this.mInferTypeFromDefault(dNormalizedProp.default);
          if (cInferred) dNormalizedProp.type = cInferred;
        }

        if (dNormalizedProp.properties) {
          Object.assign(
            dNormalizedProp,
            this.mNormalizeJsonSchema(dNormalizedProp as Record<string, unknown>)
          );
        }

        if (dNormalizedProp.items && typeof dNormalizedProp.items === 'object') {
          dNormalizedProp.items = this.mNormalizeJsonSchema(
            dNormalizedProp.items as Record<string, unknown>
          );
        }

        dNormalizedProps[cKey] = dNormalizedProp;
      }
      dNormalized.properties = dNormalizedProps;
    }

    return dNormalized;
  }

  private mMcpDefinition(
    pName: string,
    pDescription: string | undefined,
    pSchema?: Record<string, unknown>
  ): OpenAIToolDefinition {
    return {
      type: ToolCallType.FUNCTION,
      function: {
        name: pName,
        description: pDescription,
        parameters: pSchema ?? { type: JsonSchemaType.OBJECT, properties: {}, required: [] }
      }
    };
  }

  get mBuiltinTools(): OpenAIToolDefinition[] {
    return this.vBuiltinTools;
  }

  get mMcpTools(): OpenAIToolDefinition[] {
    return this.mMcpEntries().map((pE) => pE.definition);
  }

  get mFrontendTools(): OpenAIToolDefinition[] {
    return fConfig().jsSandboxEnabled
      ? [fBuildSandboxToolDefinition(!!fConfig().symbolicMathEnabled)]
      : [];
  }

  get mCustomTools(): OpenAIToolDefinition[] {
    const cRaw = fConfig().customJson;
    if (!cRaw || typeof cRaw !== 'string') return [];

    try {
      const cParsed = JSON.parse(cRaw);
      if (!Array.isArray(cParsed)) return [];

      return cParsed.filter(
        (pT: unknown): pT is OpenAIToolDefinition =>
          typeof pT === 'object' &&
          pT !== null &&
          'type' in pT &&
          (pT as OpenAIToolDefinition).type === 'function' &&
          'function' in pT &&
          typeof (pT as OpenAIToolDefinition).function?.name === 'string'
      );
    } catch {
      return [];
    }
  }

  /** Normalize MCP tools from live connections when available, fall back to health check data */
  private mMcpEntries(): {
    serverId: string;
    serverName: string;
    definition: OpenAIToolDefinition;
  }[] {
    const aOut: { serverId: string; serverName: string; definition: OpenAIToolDefinition }[] = [];

    const cConnections = cMcpStore.mGetConnections();
    if (cConnections.size > 0) {
      for (const [cServerId, cConnection] of cConnections) {
        const cServerName = cMcpStore.mGetServerDisplayName(cServerId);
        for (const cTool of cConnection.tools) {
          const cRawSchema = (cTool.inputSchema as Record<string, unknown>) ?? {
            type: JsonSchemaType.OBJECT,
            properties: {},
            required: []
          };
          aOut.push({
            serverId: cServerId,
            serverName: cServerName,
            definition: {
              type: ToolCallType.FUNCTION,
              function: {
                name: cTool.name,
                description: cTool.description,
                parameters: this.mNormalizeJsonSchema(cRawSchema)
              }
            }
          });
        }
      }
    } else {
      for (const {
        serverId: cServerId,
        serverName: cServerName,
        tools: cTools
      } of this.mGetMcpToolsFromHealthChecks()) {
        for (const cTool of cTools) {
          aOut.push({
            serverId: cServerId,
            serverName: cServerName,
            definition: this.mMcpDefinition(cTool.name, cTool.description)
          });
        }
      }
    }

    return aOut;
  }

  /** Canonical flat list of tool entries with source metadata and stable keys, deduped by key */
  get mAllTools(): ToolEntry[] {
    const aEntries: ToolEntry[] = [];
    const cSeen = new SvelteSet<string>();

    const fPush = (pEntry: ToolEntry) => {
      if (cSeen.has(pEntry.key)) return;
      cSeen.add(pEntry.key);
      aEntries.push(pEntry);
    };

    for (const cDef of this.vBuiltinTools) {
      const cName = cDef.function.name;
      fPush({
        source: ToolSource.BUILTIN,
        key: this.mToolKey(ToolSource.BUILTIN, cName),
        definition: cDef
      });
    }

    for (const cDef of this.mFrontendTools) {
      const cName = cDef.function.name;
      fPush({
        source: ToolSource.FRONTEND,
        key: this.mToolKey(ToolSource.FRONTEND, cName),
        definition: cDef
      });
    }

    for (const {
      serverId: cServerId,
      serverName: cServerName,
      definition: cDefinition
    } of this.mMcpEntries()) {
      const cName = cDefinition.function.name;
      fPush({
        source: ToolSource.MCP,
        serverId: cServerId,
        serverName: cServerName,
        key: this.mToolKey(ToolSource.MCP, cName, cServerId),
        definition: cDefinition
      });
    }

    for (const cDef of this.mCustomTools) {
      const cName = cDef.function.name;
      fPush({
        source: ToolSource.CUSTOM,
        key: this.mToolKey(ToolSource.CUSTOM, cName),
        definition: cDef
      });
    }

    return aEntries;
  }

  /** Tools grouped by category for tree display, derived from the canonical entries */
  get mToolGroups(): ToolGroup[] {
    const aGroups: ToolGroup[] = [];
    const cByKey = new SvelteMap<string, ToolGroup>();

    for (const cEntry of this.mAllTools) {
      const cGroupKey =
        cEntry.source === ToolSource.MCP ? `mcp:${cEntry.serverId ?? ''}` : cEntry.source;

      let vGroup = cByKey.get(cGroupKey);
      if (!vGroup) {
        vGroup = {
          source: cEntry.source,
          key: cGroupKey,
          label: this.mGroupLabel(cEntry),
          serverId: cEntry.serverId,
          tools: []
        };
        cByKey.set(cGroupKey, vGroup);
        aGroups.push(vGroup);
      }

      vGroup.tools.push(cEntry);
    }

    return aGroups;
  }

  private mGroupLabel(pEntry: ToolEntry): string {
    switch (pEntry.source) {
      case ToolSource.MCP:
        return pEntry.serverName ?? '';
      case ToolSource.CUSTOM:
        return cToolGroupLabels[ToolSource.CUSTOM];
      case ToolSource.FRONTEND:
        return cToolGroupLabels[ToolSource.FRONTEND];
      default:
        return cToolGroupLabels[ToolSource.BUILTIN];
    }
  }

  /**
   * The one enabled tool a bare tool name refers to, the identity shared by what
   * the LLM is offered, the permission asked for and the tool that runs.
   * Sources keep their precedence (built-in, frontend/sandbox, MCP, custom JSON):
   * the first source with an enabled tool of that name owns it. When that source
   * is MCP and more than one enabled server lists the name, the name is ambiguous
   * and resolves to nothing, so no server can run under another server's approval.
   * Disabled tools never resolve.
   */
  mResolveEnabledEntry(pToolName: string): ToolEntry | null {
    return this.mResolveEnabledEntryIn(this.mAllTools, pToolName);
  }

  /** {@link mResolveEnabledEntry} over an already built entry list */
  private mResolveEnabledEntryIn(pEntries: ToolEntry[], pToolName: string): ToolEntry | null {
    let vResolved: ToolEntry | null = null;

    for (const cEntry of pEntries) {
      if (cEntry.definition.function.name !== pToolName) continue;
      if (this.vDisabledTools.has(cEntry.key)) continue;

      if (!vResolved) {
        vResolved = cEntry;
        continue;
      }

      // Lower-precedence sources are shadowed by the owner already found
      if (vResolved.source !== cEntry.source) break;

      // Two enabled MCP servers list the same name: ambiguous
      if (cEntry.source === ToolSource.MCP && cEntry.serverId !== vResolved.serverId) return null;
    }

    return vResolved;
  }

  /**
   * Tool names listed by more than one enabled MCP server and therefore withheld
   * from the LLM until the user disables all but one of them.
   */
  get mConflictingMcpToolNames(): string[] {
    const cEntries = this.mAllTools;
    const dServersByName = new SvelteMap<string, SvelteSet<string>>();

    for (const cEntry of cEntries) {
      if (cEntry.source !== ToolSource.MCP || this.vDisabledTools.has(cEntry.key)) continue;
      const cName = cEntry.definition.function.name;
      let sServers = dServersByName.get(cName);
      if (!sServers) {
        sServers = new SvelteSet<string>();
        dServersByName.set(cName, sServers);
      }
      sServers.add(cEntry.serverId ?? '');
    }

    const aConflicts: string[] = [];
    for (const [cName, sServers] of dServersByName) {
      if (sServers.size > 1 && this.mResolveEnabledEntryIn(cEntries, cName) === null)
        aConflicts.push(cName);
    }

    return aConflicts;
  }

  /**
   * Enabled tool definitions for sending to the LLM.
   * MCP tool schemas are normalized here so the wire payload is consistent
   * across all four sources (built-in, frontend/sandbox, MCP, custom JSON).
   * The API identifies tools by name, so a name is sent at most once, and only
   * when it resolves to a single tool (see {@link mResolveEnabledEntry}).
   */
  mGetEnabledToolsForLLM(): OpenAIToolDefinition[] {
    const cEntries = this.mAllTools;
    const aResult: OpenAIToolDefinition[] = [];
    const sSeen = new SvelteSet<string>();

    for (const cEntry of cEntries) {
      const cName = cEntry.definition.function.name;
      if (sSeen.has(cName)) continue;
      sSeen.add(cName);

      const cResolved = this.mResolveEnabledEntryIn(cEntries, cName);
      if (cResolved) aResult.push(cResolved.definition);
    }

    return aResult;
  }

  get mAllToolDefinitions(): OpenAIToolDefinition[] {
    return this.mAllTools.map((pT) => pT.definition);
  }

  get mLoading(): boolean {
    return this.vLoading;
  }

  get mError(): string | null {
    return this.vError;
  }

  get mIsToolsEndpointUnreachable(): boolean {
    return this.vToolsEndpointUnreachable;
  }

  get mDisabledTools(): SvelteSet<string> {
    return this.vDisabledTools;
  }

  mIsToolEnabled(pKey: string): boolean {
    return !this.vDisabledTools.has(pKey);
  }

  mToggleTool(pKey: string): void {
    if (this.vDisabledTools.has(pKey)) {
      this.vDisabledTools.delete(pKey);
    } else {
      this.vDisabledTools.add(pKey);
    }
    this.mPersistDisabledTools();
  }

  mSetToolEnabled(pKey: string, pEnabled: boolean): void {
    if (pEnabled) {
      this.vDisabledTools.delete(pKey);
    } else {
      this.vDisabledTools.add(pKey);
    }
  }

  /** Enable all tools belonging to a specific MCP server */
  mEnableAllToolsForServer(pServerId: string): void {
    const cConnection = cMcpStore.mGetConnections().get(pServerId);
    if (!cConnection) return;
    for (const cTool of cConnection.tools) {
      this.vDisabledTools.delete(this.mToolKey(ToolSource.MCP, cTool.name, pServerId));
    }
    this.mPersistDisabledTools();
  }

  mToggleGroup(pGroup: ToolGroup): void {
    const cAllEnabled = pGroup.tools.every((pT) => this.mIsToolEnabled(pT.key));
    const cTarget = !cAllEnabled;
    for (const cTool of pGroup.tools) {
      if (cTarget) this.vDisabledTools.delete(cTool.key);
      else this.vDisabledTools.add(cTool.key);
    }
    this.mPersistDisabledTools();
  }

  mIsGroupFullyEnabled(pGroup: ToolGroup): boolean {
    return pGroup.tools.length > 0 && pGroup.tools.every((pT) => this.mIsToolEnabled(pT.key));
  }

  /** Get MCP tools from health check data, used when live connections aren't established yet */
  private mGetMcpToolsFromHealthChecks(): {
    serverId: string;
    serverName: string;
    tools: { name: string; description?: string }[];
  }[] {
    const aResult: ReturnType<ToolsStore['mGetMcpToolsFromHealthChecks']> = [];
    for (const cServer of cMcpStore.mGetServers()) {
      if (!cServer.enabled) continue;
      const cHealth = cMcpStore.mGetHealthCheckState(cServer.id);
      if (cHealth.status === HealthCheckStatus.SUCCESS && cHealth.tools.length > 0) {
        aResult.push({
          serverId: cServer.id,
          serverName: cMcpStore.mGetServerLabel(cServer),
          tools: cHealth.tools
        });
      }
    }
    return aResult;
  }

  /** Determine the source of the tool a name resolves to */
  mGetToolSource(pToolName: string): ToolSource | null {
    return this.mResolveEnabledEntry(pToolName)?.source ?? null;
  }

  /** Get the display label for the server that owns a tool entry */
  mGetEntryServerLabel(pEntry: ToolEntry): string {
    if (pEntry.serverName) return cMcpStore.mGetServerDisplayName(pEntry.serverName);
    if (pEntry.source === ToolSource.BUILTIN) return cToolServerLabels[ToolSource.BUILTIN];
    if (pEntry.source === ToolSource.CUSTOM) return cToolServerLabels[ToolSource.CUSTOM];
    if (pEntry.source === ToolSource.FRONTEND) return cToolServerLabels[ToolSource.FRONTEND];
    return '';
  }

  /** Get the display label for the server that owns the tool a name resolves to */
  mGetToolServerLabel(pToolName: string): string {
    const cEntry = this.mResolveEnabledEntry(pToolName);
    return cEntry ? this.mGetEntryServerLabel(cEntry) : '';
  }

  /** Permission key of the tool a name resolves to, identical to the selection key */
  mGetPermissionKey(pToolName: string): string | null {
    return this.mResolveEnabledEntry(pToolName)?.key ?? null;
  }

  /** Check if there are any enabled tools available (builtin, MCP, or custom) */
  get mHasEnabledTools(): boolean {
    return this.mGetEnabledToolsForLLM().length > 0;
  }

  async mFetchBuiltinTools(): Promise<void> {
    if (this.vLoading) return;

    this.vLoading = true;
    this.vError = null;
    this.vToolsEndpointUnreachable = false;

    try {
      const cToolInfos = await ToolsService.mList();
      this.vBuiltinTools = cToolInfos.map((pInfo) => pInfo.definition);
    } catch (vErr) {
      this.vError = vErr instanceof Error ? vErr.message : String(vErr);
      // 403 from /tools means the server was started without --tools
      if (vErr instanceof ApiError && vErr.vStatus === 403) {
        this.vToolsEndpointUnreachable = true;
        console.info('[ToolsStore] Built-in tools are disabled on the server');
      } else {
        console.error('[ToolsStore] Failed to fetch built-in tools:', vErr);
      }
    } finally {
      this.vLoading = false;
    }
  }
}

export const cToolsStore = new ToolsStore();

export const fAllTools = () => cToolsStore.mAllTools;
export const fAllToolDefinitions = () => cToolsStore.mAllToolDefinitions;
export const fToolGroups = () => cToolsStore.mToolGroups;
