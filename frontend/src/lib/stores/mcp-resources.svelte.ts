/**
 * mcpResourceStore - Reactive State Store for MCP Resources
 *
 * Manages MCP protocol resources:
 * - Resource discovery and listing per server
 * - Resource content caching
 * - Resource subscriptions
 * - Resource attachments for chat context
 *
 * @see MCP Protocol Specification: https://modelcontextprotocol.io/specification/2025-06-18/server/resources
 */

import { SvelteMap } from 'svelte/reactivity';
import { AttachmentType } from '$lib/enums';
import {
  cMcpResourceAttachmentIdPrefix,
  cMcpResourceCacheMaxEntries,
  cMcpResourceCacheTtlMs,
  cNewline,
  cResourceUnknownType,
  cBinaryContentLabel
} from '$lib/constants';
import { normalizeResourceUri } from '$lib/utils';
import type {
  MCPResource,
  MCPResourceTemplate,
  MCPResourceContent,
  MCPResourceInfo,
  MCPResourceTemplateInfo,
  MCPCachedResource,
  MCPResourceAttachment,
  MCPResourceSubscription,
  MCPServerResources,
  DatabaseMessageExtraMcpResource
} from '$lib/types';

function fGenerateAttachmentId(): string {
  return `${cMcpResourceAttachmentIdPrefix}-${Date.now()}-${Math.random().toString(36).substring(2, 9)}`;
}

class MCPResourceStore {
  private vServerResources = $state<SvelteMap<string, MCPServerResources>>(new SvelteMap());
  private vCachedResources = $state<SvelteMap<string, MCPCachedResource>>(new SvelteMap());
  private vSubscriptions = $state<SvelteMap<string, MCPResourceSubscription>>(new SvelteMap());
  private vAttachments = $state<MCPResourceAttachment[]>([]);
  private vIsLoading = $state(false);

  get mServerResources(): Map<string, MCPServerResources> {
    return this.vServerResources;
  }

  get mCachedResources(): Map<string, MCPCachedResource> {
    return this.vCachedResources;
  }

  get mSubscriptions(): Map<string, MCPResourceSubscription> {
    return this.vSubscriptions;
  }

  get mAttachments(): MCPResourceAttachment[] {
    return this.vAttachments;
  }

  get mIsLoading(): boolean {
    return this.vIsLoading;
  }

  get mTotalResourceCount(): number {
    let vCount = 0;
    for (const cServerRes of this.vServerResources.values()) {
      vCount += cServerRes.resources.length;
    }

    return vCount;
  }

  get mTotalTemplateCount(): number {
    let vCount = 0;
    for (const cServerRes of this.vServerResources.values()) {
      vCount += cServerRes.templates.length;
    }

    return vCount;
  }

  get mAttachmentCount(): number {
    return this.vAttachments.length;
  }

  get mHasAttachments(): boolean {
    return this.vAttachments.length > 0;
  }

  /**
   *
   *
   * Server Resources Management
   *
   *
   */

  /**
   * Set resources for a server (called after listResources)
   */
  mSetServerResources(
    pServerName: string,
    pResources: MCPResource[],
    pTemplates: MCPResourceTemplate[]
  ): void {
    this.vServerResources.set(pServerName, {
      serverName: pServerName,
      resources: pResources,
      templates: pTemplates,
      lastFetched: new Date(),
      loading: false,
      error: undefined
    });
    console.log(
      `[MCPResources][${pServerName}] Set ${pResources.length} resources, ${pTemplates.length} templates`
    );
  }

  /**
   * Set loading state for a server's resources
   */
  mSetServerLoading(pServerName: string, pLoading: boolean): void {
    const cExisting = this.vServerResources.get(pServerName);
    if (cExisting) {
      this.vServerResources.set(pServerName, { ...cExisting, loading: pLoading });
    } else {
      this.vServerResources.set(pServerName, {
        serverName: pServerName,
        resources: [],
        templates: [],
        loading: pLoading,
        error: undefined
      });
    }
  }

  /**
   * Set error state for a server's resources
   */
  mSetServerError(pServerName: string, pError: string): void {
    const cExisting = this.vServerResources.get(pServerName);

    if (cExisting) {
      this.vServerResources.set(pServerName, { ...cExisting, loading: false, error: pError });
    } else {
      this.vServerResources.set(pServerName, {
        serverName: pServerName,
        resources: [],
        templates: [],
        loading: false,
        error: pError
      });
    }
  }

  /**
   * Get resources for a specific server
   */
  mGetServerResources(pServerName: string): MCPServerResources | undefined {
    return this.vServerResources.get(pServerName);
  }

  /**
   * Get all resources as MCPResourceInfo array (flattened with server names)
   */
  mGetAllResourceInfos(): MCPResourceInfo[] {
    const aResult: MCPResourceInfo[] = [];

    for (const [cServerName, cServerRes] of this.vServerResources) {
      for (const cResource of cServerRes.resources) {
        aResult.push({
          uri: cResource.uri,
          name: cResource.name,
          title: cResource.title,
          description: cResource.description,
          mimeType: cResource.mimeType,
          serverName: cServerName,
          annotations: cResource.annotations,
          icons: cResource.icons
        });
      }
    }

    return aResult;
  }

  /**
   * Get all templates as MCPResourceTemplateInfo array (flattened with server names)
   */
  mGetAllTemplateInfos(): MCPResourceTemplateInfo[] {
    const aResult: MCPResourceTemplateInfo[] = [];

    for (const [cServerName, cServerRes] of this.vServerResources) {
      for (const cTemplate of cServerRes.templates) {
        aResult.push({
          uriTemplate: cTemplate.uriTemplate,
          name: cTemplate.name,
          title: cTemplate.title,
          description: cTemplate.description,
          mimeType: cTemplate.mimeType,
          serverName: cServerName,
          annotations: cTemplate.annotations,
          icons: cTemplate.icons
        });
      }
    }

    return aResult;
  }

  /**
   * Clear resources for a server (e.g., when disconnected)
   */
  mClearServerResources(pServerName: string): void {
    this.vServerResources.delete(pServerName);

    // Also clear cached content for this server's resources
    for (const [cUri, cCached] of this.vCachedResources) {
      if (cCached.resource.serverName === pServerName) {
        this.vCachedResources.delete(cUri);
      }
    }

    // Clear subscriptions for this server
    for (const [cUri, cSub] of this.vSubscriptions) {
      if (cSub.serverName === pServerName) {
        this.vSubscriptions.delete(cUri);
      }
    }

    console.log(`[MCPResources][${pServerName}] Cleared all resources`);
  }

  /**
   *
   *
   * Resource Content Caching
   *
   *
   */

  /**
   * Cache resource content after reading
   */
  mCacheResourceContent(pResource: MCPResourceInfo, pContent: MCPResourceContent[]): void {
    // Enforce cache size limit
    if (this.vCachedResources.size >= cMcpResourceCacheMaxEntries) {
      // Remove oldest entry
      const cOldestKey = this.vCachedResources.keys().next().value;

      if (cOldestKey) {
        this.vCachedResources.delete(cOldestKey);
      }
    }

    this.vCachedResources.set(pResource.uri, {
      resource: pResource,
      content: pContent,
      fetchedAt: new Date(),
      subscribed: this.vSubscriptions.has(pResource.uri)
    });
    console.log(`[MCPResources] Cached content for: ${pResource.uri}`);
  }

  /**
   * Get cached content for a resource
   */
  mGetCachedContent(pUri: string): MCPCachedResource | undefined {
    const cCached = this.vCachedResources.get(pUri);
    if (!cCached) return undefined;

    // Check if cache is still valid
    const cAge = Date.now() - cCached.fetchedAt.getTime();

    if (cAge > cMcpResourceCacheTtlMs && !cCached.subscribed) {
      // Cache expired and not subscribed, remove it
      this.vCachedResources.delete(pUri);

      return undefined;
    }

    return cCached;
  }

  /**
   * Invalidate cached content for a resource (e.g., on update notification)
   */
  mInvalidateCache(pUri: string): void {
    this.vCachedResources.delete(pUri);
    console.log(`[MCPResources] Invalidated cache for: ${pUri}`);
  }

  /**
   * Clear all cached content
   */
  mClearCache(): void {
    this.vCachedResources.clear();
    console.log(`[MCPResources] Cleared all cached content`);
  }

  /**
   *
   *
   * Subscriptions
   *
   *
   */

  /**
   * Register a subscription for a resource
   */
  mAddSubscription(pUri: string, pServerName: string): void {
    this.vSubscriptions.set(pUri, {
      uri: pUri,
      serverName: pServerName,
      subscribedAt: new Date()
    });

    // Update cached resource if exists
    const cCached = this.vCachedResources.get(pUri);
    if (cCached) {
      this.vCachedResources.set(pUri, { ...cCached, subscribed: true });
    }

    console.log(`[MCPResources] Added subscription: ${pUri}`);
  }

  /**
   * Remove a subscription for a resource
   */
  mRemoveSubscription(pUri: string): void {
    this.vSubscriptions.delete(pUri);

    // Update cached resource if exists
    const cCached = this.vCachedResources.get(pUri);
    if (cCached) {
      this.vCachedResources.set(pUri, { ...cCached, subscribed: false });
    }

    console.log(`[MCPResources] Removed subscription: ${pUri}`);
  }

  /**
   * Check if a resource is subscribed
   */
  mIsSubscribed(pUri: string): boolean {
    return this.vSubscriptions.has(pUri);
  }

  /**
   * Handle resource update notification
   */
  mHandleResourceUpdate(pUri: string): void {
    // Invalidate cache so next read gets fresh content
    this.mInvalidateCache(pUri);

    // Update subscription last update time
    const cSub = this.vSubscriptions.get(pUri);
    if (cSub) {
      this.vSubscriptions.set(pUri, { ...cSub, lastUpdate: new Date() });
    }

    console.log(`[MCPResources] Resource updated: ${pUri}`);
  }

  /**
   * Handle resources list changed notification
   */
  mHandleResourcesListChanged(pServerName: string): void {
    // Mark server resources as needing refresh
    const cExisting = this.vServerResources.get(pServerName);
    if (cExisting) {
      this.vServerResources.set(pServerName, {
        ...cExisting,
        lastFetched: undefined // Mark as stale
      });
    }
    console.log(`[MCPResources][${pServerName}] Resources list changed, needs refresh`);
  }

  /**
   *
   *
   * Attachments (for chat context)
   *
   *
   */

  /**
   * Add a resource attachment to the current chat context
   */
  mAddAttachment(pResource: MCPResourceInfo): MCPResourceAttachment {
    const dAttachment: MCPResourceAttachment = {
      id: fGenerateAttachmentId(),
      resource: pResource,
      loading: true
    };

    this.vAttachments = [...this.vAttachments, dAttachment];
    console.log(`[MCPResources] Added attachment: ${pResource.uri}`);

    return dAttachment;
  }

  /**
   * Update attachment with fetched content
   */
  mUpdateAttachmentContent(pAttachmentId: string, pContent: MCPResourceContent[]): void {
    this.vAttachments = this.vAttachments.map((pAtt) =>
      pAtt.id === pAttachmentId
        ? { ...pAtt, content: pContent, loading: false, error: undefined }
        : pAtt
    );
  }

  /**
   * Update attachment with error
   */
  mUpdateAttachmentError(pAttachmentId: string, pError: string): void {
    this.vAttachments = this.vAttachments.map((pAtt) =>
      pAtt.id === pAttachmentId ? { ...pAtt, loading: false, error: pError } : pAtt
    );
  }

  /**
   * Remove an attachment
   */
  mRemoveAttachment(pAttachmentId: string): void {
    this.vAttachments = this.vAttachments.filter((pAtt) => pAtt.id !== pAttachmentId);
    console.log(`[MCPResources] Removed attachment: ${pAttachmentId}`);
  }

  /**
   * Clear all attachments
   */
  mClearAttachments(): void {
    this.vAttachments = [];
    console.log(`[MCPResources] Cleared all attachments`);
  }

  /**
   * Get attachment by ID
   */
  mGetAttachment(pAttachmentId: string): MCPResourceAttachment | undefined {
    return this.vAttachments.find((pAtt) => pAtt.id === pAttachmentId);
  }

  /**
   * Check if a resource is already attached
   */
  mIsAttached(pUri: string): boolean {
    const cNormalizedUri = normalizeResourceUri(pUri);

    return this.vAttachments.some(
      (pAtt) =>
        pAtt.resource.uri === pUri || normalizeResourceUri(pAtt.resource.uri) === cNormalizedUri
    );
  }

  /**
   *
   *
   * Utility Methods
   *
   *
   */

  /**
   * Set global loading state
   */
  mSetLoading(pLoading: boolean): void {
    this.vIsLoading = pLoading;
  }

  /**
   * Find resource info by URI across all servers
   */
  mFindResourceByUri(pUri: string): MCPResourceInfo | undefined {
    const cNormalizedUri = normalizeResourceUri(pUri);

    for (const [cServerName, cServerRes] of this.vServerResources) {
      const cResource =
        cServerRes.resources.find((pR) => pR.uri === pUri) ??
        cServerRes.resources.find((pR) => normalizeResourceUri(pR.uri) === cNormalizedUri);

      if (cResource) {
        return {
          uri: cResource.uri,
          name: cResource.name,
          title: cResource.title,
          description: cResource.description,
          mimeType: cResource.mimeType,
          serverName: cServerName,
          annotations: cResource.annotations,
          icons: cResource.icons
        };
      }
    }

    return undefined;
  }

  /**
   * Find server name for a resource URI
   */
  mFindServerForUri(pUri: string): string | undefined {
    for (const [cServerName, cServerRes] of this.vServerResources) {
      if (cServerRes.resources.some((pR) => pR.uri === pUri)) {
        return cServerName;
      }
    }

    return undefined;
  }

  /**
   * Clear all state (e.g., on full reset)
   */
  mClear(): void {
    this.vServerResources.clear();
    this.vCachedResources.clear();
    this.vSubscriptions.clear();
    this.vAttachments = [];
    this.vIsLoading = false;
    console.log(`[MCPResources] Cleared all state`);
  }

  /**
   * Get resource content as text for chat context
   * Formats content for inclusion in LLM prompts
   */
  mFormatAttachmentsForContext(): string {
    if (this.vAttachments.length === 0) return '';

    const aParts: string[] = [];

    for (const cAttachment of this.vAttachments) {
      if (cAttachment.error) continue;
      if (!cAttachment.content || cAttachment.content.length === 0) continue;

      const cResourceName = cAttachment.resource.title || cAttachment.resource.name;
      const cServerName = cAttachment.resource.serverName;

      for (const cContent of cAttachment.content) {
        if ('text' in cContent && cContent.text) {
          aParts.push(
            `\n\n--- Resource: ${cResourceName} (from ${cServerName}) ---\n${cContent.text}`
          );
        } else if ('blob' in cContent && cContent.blob) {
          // For binary content, just note it exists
          aParts.push(
            `\n\n--- Resource: ${cResourceName} (from ${cServerName}) ---\n[${cBinaryContentLabel}: ${cContent.mimeType || cResourceUnknownType}]`
          );
        }
      }
    }

    return aParts.join('');
  }

  /**
   * Convert current resource attachments to DatabaseMessageExtra[] for persisting with a message.
   * Each attachment becomes a DatabaseMessageExtraMcpResource stored on the user message.
   */
  mToMessageExtras(): DatabaseMessageExtraMcpResource[] {
    const aExtras: DatabaseMessageExtraMcpResource[] = [];

    for (const cAttachment of this.vAttachments) {
      if (cAttachment.error) continue;
      if (!cAttachment.content || cAttachment.content.length === 0) continue;

      const cResourceName = cAttachment.resource.title || cAttachment.resource.name;
      const aContentParts: string[] = [];

      for (const cContent of cAttachment.content) {
        if ('text' in cContent && cContent.text) {
          aContentParts.push(cContent.text);
        } else if ('blob' in cContent && cContent.blob) {
          aContentParts.push(
            `[${cBinaryContentLabel}: ${cContent.mimeType || cResourceUnknownType}]`
          );
        }
      }

      if (aContentParts.length > 0) {
        aExtras.push({
          type: AttachmentType.MCP_RESOURCE,
          name: cResourceName,
          uri: cAttachment.resource.uri,
          serverName: cAttachment.resource.serverName,
          content: aContentParts.join(cNewline),
          mimeType: cAttachment.resource.mimeType
        });
      }
    }

    return aExtras;
  }
}

export const cMcpResourceStore = new MCPResourceStore();

// Export convenience functions
export const fMcpResources = () => cMcpResourceStore.mServerResources;
export const fMcpResourceAttachments = () => cMcpResourceStore.mAttachments;
export const fMcpResourceAttachmentCount = () => cMcpResourceStore.mAttachmentCount;
export const fMcpHasResourceAttachments = () => cMcpResourceStore.mHasAttachments;
export const fMcpTotalResourceCount = () => cMcpResourceStore.mTotalResourceCount;
export const fMcpResourcesLoading = () => cMcpResourceStore.mIsLoading;
