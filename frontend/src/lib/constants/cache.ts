/**
 * Cache configuration constants
 */

/**
 * Default TTL (Time-To-Live) for cache entries in milliseconds
 * @default 5 minutes
 */
export const cDefaultCacheTtlMs = 5 * 60 * 1000;

/**
 * Default maximum number of entries in a cache
 * @default 100
 */
export const cDefaultCacheMaxEntries = 100;

/**
 * TTL for model props cache in milliseconds
 * Props don't change frequently, so we can cache them longer
 * @default 10 minutes
 */
export const cModelPropsCacheTtlMs = 10 * 60 * 1000;

/**
 * Maximum number of model props to cache
 * @default 50
 */
export const cModelPropsCacheMaxEntries = 50;

/**
 * Maximum number of MCP resources to cache
 * @default 50
 */
export const cMcpResourceCacheMaxEntries = 50;

/**
 * TTL for MCP resource cache entries in milliseconds
 * @default 5 minutes
 */
export const cMcpResourceCacheTtlMs = 5 * 60 * 1000;

/**
 * Maximum number of inactive conversation states to keep in memory
 * States for conversations beyond this limit will be cleaned up
 * @default 10
 */
export const cMaxInactiveConversationStates = 10;

/**
 * Maximum age (in ms) for inactive conversation states before cleanup
 * States older than this will be removed during cleanup
 * @default 30 minutes
 */
export const cInactiveConversationStateMaxAgeMs = 30 * 60 * 1000;
