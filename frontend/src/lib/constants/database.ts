/**
 * Database-related constants (IndexedDB, Dexie).
 *
 * Centralized to ensure consistency across the app and simplify future
 * naming changes.
 */

import { cStorageAppName } from './storage';

/** IndexedDB database name */
export const cDbName = cStorageAppName;

/** IndexedDB store / table names */
export const cIdxdbTables = {
  conversations: 'conversations',
  messages: 'messages'
} as const;

/** IndexedDB store schemas */
export const cIdxdbStoreSchemas = {
  conversations: 'id, lastModified, currNode, name',
  messages: 'id, convId, type, role, timestamp, parent, children'
} as const;

/** Combined Dexie stores definition — keys are table names, values are schemas */
export const cIdxdbStores = {
  [cIdxdbTables.conversations]: cIdxdbStoreSchemas.conversations,
  [cIdxdbTables.messages]: cIdxdbStoreSchemas.messages
} as const;
