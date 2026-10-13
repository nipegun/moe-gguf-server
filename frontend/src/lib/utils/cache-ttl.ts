import { cDefaultCacheTtlMs, cDefaultCacheMaxEntries } from '$lib/constants';

/**
 * TTL Cache - Time-To-Live cache implementation for memory optimization
 *
 * Provides automatic expiration of cached entries to prevent memory bloat
 * in long-running sessions.
 *
 * @example
 * ```ts
 * const cache = new TTLCache<string, ApiData>({ ttlMs: 5 * 60 * 1000 }); // 5 minutes
 * cache.set('key', data);
 * const value = cache.get('key'); // null if expired
 * ```
 */

export interface TTLCacheOptions {
  /** Time-to-live in milliseconds. Default: 5 minutes */
  ttlMs?: number;
  /** Maximum number of entries. Oldest entries are evicted when exceeded. Default: 100 */
  maxEntries?: number;
  /** Callback when an entry expires or is evicted */
  onEvict?: (pKey: string, pValue: unknown) => void;
}

interface CacheEntry<T> {
  value: T;
  expiresAt: number;
  lastAccessed: number;
}

export class TTLCache<K extends string, V> {
  private dCache = new Map<K, CacheEntry<V>>();
  private readonly vTtlMs: number;
  private readonly vMaxEntries: number;
  private readonly vOnEvict?: (pKey: string, pValue: unknown) => void;

  constructor(pOptions: TTLCacheOptions = {}) {
    this.vTtlMs = pOptions.ttlMs ?? cDefaultCacheTtlMs;
    this.vMaxEntries = pOptions.maxEntries ?? cDefaultCacheMaxEntries;
    this.vOnEvict = pOptions.onEvict;
  }

  /**
   * Get a value from cache. Returns null if expired or not found.
   */
  mGet(pKey: K): V | null {
    const cEntry = this.dCache.get(pKey);
    if (!cEntry) return null;

    if (Date.now() > cEntry.expiresAt) {
      this.mDelete(pKey);
      return null;
    }

    // Update last accessed time for LRU-like behavior
    cEntry.lastAccessed = Date.now();
    return cEntry.value;
  }

  /**
   * Set a value in cache with TTL.
   */
  mSet(pKey: K, pValue: V, pCustomTtlMs?: number): void {
    // Evict oldest entries if at capacity
    if (this.dCache.size >= this.vMaxEntries && !this.dCache.has(pKey)) {
      this.mEvictOldest();
    }

    const cTtl = pCustomTtlMs ?? this.vTtlMs;
    const cNow = Date.now();

    this.dCache.set(pKey, {
      value: pValue,
      expiresAt: cNow + cTtl,
      lastAccessed: cNow
    });
  }

  /**
   * Check if key exists and is not expired.
   */
  mHas(pKey: K): boolean {
    const cEntry = this.dCache.get(pKey);
    if (!cEntry) return false;

    if (Date.now() > cEntry.expiresAt) {
      this.mDelete(pKey);
      return false;
    }

    return true;
  }

  /**
   * Delete a specific key from cache.
   */
  mDelete(pKey: K): boolean {
    const cEntry = this.dCache.get(pKey);
    if (cEntry && this.vOnEvict) {
      this.vOnEvict(pKey, cEntry.value);
    }
    return this.dCache.delete(pKey);
  }

  /**
   * Clear all entries from cache.
   */
  mClear(): void {
    if (this.vOnEvict) {
      for (const [cKey, cEntry] of this.dCache) {
        this.vOnEvict(cKey, cEntry.value);
      }
    }
    this.dCache.clear();
  }

  /**
   * Get the number of entries (including potentially expired ones).
   */
  get mSize(): number {
    return this.dCache.size;
  }

  /**
   * Remove all expired entries from cache.
   * Call periodically for proactive cleanup.
   */
  mPrune(): number {
    const cNow = Date.now();
    let vPruned = 0;

    for (const [cKey, cEntry] of this.dCache) {
      if (cNow > cEntry.expiresAt) {
        this.mDelete(cKey);
        vPruned++;
      }
    }

    return vPruned;
  }

  /**
   * Get all valid (non-expired) keys.
   */
  mKeys(): K[] {
    const cNow = Date.now();
    const aValidKeys: K[] = [];

    for (const [cKey, cEntry] of this.dCache) {
      if (cNow <= cEntry.expiresAt) {
        aValidKeys.push(cKey);
      }
    }

    return aValidKeys;
  }

  /**
   * Evict the oldest (least recently accessed) entry.
   */
  private mEvictOldest(): void {
    let vOldestKey: K | null = null;
    let vOldestTime = Infinity;

    for (const [cKey, cEntry] of this.dCache) {
      if (cEntry.lastAccessed < vOldestTime) {
        vOldestTime = cEntry.lastAccessed;
        vOldestKey = cKey;
      }
    }

    if (vOldestKey !== null) {
      this.mDelete(vOldestKey);
    }
  }

  /**
   * Refresh TTL for an existing entry without changing the value.
   */
  mTouch(pKey: K): boolean {
    const cEntry = this.dCache.get(pKey);
    if (!cEntry) return false;

    const cNow = Date.now();
    if (cNow > cEntry.expiresAt) {
      this.mDelete(pKey);
      return false;
    }

    cEntry.expiresAt = cNow + this.vTtlMs;
    cEntry.lastAccessed = cNow;
    return true;
  }
}

/**
 * Reactive TTL Map for Svelte stores
 * Wraps SvelteMap with TTL functionality
 */
export class ReactiveTTLMap<K extends string, V> {
  private vEntries = $state<Map<K, CacheEntry<V>>>(new Map());
  private readonly vTtlMs: number;
  private readonly vMaxEntries: number;

  constructor(pOptions: TTLCacheOptions = {}) {
    this.vTtlMs = pOptions.ttlMs ?? cDefaultCacheTtlMs;
    this.vMaxEntries = pOptions.maxEntries ?? cDefaultCacheMaxEntries;
  }

  mGet(pKey: K): V | null {
    const cEntry = this.vEntries.get(pKey);
    if (!cEntry) return null;

    if (Date.now() > cEntry.expiresAt) {
      this.vEntries.delete(pKey);
      return null;
    }

    cEntry.lastAccessed = Date.now();
    return cEntry.value;
  }

  mSet(pKey: K, pValue: V, pCustomTtlMs?: number): void {
    if (this.vEntries.size >= this.vMaxEntries && !this.vEntries.has(pKey)) {
      this.mEvictOldest();
    }

    const cTtl = pCustomTtlMs ?? this.vTtlMs;
    const cNow = Date.now();

    this.vEntries.set(pKey, {
      value: pValue,
      expiresAt: cNow + cTtl,
      lastAccessed: cNow
    });
  }

  mHas(pKey: K): boolean {
    const cEntry = this.vEntries.get(pKey);
    if (!cEntry) return false;

    if (Date.now() > cEntry.expiresAt) {
      this.vEntries.delete(pKey);
      return false;
    }

    return true;
  }

  mDelete(pKey: K): boolean {
    return this.vEntries.delete(pKey);
  }

  mClear(): void {
    this.vEntries.clear();
  }

  get mSize(): number {
    return this.vEntries.size;
  }

  mPrune(): number {
    const cNow = Date.now();
    let vPruned = 0;

    for (const [cKey, cEntry] of this.vEntries) {
      if (cNow > cEntry.expiresAt) {
        this.vEntries.delete(cKey);
        vPruned++;
      }
    }

    return vPruned;
  }

  private mEvictOldest(): void {
    let vOldestKey: K | null = null;
    let vOldestTime = Infinity;

    for (const [cKey, cEntry] of this.vEntries) {
      if (cEntry.lastAccessed < vOldestTime) {
        vOldestTime = cEntry.lastAccessed;
        vOldestKey = cKey;
      }
    }

    if (vOldestKey !== null) {
      this.vEntries.delete(vOldestKey);
    }
  }
}
