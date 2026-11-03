import { PropsService } from '$lib/services/props.service';
import { ServerRole } from '$lib/enums';
import { ApiError } from '$lib/utils/api-fetch';

const cLoadingRetryIntervalMs = 1000;

/**
 * serverStore - Server connection state, configuration, and role detection
 *
 * This store manages the server connection state and properties fetched from `/props`.
 * It provides reactive state for server configuration and role detection.
 *
 * **Architecture & Relationships:**
 * - **PropsService**: Stateless service for fetching `/props` data
 * - **serverStore** (this class): Reactive store for server state
 * - **modelsStore**: Independent store for model management (uses PropsService directly)
 *
 * **Key Features:**
 * - **Server State**: Connection status, loading, error handling
 * - **Role Detection**: MODEL (single model) vs ROUTER (multi-model)
 * - **Default Params**: Server-wide generation defaults
 */
class ServerStore {
  /**
   *
   *
   * State
   *
   *
   */

  vProps = $state<ApiMgsServerProps | null>(null);
  vLoading = $state(false);
  vError = $state<string | null>(null);
  vStatus = $state<number | null>(null);
  vRole = $state<ServerRole | null>(null);
  private vFetchPromise: Promise<void> | null = null;
  private vRetryTimer: ReturnType<typeof setTimeout> | null = null;

  /**
   *
   *
   * Getters
   *
   *
   */

  get mDefaultParams(): ApiMgsServerProps['default_generation_settings']['params'] | null {
    return this.vProps?.default_generation_settings?.params || null;
  }

  get mContextSize(): number | null {
    const cNCtx = this.vProps?.default_generation_settings?.n_ctx;

    return typeof cNCtx === 'number' ? cNCtx : null;
  }

  get mUiSettings(): Record<string, string | number | boolean> | undefined {
    return this.vProps?.ui_settings;
  }

  get mIsRouterMode(): boolean {
    return this.vRole === ServerRole.ROUTER;
  }

  get mIsModelMode(): boolean {
    return this.vRole === ServerRole.MODEL;
  }

  /**
   *
   *
   * Data Handling
   *
   *
   */

  /**
   * @param background - Set by the automatic "still loading" poll. Skips the
   * `loading` flag flip so the UI doesn't bounce between the full loading
   * splash and the chat screen every retry tick.
   */
  async mFetch({ background: pBackground = false }: { background?: boolean } = {}): Promise<void> {
    if (this.vFetchPromise) return this.vFetchPromise;

    this.mClearRetryTimer();
    if (!pBackground) {
      this.vLoading = true;
    }
    // Don't clear an existing "still loading" error before a retry -
    // doing so would unmount/remount the error banner every second.
    if (this.vStatus !== 503) {
      this.vError = null;
    }

    const cFetchPromise = (async () => {
      try {
        const cProps = await PropsService.mFetch();
        this.vProps = cProps;
        this.vError = null;
        this.vStatus = null;
        this.mDetectRole(cProps);
      } catch (vError: unknown) {
        this.vError = vError instanceof Error ? vError.message : String(vError);
        this.vStatus = vError instanceof ApiError ? vError.vStatus : null;
        console.error('Error fetching server properties:', vError);

        if (this.vStatus === 503) {
          this.mScheduleRetry();
        }
      } finally {
        if (!pBackground) {
          this.vLoading = false;
        }
        this.vFetchPromise = null;
      }
    })();

    this.vFetchPromise = cFetchPromise;
    await cFetchPromise;
  }

  mClear(): void {
    this.mClearRetryTimer();
    this.vProps = null;
    this.vError = null;
    this.vStatus = null;
    this.vLoading = false;
    this.vRole = null;
    this.vFetchPromise = null;
  }

  private mScheduleRetry(): void {
    if (this.vRetryTimer) return;
    this.vRetryTimer = setTimeout(() => {
      this.vRetryTimer = null;
      this.mFetch({ background: true });
    }, cLoadingRetryIntervalMs);
  }

  private mClearRetryTimer(): void {
    if (this.vRetryTimer) {
      clearTimeout(this.vRetryTimer);
      this.vRetryTimer = null;
    }
  }

  /**
   *
   *
   * Utilities
   *
   *
   */

  private mDetectRole(pProps: ApiMgsServerProps): void {
    const cNewRole = pProps?.role === ServerRole.ROUTER ? ServerRole.ROUTER : ServerRole.MODEL;
    if (this.vRole !== cNewRole) {
      this.vRole = cNewRole;
      console.info(`Server running in ${cNewRole === ServerRole.ROUTER ? 'ROUTER' : 'MODEL'} mode`);
    }
  }
}

export const cServerStore = new ServerStore();

export const fServerProps = () => cServerStore.vProps;
export const fServerLoading = () => cServerStore.vLoading;
export const fServerError = () => cServerStore.vError;
export const fServerStatus = () => cServerStore.vStatus;
export const fServerRole = () => cServerStore.vRole;
export const fDefaultParams = () => cServerStore.mDefaultParams;
export const fContextSize = () => cServerStore.mContextSize;
export const fIsRouterMode = () => cServerStore.mIsRouterMode;
export const fIsModelMode = () => cServerStore.mIsModelMode;
