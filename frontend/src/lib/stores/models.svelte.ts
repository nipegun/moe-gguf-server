import { fTranslate } from '../i18n';
import { base } from '$app/paths';
import { SvelteMap, SvelteSet } from 'svelte/reactivity';
import { toast } from 'svelte-sonner';
import { ServerModelStatus, ServerModelsSseEventType, ModelModality } from '$lib/enums';
import { ModelsService } from '$lib/services/models.service';
import { PropsService } from '$lib/services/props.service';
import { cServerStore, fIsRouterMode } from '$lib/stores/server.svelte';
import {
  fDetectThinkingSupport,
  fDetectThinkingSupportWithReason
} from '$lib/utils/chat-template-thinking-detector';
import { TTLCache } from '$lib/utils/cache-ttl';
import { fGetAuthHeaders } from '$lib/utils/api-headers';
import {
  cModelPropsCacheTtlMs,
  cModelPropsCacheMaxEntries,
  dApiModels,
  cSseRecordSeparator,
  cSseLineSeparator,
  cSseDataPrefix
} from '$lib/constants';

import { cConversationsStore } from '$lib/stores/conversations.svelte';
import { cUiSettingsStore } from '$lib/stores/ui-settings.svelte';

/**
 * modelsStore - Reactive store for model management in both MODEL and ROUTER modes.
 *
 * **Architecture & Relationships:**
 * - **ModelsService**: Stateless service for model API communication
 * - **PropsService**: Stateless service for props/modalities fetching
 * - **modelsStore** (this class): Reactive store for model state
 * - **conversationsStore**: Tracks which conversations use which models
 *
 * **API Inconsistency Workaround:**
 * In MODEL mode, `/props` returns modalities for the single model.
 * In ROUTER mode, `/props` has no modalities — must use `/props?model=<id>` per model.
 * This store normalizes this behavior so consumers don't need to know the server mode.
 */
class ModelsStore {
  /**
   *
   *
   * State
   *
   *
   */

  vModels = $state<ModelOption[]>([]);
  vRouterModels = $state<ApiModelDataEntry[]>([]);
  vLoading = $state(false);
  vUpdating = $state(false);
  vError = $state<string | null>(null);
  vSelectedModelId = $state<string | null>(null);
  vSelectedModelName = $state<string | null>(null);

  // Dedup concurrent fetch() callers — all awaiters share the same inflight promise.
  // Without this, ?model=<name> URL handler races an in-progress fetch and sees an empty list.
  private vInflightFetch: Promise<void> | null = null;

  private vModelUsage = $state<Map<string, SvelteSet<string>>>(new Map());
  private vModelLoadingStates = new SvelteMap<string, boolean>();

  // /models/sse feed state, the single source of truth for status and load progress
  private vStatusAbort: AbortController | null = null;
  private vStatusReaderActive = false;
  private vLoadProgress = new SvelteMap<string, ModelLoadProgress>();
  private dStatusWaiters = new Map<
    string,
    { target: ServerModelStatus; resolve: () => void; reject: (pE: Error) => void }
  >();

  /** Favorite models, kept in the server's settings document (section "favoriteModels") */
  vFavoriteModelIds = $state<Set<string>>(new SvelteSet());

  constructor() {
    cUiSettingsStore.mRegister('favoriteModels', (pSaved) => this.mApplySavedFavorites(pSaved));
  }

  /**
   * Model-specific props cache with TTL.
   * Key: modelId, Value: props data including modalities.
   * TTL: 10 minutes — props don't change frequently.
   */
  private vModelPropsCache = new TTLCache<string, ApiMgsServerProps>({
    ttlMs: cModelPropsCacheTtlMs,
    maxEntries: cModelPropsCacheMaxEntries
  });
  private vModelPropsFetching = $state<Set<string>>(new Set());

  /**
   * Version counter for props cache — used to trigger reactivity when props are updated.
   */
  vPropsCacheVersion = $state(0);

  /**
   *
   *
   * Computed Getters
   *
   *
   */

  get mSelectedModel(): ModelOption | null {
    if (!this.vSelectedModelId) return null;
    return this.vModels.find((pM) => pM.id === this.vSelectedModelId) ?? null;
  }

  get mLoadedModelIds(): string[] {
    return this.vRouterModels
      .filter(
        (pM) =>
          pM.status.value === ServerModelStatus.LOADED ||
          pM.status.value === ServerModelStatus.SLEEPING
      )
      .map((pM) => pM.id);
  }

  get mLoadingModelIds(): string[] {
    return Array.from(this.vModelLoadingStates.entries())
      .filter(([, pLoading]) => pLoading)
      .map(([pId]) => pId);
  }

  /**
   * Get model name in MODEL mode (single model).
   * Extracts from model_path or model_alias from server props.
   * In ROUTER mode, returns null (model is per-conversation).
   */
  get mSingleModelName(): string | null {
    if (fIsRouterMode()) return null;

    const cProps = cServerStore.vProps;
    if (cProps?.model_alias) return cProps.model_alias;
    if (!cProps?.model_path) return null;

    return cProps.model_path.split(/(\\|\/)/).pop() || null;
  }

  get mSelectedModelContextSize(): number | null {
    if (!this.vSelectedModelName) return null;
    return this.mGetModelContextSize(this.vSelectedModelName);
  }

  /**
   *
   *
   * Modalities
   *
   *
   */

  mGetModelModalities(pModelId: string): ModelModalities | null {
    if (!fIsRouterMode() && cServerStore.vProps?.modalities) {
      return this.mBuildModalities(cServerStore.vProps.modalities);
    }

    const cModel = this.vModels.find((pM) => pM.model === pModelId || pM.id === pModelId);
    if (cModel?.modalities) {
      return cModel.modalities;
    }

    const cProps = this.vModelPropsCache.mGet(pModelId);
    if (cProps?.modalities) {
      return this.mBuildModalities(cProps.modalities);
    }

    return null;
  }

  mModelSupportsVision(pModelId: string): boolean {
    return this.mGetModelModalities(pModelId)?.vision ?? false;
  }

  mModelSupportsAudio(pModelId: string): boolean {
    return this.mGetModelModalities(pModelId)?.audio ?? false;
  }

  mModelSupportsVideo(pModelId: string): boolean {
    return this.mGetModelModalities(pModelId)?.video ?? false;
  }

  mGetModelModalitiesArray(pModelId: string): ModelModality[] {
    const cModalities = this.mGetModelModalities(pModelId);
    if (!cModalities) return [];

    const aResult: ModelModality[] = [];
    if (cModalities.vision) aResult.push(ModelModality.VISION);
    if (cModalities.audio) aResult.push(ModelModality.AUDIO);
    if (cModalities.video) aResult.push(ModelModality.VIDEO);

    return aResult;
  }

  mGetModelProps(pModelId: string): ApiMgsServerProps | null {
    return this.vModelPropsCache.mGet(pModelId);
  }

  mGetModelContextSize(pModelId: string): number | null {
    const cProps = this.mGetModelProps(pModelId);
    const cNCtx = cProps?.default_generation_settings?.n_ctx;

    return typeof cNCtx === 'number' ? cNCtx : null;
  }

  mIsModelPropsFetching(pModelId: string): boolean {
    return this.vModelPropsFetching.has(pModelId);
  }

  /**
   *
   *
   * Status Queries
   *
   *
   */

  mIsModelLoaded(pModelId: string): boolean {
    const cModel = this.vRouterModels.find((pM) => pM.id === pModelId);

    return (
      cModel?.status.value === ServerModelStatus.LOADED ||
      cModel?.status.value === ServerModelStatus.SLEEPING
    );
  }

  mIsModelOperationInProgress(pModelId: string): boolean {
    return this.vModelLoadingStates.get(pModelId) ?? false;
  }

  mGetModelStatus(pModelId: string): ServerModelStatus | null {
    const cModel = this.vRouterModels.find((pM) => pM.id === pModelId);

    return cModel?.status.value ?? null;
  }

  mGetModelUsage(pModelId: string): SvelteSet<string> {
    return this.vModelUsage.get(pModelId) ?? new SvelteSet<string>();
  }

  mIsModelInUse(pModelId: string): boolean {
    const cUsage = this.vModelUsage.get(pModelId);

    return cUsage !== undefined && cUsage.size > 0;
  }
  //
  // Thinking Support Detection
  //

  /**
   * Whether the selected model's chat template supports thinking/reasoning.
   * Uses heuristic detection on the model's chat_template from /props.
   *
   * - MODEL mode: the global /props already describes the single loaded model,
   *   so its chat_template is used directly and no per-model cache is involved
   * - ROUTER mode: fetches /props?model=<id> for the selected model (cached),
   *   triggering an async fetch if not yet cached
   */
  get mSupportsThinking(): boolean {
    if (!fIsRouterMode()) {
      return fDetectThinkingSupport(cServerStore.vProps?.chat_template ?? '');
    }

    const cModelId = this.vSelectedModelName;
    if (!cModelId) return false;

    if (!this.vModelPropsCache.mGet(cModelId)) {
      this.mFetchModelProps(cModelId);
    }
    const cProps = this.mGetModelProps(cModelId);
    return fDetectThinkingSupport(cProps?.chat_template ?? '');
  }

  /**
   * Check if a specific model supports thinking.
   * In MODEL mode the global /props describes the single loaded model.
   * In ROUTER mode, fetches model props if not cached.
   */
  mCheckModelSupportsThinking(pModelId: string): boolean {
    if (!fIsRouterMode()) {
      return fDetectThinkingSupport(cServerStore.vProps?.chat_template ?? '');
    }

    if (!pModelId) return false;

    if (!this.vModelPropsCache.mGet(pModelId)) {
      this.mFetchModelProps(pModelId);
    }

    const cProps = this.mGetModelProps(pModelId);
    return fDetectThinkingSupport(cProps?.chat_template ?? '');
  }

  /**
   * Detailed thinking support detection result with reason for debugging/UI.
   */
  get mThinkingSupportDetails(): { supported: boolean; reason: string } {
    if (!fIsRouterMode()) {
      return fDetectThinkingSupportWithReason(cServerStore.vProps?.chat_template ?? '');
    }

    const cModelId = this.vSelectedModelName;
    if (!cModelId) {
      return { supported: false, reason: fTranslate('text_f0f9cb152ba7') };
    }

    if (!this.vModelPropsCache.mGet(cModelId)) {
      this.mFetchModelProps(cModelId);
    }
    const cProps = this.mGetModelProps(cModelId);
    return fDetectThinkingSupportWithReason(cProps?.chat_template ?? '');
  }

  /**
   *
   *
   * Data Fetching
   *
   *
   */

  /**
   * Fetch list of models from server and detect server role.
   * Also fetches modalities for MODEL mode (single model).
   */
  async mFetch(pForce = false): Promise<void> {
    if (this.vInflightFetch) return this.vInflightFetch;
    if (this.vModels.length > 0 && !pForce) return;

    this.vInflightFetch = this.mRunFetch();
    try {
      await this.vInflightFetch;
    } finally {
      this.vInflightFetch = null;
    }
  }

  private async mRunFetch(): Promise<void> {
    this.vLoading = true;
    this.vError = null;

    try {
      if (!cServerStore.vProps) {
        await cServerStore.mFetch();
      }

      const cRouter = fIsRouterMode();

      if (cRouter) {
        const cResponse = await ModelsService.mListRouter();

        this.vRouterModels = cResponse.data;
        this.vModels = this.mBuildModelOptions(cResponse);

        await this.mFetchModalitiesForLoadedModels();

        const cVisible = this.mGetVisibleModels();

        if (cVisible.length === 1 && this.mIsModelLoaded(cVisible[0].model)) {
          this.mSelectModelById(cVisible[0].id);
        }
      } else {
        this.vModels = await this.mFetchModelModeInternal();
      }
    } catch (vError) {
      this.vModels = [];
      this.vError = vError instanceof Error ? vError.message : fTranslate('text_0af848b28294');

      throw vError;
    } finally {
      this.vLoading = false;
    }
  }

  /** Fetch models in MODEL mode (single model, standard OpenAI-compatible). */
  private async mFetchModelModeInternal(): Promise<ModelOption[]> {
    const cResponse = await ModelsService.mList();

    return this.mBuildModelOptions(cResponse);
  }

  /**
   * Build ModelOption[] from an API response.
   * Both MODEL and ROUTER modes share the same mapping logic;
   * they differ only in which endpoint is called.
   */
  private mBuildModelOptions(
    pResponse: ApiModelListResponse | ApiRouterModelsListResponse
  ): ModelOption[] {
    return pResponse.data.map((pItem: ApiModelDataEntry, pIndex: number) => {
      const cDetails = pResponse.models?.[pIndex];
      const cRawCapabilities = Array.isArray(cDetails?.capabilities) ? cDetails?.capabilities : [];
      const cDisplayNameSource =
        cDetails?.name && cDetails.name.trim().length > 0 ? cDetails.name : pItem.id;
      const cModelId = cDetails?.model || pItem.id;

      return {
        id: pItem.id,
        name: this.mToDisplayName(cDisplayNameSource),
        model: cModelId,
        description: cDetails?.description,
        capabilities: cRawCapabilities.filter((pValue: unknown): pValue is string =>
          Boolean(pValue)
        ),
        details: cDetails?.details,
        meta: pItem.meta ?? null,
        parsedId: ModelsService.mParseModelId(cModelId),
        aliases: pItem.aliases ?? [],
        tags: pItem.tags ?? []
      };
    });
  }

  /**
   * Fetch router models with full metadata (ROUTER mode only).
   * No-op outside router mode. Called on app load, when the model dropdown opens
   * and after a model loads or unloads, to refresh the status of every model.
   */
  async mFetchRouterModels(): Promise<void> {
    if (!fIsRouterMode()) return;

    try {
      const cResponse = await ModelsService.mListRouter();
      this.vRouterModels = cResponse.data;
      await this.mFetchModalitiesForLoadedModels();

      const cVisible = this.mGetVisibleModels();
      if (cVisible.length === 1 && this.mIsModelLoaded(cVisible[0].model)) {
        this.mSelectModelById(cVisible[0].id);
      }
    } catch (vError) {
      console.warn('Failed to fetch router models:', vError);
      this.vRouterModels = [];
    }
  }

  /**
   * Fetch props for a specific model from /props endpoint.
   * Uses caching to avoid redundant requests.
   *
   * In ROUTER mode, this only fetches props if the model is loaded,
   * since unloaded models return 400 from /props endpoint.
   *
   * @param pModelId - Model identifier to fetch props for
   * @returns Props data or null if fetch failed or model not loaded
   */
  async mFetchModelProps(pModelId: string): Promise<ApiMgsServerProps | null> {
    const cCached = this.vModelPropsCache.mGet(pModelId);
    if (cCached) return cCached;

    if (cServerStore.mIsRouterMode && !this.mIsModelLoaded(pModelId)) {
      return null;
    }

    if (this.vModelPropsFetching.has(pModelId)) return null;

    this.vModelPropsFetching.add(pModelId);

    try {
      const cProps = await PropsService.mFetchForModel(pModelId);
      this.vModelPropsCache.mSet(pModelId, cProps);
      this.vPropsCacheVersion++;
      return cProps;
    } catch (vError) {
      console.warn(`Failed to fetch props for model ${pModelId}:`, vError);
      return null;
    } finally {
      this.vModelPropsFetching.delete(pModelId);
    }
  }

  /** Fetch modalities for all loaded models from /props endpoint. */
  async mFetchModalitiesForLoadedModels(): Promise<void> {
    const cLoadedModelIds = this.mLoadedModelIds;
    if (cLoadedModelIds.length === 0) return;

    const cPropsPromises = cLoadedModelIds.map((pModelId) => this.mFetchModelProps(pModelId));

    try {
      const cResults = await Promise.all(cPropsPromises);

      this.vModels = this.vModels.map((pModel) => {
        const cModelIndex = cLoadedModelIds.indexOf(pModel.model);
        if (cModelIndex === -1) return pModel;

        const cProps = cResults[cModelIndex];
        if (!cProps?.modalities) return pModel;

        return { ...pModel, modalities: this.mBuildModalities(cProps.modalities) };
      });

      this.vPropsCacheVersion++;
    } catch (vError) {
      console.warn('Failed to fetch modalities for loaded models:', vError);
    }
  }

  /**
   * Update modalities for a specific model.
   * Called when a model is loaded or when we need fresh modality data.
   */
  async mUpdateModelModalities(pModelId: string): Promise<void> {
    const cProps = await this.mFetchModelProps(pModelId);
    if (!cProps?.modalities) return;

    this.vModels = this.vModels.map((pModel) =>
      pModel.model === pModelId
        ? { ...pModel, modalities: this.mBuildModalities(cProps.modalities!) }
        : pModel
    );

    this.vPropsCacheVersion++;
  }

  /**
   * Filter to models visible in the UI (ui !== false).
   */
  private mGetVisibleModels(): ModelOption[] {
    return this.vModels.filter((pOption) => this.mGetModelProps(pOption.model)?.ui !== false);
  }

  /**
   * Gets the model name from the last assistant message in the active conversation.
   * Used by both the chat page and settings page to maintain model consistency.
   */
  mGetModelFromLastAssistantResponse(): string | null {
    const cMessages = cConversationsStore.vActiveMessages;
    if (!cMessages || cMessages.length === 0) return null;

    for (let vI = cMessages.length - 1; vI >= 0; vI--) {
      if (cMessages[vI].model) {
        return cMessages[vI].model;
      }
    }

    return null;
  }

  /**
   * Auto-selects the model from the last assistant response if available and loaded.
   * Returns true if a model was selected, false otherwise.
   */
  async mSelectModelFromLastAssistantResponse(): Promise<boolean> {
    const cLastModel = this.mGetModelFromLastAssistantResponse();
    if (!cLastModel || this.vSelectedModelName === cLastModel) return false;

    const cMatchingModel = this.vModels.find((pOption) => pOption.model === cLastModel);
    if (!cMatchingModel || !this.mIsModelLoaded(cLastModel)) return false;

    try {
      await this.mSelectModelById(cMatchingModel.id);
      console.log(`[modelsStore] Automatically selected model: ${cLastModel} from last message`);
      return true;
    } catch (vError) {
      console.warn('[modelsStore] Failed to automatically select model from last message:', vError);
      return false;
    }
  }

  /**
   * Auto-selects the first available model if none is selected.
   * Prioritizes:
   * 1. Model from active conversation's last assistant response (if loaded)
   * 2. Model from active conversation's last assistant response (if not loaded)
   * 3. First loaded model (not from active conversation)
   * 4. A favorite model
   * 5. First available model
   */
  async mEnsureFirstModelSelected(): Promise<void> {
    if (this.vSelectedModelName) return;

    const cAvailableModels = this.mGetVisibleModels();
    if (cAvailableModels.length === 0) return;

    // Try to select model from last assistant response first
    const cLastModel = this.mGetModelFromLastAssistantResponse();
    if (cLastModel) {
      const cLastModelOption = cAvailableModels.find((pM) => pM.model === cLastModel);
      if (cLastModelOption) {
        await this.mSelectModelById(cLastModelOption.id);
        if (this.mIsModelLoaded(cLastModel)) {
          await this.mFetchModelProps(cLastModel);
        }
        return;
      }
    }

    // Try a loaded model first
    const cLoadedModel = cAvailableModels.find((pM) => this.mIsModelLoaded(pM.model));
    if (cLoadedModel) {
      await this.mSelectModelById(cLoadedModel.id);
      await this.mFetchModelProps(cLoadedModel.model);
      return;
    }

    // Try loading a favorite model
    const cFavorite = this.vFavoriteModelIds.values().next()?.value;
    if (cFavorite) {
      await this.mSelectModelById(cFavorite);
      return;
    }

    // Fall back to the first available model
    await this.mSelectModelById(cAvailableModels[0].id);
  }

  /**
   *
   *
   * Model Selection
   *
   *
   */

  async mSelectModelById(pModelId: string): Promise<void> {
    if (!pModelId || this.vUpdating) return;
    if (this.vSelectedModelId === pModelId) return;

    const cOption = this.vModels.find((pModel) => pModel.id === pModelId);
    if (!cOption) throw new Error(fTranslate('text_1861f766e6f3'));

    this.vUpdating = true;
    this.vError = null;

    try {
      this.vSelectedModelId = cOption.id;
      this.vSelectedModelName = cOption.model;
    } finally {
      this.vUpdating = false;
    }
  }

  /**
   * Select a model by its model name (used for syncing with conversation model).
   */
  mSelectModelByName(pModelName: string): void {
    const cOption = this.vModels.find((pModel) => pModel.model === pModelName);
    if (cOption) {
      this.vSelectedModelId = cOption.id;
      this.vSelectedModelName = cOption.model;
    }
  }

  mClearSelection(): void {
    this.vSelectedModelId = null;
    this.vSelectedModelName = null;
  }

  mFindModelByName(pModelName: string): ModelOption | null {
    return (
      this.vModels.find(
        (pModel) =>
          pModel.model === pModelName ||
          pModel.id === pModelName ||
          pModel.aliases?.includes(pModelName)
      ) ?? null
    );
  }

  mFindModelById(pModelId: string): ModelOption | null {
    return this.vModels.find((pModel) => pModel.id === pModelId) ?? null;
  }

  mHasModel(pModelName: string): boolean {
    return this.vModels.some((pModel) => pModel.model === pModelName);
  }

  /**
   *
   *
   * Loading / Unloading Models
   *
   *
   */

  // reconnect delay after the feed drops or the server is not ready yet
  private static readonly vSseReconnectMs = 1000;

  /**
   * Open the /models/sse feed and keep it live with auto reconnect.
   * Idempotent and router mode only. The feed drives status and progress,
   * so it replaces any post-operation polling.
   */
  mSubscribeStatus(): void {
    if (this.vStatusReaderActive) return;
    if (!fIsRouterMode()) return;

    this.vStatusReaderActive = true;
    this.vStatusAbort = new AbortController();
    void this.mRunStatusReader(this.vStatusAbort.signal);
  }

  /**
   * Close the /models/sse feed and drop transient progress.
   */
  mUnsubscribeStatus(): void {
    this.vStatusReaderActive = false;
    this.vStatusAbort?.abort();
    this.vStatusAbort = null;
    this.vLoadProgress.clear();
  }

  /**
   * Current load progress for a model, or null when not loading.
   */
  mGetLoadProgress(pModelId: string): ModelLoadProgress | null {
    return this.vLoadProgress.get(pModelId) ?? null;
  }

  /**
   * Read the feed and reconnect until unsubscribed. Splits the byte stream
   * into SSE records on the blank line boundary.
   */
  private async mRunStatusReader(pSignal: AbortSignal): Promise<void> {
    const cDecoder = new TextDecoder();

    while (!pSignal.aborted) {
      try {
        const cResponse = await fetch(`${base}${dApiModels.SSE}`, {
          headers: fGetAuthHeaders(),
          signal: pSignal
        });

        if (cResponse.ok && cResponse.body) {
          const cReader = cResponse.body.getReader();
          let vBuffer = '';

          while (!pSignal.aborted) {
            const { value: cValue, done: cDone } = await cReader.read();
            if (cDone) break;

            vBuffer += cDecoder.decode(cValue, { stream: true });

            let vBoundary = vBuffer.indexOf(cSseRecordSeparator);
            while (vBoundary !== -1) {
              this.mHandleStatusRecord(vBuffer.slice(0, vBoundary));
              vBuffer = vBuffer.slice(vBoundary + cSseRecordSeparator.length);
              vBoundary = vBuffer.indexOf(cSseRecordSeparator);
            }
          }
        }
      } catch {
        // network drop or abort falls through to the reconnect delay
      }

      if (pSignal.aborted) return;

      await new Promise((pResolve) => setTimeout(pResolve, ModelsStore.vSseReconnectMs));
    }
  }

  /**
   * Parse one SSE record. The payload rides in the data lines as a JSON
   * envelope that carries its own model, event and data fields.
   */
  private mHandleStatusRecord(pRecord: string): void {
    const cPayload = pRecord
      .split(cSseLineSeparator)
      .filter((pLine) => pLine.startsWith(cSseDataPrefix))
      .map((pLine) => pLine.slice(cSseDataPrefix.length).trim())
      .join(cSseLineSeparator);

    if (cPayload.length === 0) return;

    let vEnvelope: ApiModelsSseEvent;
    try {
      vEnvelope = JSON.parse(cPayload);
    } catch {
      return;
    }

    this.mApplyStatusEvent(vEnvelope);
  }

  /**
   * Route one feed record by event kind. Only the status_* events carry a
   * status payload, models_reload triggers a list refresh, model_remove drops
   * the row, download_* belong to the download surface, not here.
   */
  private mApplyStatusEvent(pEvent: ApiModelsSseEvent): void {
    switch (pEvent.event) {
      case ServerModelsSseEventType.STATUS_CHANGE:
      case ServerModelsSseEventType.MODEL_STATUS:
      case ServerModelsSseEventType.STATUS_UPDATE:
        this.mApplyModelStatus(pEvent);
        break;
      case ServerModelsSseEventType.MODELS_RELOAD:
        void this.mFetchRouterModels();
        break;
      case ServerModelsSseEventType.MODEL_REMOVE:
        this.mRemoveRouterModel(pEvent.model);
        break;
      case ServerModelsSseEventType.DOWNLOAD_PROGRESS:
        break;
    }
  }

  /**
   * Apply a status envelope: update the model row, track or clear progress,
   * settle any pending load or unload awaiter.
   */
  private mApplyModelStatus(pEvent: ApiModelsSseEvent): void {
    const cModel = pEvent.model;
    const cData = pEvent.data;
    if (!cModel || !cData?.status) return;

    const cStatus = cData.status;

    this.mSetRouterModelStatus(cModel, cStatus);

    if (cStatus === ServerModelStatus.LOADING) {
      if (cData.progress) this.vLoadProgress.set(cModel, cData.progress);
    } else {
      this.vLoadProgress.delete(cModel);
    }

    if (cStatus === ServerModelStatus.LOADED) {
      void this.mUpdateModelModalities(cModel);
    }

    const cFailed =
      cStatus === ServerModelStatus.FAILED ||
      (cStatus === ServerModelStatus.UNLOADED && (cData.exit_code ?? 0) !== 0);

    if (cFailed) {
      this.mRejectStatus(
        cModel,
        new Error(fTranslate('text_99a8e7331df4', { p0: this.mToDisplayName(cModel) }))
      );
      return;
    }

    this.mSettleStatus(cModel, cStatus);
  }

  /**
   * Drop a model row reported gone by the feed and settle its awaiters.
   */
  private mRemoveRouterModel(pModelId: string): void {
    if (this.vRouterModels.findIndex((pM) => pM.id === pModelId) === -1) return;

    this.vRouterModels = this.vRouterModels.filter((pM) => pM.id !== pModelId);
    this.vLoadProgress.delete(pModelId);
    this.mRejectStatus(
      pModelId,
      new Error(fTranslate('text_f63ebdd07a74', { p0: this.mToDisplayName(pModelId) }))
    );
  }

  /**
   * Update one model row status in place, reassigning to trigger reactivity.
   */
  private mSetRouterModelStatus(pModelId: string, pStatus: ServerModelStatus): void {
    const cIdx = this.vRouterModels.findIndex((pM) => pM.id === pModelId);
    if (cIdx === -1) return;

    const cCurrent = this.vRouterModels[cIdx];
    if (cCurrent.status.value === pStatus) return;

    const aNext = [...this.vRouterModels];
    aNext[cIdx] = { ...cCurrent, status: { ...cCurrent.status, value: pStatus } };
    this.vRouterModels = aNext;
  }

  /**
   * Register an awaiter that resolves when the feed reports target status.
   * One operation runs per model at a time, so one awaiter per model is kept.
   */
  private mWaitForStatus(pModelId: string, pTarget: ServerModelStatus): Promise<void> {
    return new Promise((pResolve, pReject) => {
      this.dStatusWaiters.set(pModelId, { target: pTarget, resolve: pResolve, reject: pReject });
    });
  }

  /**
   * Resolve and drop the awaiter when the model reaches its target status.
   */
  private mSettleStatus(pModelId: string, pStatus: ServerModelStatus): void {
    const cWaiter = this.dStatusWaiters.get(pModelId);
    if (cWaiter && cWaiter.target === pStatus) {
      this.dStatusWaiters.delete(pModelId);
      cWaiter.resolve();
    }
  }

  /**
   * Reject and drop the awaiter for a model.
   */
  private mRejectStatus(pModelId: string, pError: Error): void {
    const cWaiter = this.dStatusWaiters.get(pModelId);
    if (cWaiter) {
      this.dStatusWaiters.delete(pModelId);
      cWaiter.reject(pError);
    }
  }

  async mLoadModel(pModelId: string): Promise<void> {
    if (this.mIsModelLoaded(pModelId)) return;
    if (this.vModelLoadingStates.get(pModelId)) return;

    this.vModelLoadingStates.set(pModelId, true);
    this.vError = null;

    // the feed drives completion, so it must be live before the request
    this.mSubscribeStatus();

    const cReachedLoaded = this.mWaitForStatus(pModelId, ServerModelStatus.LOADED);
    cReachedLoaded.catch(() => {});

    try {
      await ModelsService.mLoad(pModelId);
      await cReachedLoaded;
      toast.success(fTranslate('text_1815f7227058', { p0: this.mToDisplayName(pModelId) }));
    } catch (vError) {
      this.mRejectStatus(
        pModelId,
        vError instanceof Error ? vError : new Error(fTranslate('text_f318eb3d47fb'))
      );
      this.vError = vError instanceof Error ? vError.message : fTranslate('text_223544360f39');
      toast.error(fTranslate('text_afd9701b0c96', { p0: this.mToDisplayName(pModelId) }));
      throw vError;
    } finally {
      this.vModelLoadingStates.set(pModelId, false);
    }
  }

  async mUnloadModel(pModelId: string): Promise<void> {
    if (!this.mIsModelLoaded(pModelId)) return;
    if (this.vModelLoadingStates.get(pModelId)) return;

    this.vModelLoadingStates.set(pModelId, true);
    this.vError = null;

    this.mSubscribeStatus();

    const cReachedUnloaded = this.mWaitForStatus(pModelId, ServerModelStatus.UNLOADED);
    cReachedUnloaded.catch(() => {});

    try {
      await ModelsService.mUnload(pModelId);
      await cReachedUnloaded;
      toast.info(fTranslate('text_c1cb60e75609', { p0: this.mToDisplayName(pModelId) }));
    } catch (vError) {
      this.mRejectStatus(
        pModelId,
        vError instanceof Error ? vError : new Error(fTranslate('text_2f7941bef771'))
      );
      this.vError = vError instanceof Error ? vError.message : fTranslate('text_b48a72b4a9af');
      toast.error(fTranslate('text_0bb18649a652', { p0: this.mToDisplayName(pModelId) }));
      throw vError;
    } finally {
      this.vModelLoadingStates.set(pModelId, false);
    }
  }

  async mEnsureModelLoaded(pModelId: string): Promise<void> {
    if (this.mIsModelLoaded(pModelId)) return;
    await this.mLoadModel(pModelId);
  }

  /**
   *
   *
   * Favorites
   *
   *
   */

  mIsFavorite(pModelId: string): boolean {
    return this.vFavoriteModelIds.has(pModelId);
  }

  mToggleFavorite(pModelId: string): void {
    const cNext = new SvelteSet(this.vFavoriteModelIds);

    if (cNext.has(pModelId)) {
      cNext.delete(pModelId);
    } else {
      cNext.add(pModelId);
    }

    this.vFavoriteModelIds = cNext;

    cUiSettingsStore.mSet('favoriteModels', [...cNext]);
  }

  private mApplySavedFavorites(pSaved: string[] | undefined): void {
    this.vFavoriteModelIds = new SvelteSet(
      Array.isArray(pSaved) ? pSaved.filter((pId) => typeof pId === 'string') : []
    );
  }

  /**
   *
   *
   * Utilities
   *
   *
   */

  private mToDisplayName(pId: string): string {
    const cSegments = pId.split(/\\|\//);
    const cCandidate = cSegments.pop();
    return cCandidate && cCandidate.trim().length > 0 ? cCandidate : pId;
  }

  private mBuildModalities(
    pModalities: NonNullable<ApiMgsServerProps['modalities']>
  ): ModelModalities {
    return {
      vision: pModalities.vision ?? false,
      audio: pModalities.audio ?? false,
      video: pModalities.video ?? false
    };
  }

  mClear(): void {
    this.mUnsubscribeStatus();
    this.dStatusWaiters.forEach((pWaiter) =>
      pWaiter.reject(new Error(fTranslate('text_d00d27c26586')))
    );
    this.dStatusWaiters.clear();
    this.vModels = [];
    this.vRouterModels = [];
    this.vLoading = false;
    this.vUpdating = false;
    this.vError = null;
    this.vSelectedModelId = null;
    this.vSelectedModelName = null;
    this.vModelUsage.clear();
    this.vModelLoadingStates.clear();
    this.vModelPropsCache.mClear();
    this.vModelPropsFetching.clear();
  }

  /**
   * Prune expired entries from caches.
   * Call periodically for proactive memory cleanup.
   */
  mPruneExpiredCache(): number {
    return this.vModelPropsCache.mPrune();
  }
}

export const cModelsStore = new ModelsStore();

export const fModelOptions = () => cModelsStore.vModels;
export const fRouterModels = () => cModelsStore.vRouterModels;
export const fModelsLoading = () => cModelsStore.vLoading;
export const fModelsUpdating = () => cModelsStore.vUpdating;
export const fModelsError = () => cModelsStore.vError;
export const fSelectedModelId = () => cModelsStore.vSelectedModelId;
export const fSelectedModelName = () => cModelsStore.vSelectedModelName;
export const fSelectedModelOption = () => cModelsStore.mSelectedModel;
export const fLoadedModelIds = () => cModelsStore.mLoadedModelIds;
export const fLoadingModelIds = () => cModelsStore.mLoadingModelIds;
export const fPropsCacheVersion = () => cModelsStore.vPropsCacheVersion;
export const fSingleModelName = () => cModelsStore.mSingleModelName;
export const fSelectedModelContextSize = () => cModelsStore.mSelectedModelContextSize;
export const fFavoriteModelIds = () => cModelsStore.vFavoriteModelIds;
export const fSupportsThinking = () => cModelsStore.mSupportsThinking;
export const fCheckModelSupportsThinking = (pModelId: string) =>
  cModelsStore.mCheckModelSupportsThinking(pModelId);
export const fThinkingSupportDetails = () => cModelsStore.mThinkingSupportDetails;
