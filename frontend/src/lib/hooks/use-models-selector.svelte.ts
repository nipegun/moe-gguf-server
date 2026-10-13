import { onMount } from 'svelte';
import {
  cModelsStore,
  fModelOptions,
  fModelsLoading,
  fModelsUpdating,
  fSelectedModelId,
  fSingleModelName
} from '$lib/stores/models.svelte';
import { fIsRouterMode } from '$lib/stores/server.svelte';
import { fFilterModelOptions, fGroupModelOptions } from '$lib/components/app/models/utils';
import type { ModelOption } from '$lib/types/models';

export interface UseModelsSelectorOptions {
  currentModel: () => string | null;
  useGlobalSelection?: () => boolean;
  onModelChange?: () =>
    | ((pModelId: string, pModelName: string) => Promise<boolean> | boolean | void)
    | undefined;
  onOpenChange?: (pOpen: boolean) => void;
}

export interface UseModelsSelectorReturn {
  readonly options: ModelOption[];
  readonly loading: boolean;
  readonly updating: boolean;
  readonly activeId: string | null;
  readonly isRouter: boolean;
  readonly serverModel: string | null;
  readonly isHighlightedCurrentModelActive: boolean;
  readonly isCurrentModelInCache: boolean;
  readonly filteredOptions: ModelOption[];
  readonly groupedFilteredOptions: ReturnType<typeof fGroupModelOptions>;
  readonly isLoadingModel: boolean;
  readonly searchTerm: string;
  readonly showModelDialog: boolean;
  readonly infoModelId: string | null;
  setSearchTerm(pValue: string): void;
  setShowModelDialog(pValue: boolean): void;
  handleInfoClick(pModelName: string): void;
  handleSelect(pModelId: string): Promise<void>;
  handleOpenChange(pOpen: boolean): void;
  isFavorite(pModel: string): boolean;
  getDisplayOption(): ModelOption | undefined;
}

/**
 * Shared reactive state and logic for model selection.
 *
 * Used by both the desktop dropdown (`ModelsSelectorDropdown`)
 * and the mobile sheet (`ModelsSelectorSheet`) to avoid
 * duplicating store derivations, selection handling, and model loading.
 */
export function fUseModelsSelector(pOpts: UseModelsSelectorOptions): UseModelsSelectorReturn {
  const cOptions = $derived(
    fModelOptions().filter((pOption) => {
      const cModelProps = cModelsStore.mGetModelProps(pOption.model);

      return cModelProps?.ui !== false;
    })
  );
  const cLoading = $derived(fModelsLoading());
  const cUpdating = $derived(fModelsUpdating());
  const cActiveId = $derived(fSelectedModelId());
  const cIsRouter = $derived(fIsRouterMode());
  const cServerModel = $derived(fSingleModelName());

  const cCurrentModel = $derived(pOpts.currentModel());
  const cOnModelChange = $derived(pOpts.onModelChange?.());

  const cIsHighlightedCurrentModelActive = $derived.by(() => {
    if (!cIsRouter || !cCurrentModel) return false;
    const cCurrentOption = cOptions.find((pOption) => pOption.model === cCurrentModel);
    return cCurrentOption ? cCurrentOption.id === cActiveId : false;
  });

  const cIsCurrentModelInCache = $derived.by(() => {
    if (!cIsRouter || !cCurrentModel) return true;
    return cOptions.some((pOption) => pOption.model === cCurrentModel);
  });

  let vIsLoadingModel = $state(false);
  let vSearchTerm = $state('');
  let vShowModelDialog = $state(false);
  let vInfoModelId = $state<string | null>(null);
  const cFilteredOptions = $derived(fFilterModelOptions(cOptions, vSearchTerm));
  const cGroupedFilteredOptions = $derived(
    fGroupModelOptions(cFilteredOptions, cModelsStore.vFavoriteModelIds, (pM) =>
      cModelsStore.mIsModelLoaded(pM)
    )
  );

  function fHandleInfoClick(pModelName: string) {
    vInfoModelId = pModelName;
    vShowModelDialog = true;
  }

  onMount(() => {
    cModelsStore.mFetch().catch((pError) => {
      console.error('Unable to load models:', pError);
    });
  });

  function fHandleOpenChange(pOpen: boolean) {
    if (cLoading || cUpdating) return;

    if (cIsRouter) {
      vSearchTerm = '';

      if (pOpen) {
        cModelsStore.mFetchRouterModels().then(() => {
          cModelsStore.mFetchModalitiesForLoadedModels();
        });
      }

      pOpts.onOpenChange?.(pOpen);
    } else {
      vShowModelDialog = pOpen;
    }
  }

  async function fHandleSelect(pModelId: string) {
    const cOption = cOptions.find((pOpt) => pOpt.id === pModelId);
    if (!cOption) return;

    let vShouldCloseMenu = true;

    if (cOnModelChange) {
      const cResult = await cOnModelChange(cOption.id, cOption.model);

      if (cResult === false) {
        vShouldCloseMenu = false;
      }
    } else {
      await cModelsStore.mSelectModelById(cOption.id);
    }

    if (vShouldCloseMenu) {
      fHandleOpenChange(false);

      requestAnimationFrame(() => {
        const cTextarea = document.querySelector<HTMLTextAreaElement>(
          '[data-slot="chat-form"] textarea'
        );

        cTextarea?.focus({ preventScroll: true });
      });
    }

    if (!cOnModelChange && cIsRouter && !cModelsStore.mIsModelLoaded(cOption.model)) {
      vIsLoadingModel = true;

      cModelsStore
        .mLoadModel(cOption.model)
        .catch((pError) => console.error('Failed to load model:', pError))
        .finally(() => (vIsLoadingModel = false));
    }
  }

  function fGetDisplayOption(): ModelOption | undefined {
    if (!cIsRouter) {
      const cDisplayModel = cServerModel || cCurrentModel;

      if (cDisplayModel) {
        return {
          id: cServerModel ? 'current' : 'offline-current',
          model: cDisplayModel,
          name: cDisplayModel.split('/').pop() || cDisplayModel,
          capabilities: []
        };
      }

      return undefined;
    }

    if (cCurrentModel) {
      if (!cIsCurrentModelInCache) {
        return {
          id: 'not-in-cache',
          model: cCurrentModel,
          name: cCurrentModel.split('/').pop() || cCurrentModel,
          capabilities: []
        };
      }

      return cOptions.find((pOption) => pOption.model === cCurrentModel);
    }

    if (cActiveId) {
      return cOptions.find((pOption) => pOption.id === cActiveId);
    }

    return undefined;
  }

  return {
    get options() {
      return cOptions;
    },

    get loading() {
      return cLoading;
    },

    get updating() {
      return cUpdating;
    },

    get activeId() {
      return cActiveId;
    },

    get isRouter() {
      return cIsRouter;
    },

    get serverModel() {
      return cServerModel;
    },

    get isHighlightedCurrentModelActive() {
      return cIsHighlightedCurrentModelActive;
    },

    get isCurrentModelInCache() {
      return cIsCurrentModelInCache;
    },

    get filteredOptions() {
      return cFilteredOptions;
    },

    get groupedFilteredOptions() {
      return cGroupedFilteredOptions;
    },

    get isLoadingModel() {
      return vIsLoadingModel;
    },

    get searchTerm() {
      return vSearchTerm;
    },

    get showModelDialog() {
      return vShowModelDialog;
    },

    get infoModelId() {
      return vInfoModelId;
    },

    setSearchTerm(pValue: string) {
      vSearchTerm = pValue;
    },

    setShowModelDialog(pValue: boolean) {
      vShowModelDialog = pValue;
    },

    handleInfoClick: fHandleInfoClick,

    handleSelect: fHandleSelect,

    handleOpenChange: fHandleOpenChange,

    isFavorite(pModel: string) {
      return cModelsStore.vFavoriteModelIds.has(pModel);
    },

    getDisplayOption: fGetDisplayOption
  };
}
