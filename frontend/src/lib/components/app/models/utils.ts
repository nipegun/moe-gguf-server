import { SvelteMap } from 'svelte/reactivity';
import type { ModelOption } from '$lib/types/models';

export interface ModelItem {
  option: ModelOption;
  flatIndex: number;
}

export interface OrgGroup {
  orgName: string | null;
  items: ModelItem[];
}

export interface GroupedModelOptions {
  loaded: ModelItem[];
  favorites: ModelItem[];
  available: OrgGroup[];
}

export function fFilterModelOptions(pOptions: ModelOption[], pSearchTerm: string): ModelOption[] {
  const cTerm = pSearchTerm.trim().toLowerCase();
  if (!cTerm) return pOptions;

  return pOptions.filter(
    (pOption) =>
      pOption.model.toLowerCase().includes(cTerm) ||
      pOption.name?.toLowerCase().includes(cTerm) ||
      pOption.aliases?.some((pAlias: string) => pAlias.toLowerCase().includes(cTerm)) ||
      pOption.tags?.some((pTag: string) => pTag.toLowerCase().includes(cTerm))
  );
}

export function fGroupModelOptions(
  pFilteredOptions: ModelOption[],
  pFavoriteIds: Set<string>,
  pIsModelLoaded: (pModel: string) => boolean
): GroupedModelOptions {
  // Loaded models
  const aLoaded: ModelItem[] = [];
  for (let vI = 0; vI < pFilteredOptions.length; vI++) {
    if (pIsModelLoaded(pFilteredOptions[vI].model)) {
      aLoaded.push({ option: pFilteredOptions[vI], flatIndex: vI });
    }
  }

  // Favorites (excluding loaded)
  const sLoadedModelIds = new Set(aLoaded.map((pItem) => pItem.option.model));
  const aFavorites: ModelItem[] = [];
  for (let vI = 0; vI < pFilteredOptions.length; vI++) {
    if (
      pFavoriteIds.has(pFilteredOptions[vI].model) &&
      !sLoadedModelIds.has(pFilteredOptions[vI].model)
    ) {
      aFavorites.push({ option: pFilteredOptions[vI], flatIndex: vI });
    }
  }

  // Available models grouped by org (excluding loaded and favorites)
  const aAvailable: OrgGroup[] = [];
  const cOrgGroups = new SvelteMap<string, ModelItem[]>();
  for (let vI = 0; vI < pFilteredOptions.length; vI++) {
    const cOption = pFilteredOptions[vI];
    if (sLoadedModelIds.has(cOption.model) || pFavoriteIds.has(cOption.model)) continue;

    const cKey = cOption.parsedId?.orgName ?? '';
    if (!cOrgGroups.has(cKey)) cOrgGroups.set(cKey, []);
    cOrgGroups.get(cKey)!.push({ option: cOption, flatIndex: vI });
  }

  for (const [cOrgName, cItems] of cOrgGroups) {
    aAvailable.push({ orgName: cOrgName || null, items: cItems });
  }

  return { loaded: aLoaded, favorites: aFavorites, available: aAvailable };
}
