import { fTranslate } from '../i18n';
import { ReasoningEffort } from '$lib/enums';
import { aReasoningEffortLevels } from '$lib/constants/reasoning-effort';
import { dReasoningEffortTokens } from '$lib/constants/reasoning-effort-tokens';
import type { ReasoningEffortLevel } from '$lib/types';
import type { DatabaseMessage } from '$lib/types/database';
import {
  cModelsStore,
  fCheckModelSupportsThinking,
  fSupportsThinking,
  fPropsCacheVersion,
  fLoadedModelIds
} from '$lib/stores/models.svelte';
import { cChatStore } from '$lib/stores/chat.svelte';
import { cConversationsStore, fActiveMessages } from '$lib/stores/conversations.svelte';
import { fIsRouterMode } from '$lib/stores/server.svelte';

export interface UseReasoningMenuReturn {
  readonly modelSupportsThinking: boolean;
  readonly thinkingEnabled: boolean;
  readonly isOff: boolean;
  readonly currentEffort: ReasoningEffort;
  /** Translated name of the selected level, shown next to the menu entry. */
  readonly currentLabel: string;
  readonly levels: ReasoningEffortLevel[];
  isSelected(pLevel: ReasoningEffortLevel): boolean;
  tokenLabel(pLevel: ReasoningEffortLevel): string | null;
  select(pLevel: ReasoningEffortLevel): void;
}

/**
 * Shared reactive state and helpers for the reasoning effort menu.
 *
 * Used by both the desktop dropdown (`ChatFormActionAddReasoningSubmenu`)
 * and the mobile sheet (`ChatFormActionAddSheet`) to avoid duplicating the
 * thinking-support derivation and the effort selection logic.
 */
export function fUseReasoningMenu(): UseReasoningMenuReturn {
  const cConversationModel = $derived(
    cChatStore.mGetConversationModel(fActiveMessages() as DatabaseMessage[])
  );

  // a router chat can carry reasoning from an earlier turn before the props
  // cache is primed, so a model that already produced thinking still qualifies
  const cModelSupportsThinkingFromMessages = $derived.by(() => {
    const cModelId = fIsRouterMode() ? cModelsStore.vSelectedModelName || cConversationModel : null;
    if (!cModelId) return false;

    return cConversationsStore.vActiveMessages.some(
      (pM) => pM.role === 'assistant' && pM.model === cModelId && !!pM.reasoningContent
    );
  });

  const cModelSupportsThinking = $derived.by(() => {
    fLoadedModelIds();
    fPropsCacheVersion();

    if (fIsRouterMode()) {
      const cModelId = cModelsStore.vSelectedModelName || cConversationModel;
      return fCheckModelSupportsThinking(cModelId ?? '') || cModelSupportsThinkingFromMessages;
    }

    return fSupportsThinking() || cModelSupportsThinkingFromMessages;
  });

  const cCurrentEffort = $derived(cConversationsStore.mGetReasoningEffort());
  const cThinkingEnabled = $derived(
    cCurrentEffort !== ReasoningEffort.OFF && cCurrentEffort !== ReasoningEffort.DEFAULT
  );

  return {
    get modelSupportsThinking() {
      return cModelSupportsThinking;
    },
    get thinkingEnabled() {
      return cThinkingEnabled;
    },
    get isOff() {
      return cCurrentEffort === ReasoningEffort.OFF;
    },
    get currentEffort() {
      return cCurrentEffort;
    },
    get currentLabel() {
      return (
        aReasoningEffortLevels.find((pLevel) => pLevel.value === cCurrentEffort)?.label ??
        cCurrentEffort
      );
    },
    get levels() {
      return aReasoningEffortLevels;
    },
    isSelected(pLevel: ReasoningEffortLevel): boolean {
      return cCurrentEffort === pLevel.value;
    },
    tokenLabel(pLevel: ReasoningEffortLevel): string | null {
      if (pLevel.value === ReasoningEffort.DEFAULT) return fTranslate('text_09a77587d0c7');
      const cTokens = dReasoningEffortTokens[pLevel.value];
      if (cTokens === undefined) return null;
      return cTokens === -1
        ? fTranslate('text_11dde17d6c3e')
        : fTranslate('text_b7efb9e2be13', { p0: cTokens.toLocaleString() });
    },
    select(pLevel: ReasoningEffortLevel): void {
      cConversationsStore.mSetReasoningEffort(pLevel.value as ReasoningEffort);
    }
  };
}
