/**
 * Reactive state for the context usage gauge: resolves the active model,
 * fetches its cached props, parses live server stats, and exposes per-turn
 * read / fresh / cache / output and cumulative token counts.
 */

import {
  cModelsStore,
  fModelOptions,
  fSelectedModelId,
  fSingleModelName
} from '$lib/stores/models.svelte';
import { cChatStore } from '$lib/stores/chat.svelte';
import { fActiveMessages } from '$lib/stores/conversations.svelte';
import { fIsRouterMode } from '$lib/stores/server.svelte';
import { MessageRole } from '$lib/enums';
import { cStatsUnits } from '$lib/constants';
import type { ChatMessageTimings, DatabaseMessage } from '$lib/types';
import { fUseProcessingState } from './use-processing-state.svelte';
import {
  fColorLevelFromPercent,
  type ColorLevel
} from '$lib/components/app/chat/ChatForm/ChatFormContextGauge/context-gauge';

interface LiveStats {
  freshTokens: number;
  promptTokens: number;
  cacheTokens: number;
  outputTokens: number;
}

export interface UseContextGaugeReturn {
  readonly activeModelId: string | null;
  readonly isActiveModelLoaded: boolean;
  readonly isActiveModelLoading: boolean;
  readonly contextTotal: number | null;
  readonly contextUsed: number;
  readonly currentRead: number;
  readonly currentFresh: number;
  readonly currentCache: number;
  readonly currentOutput: number;
  readonly kvTotal: number;
  readonly cumulativeRead: number;
  readonly cumulativeOutput: number;
  readonly cumulativeCacheTotal: number;
  readonly averageTokensPerSecond: number | null;
  readonly contextPercent: number | null;
  readonly colorLevel: ColorLevel;
  readonly transientDetails: string[];
  readonly hasAnyUsage: boolean;
  loadModel(): Promise<void>;
  startMonitoring(): void;
}

function fLastAssistantTimings(pMessages: DatabaseMessage[]): ChatMessageTimings | undefined {
  for (let vI = pMessages.length - 1; vI >= 0; vI--) {
    const cM = pMessages[vI];
    if (cM.role === MessageRole.ASSISTANT && cM.timings) return cM.timings;
  }
  return undefined;
}

function fDeriveLiveStats(
  pState: ReturnType<typeof fUseProcessingState>['processingState']
): LiveStats | null {
  if (!pState || (pState.status !== 'preparing' && pState.status !== 'generating')) {
    return null;
  }
  const cPromptTokens = pState.promptTokens ?? 0;
  const cCacheTokens = pState.cacheTokens ?? 0;
  return {
    freshTokens: cPromptTokens,
    promptTokens: cPromptTokens + cCacheTokens,
    cacheTokens: cCacheTokens,
    outputTokens: pState.outputTokensUsed ?? 0
  };
}

const aTransientDetailsExcludedPrefixes = ['Context:', 'Output:'];

function fFilterTransientDetails(pRaw: string[]): string[] {
  return pRaw.filter((pDetail) => {
    if (aTransientDetailsExcludedPrefixes.some((pPrefix) => pDetail.startsWith(pPrefix))) {
      return false;
    }
    return !pDetail.includes(cStatsUnits.TOKENS_PER_SECOND);
  });
}

export function fUseContextGauge(): UseContextGaugeReturn {
  const cProcessingState = fUseProcessingState();

  // Resolve the model the gauge reports context for: explicit selection >
  // last assistant model > single-model mode (mirrors useChatScreenActiveModel).
  const cActiveModelId = $derived.by(() => {
    if (!fIsRouterMode()) {
      return fSingleModelName();
    }

    const cSelectedId = fSelectedModelId();
    if (cSelectedId) {
      const cModel = fModelOptions().find((pM) => pM.id === cSelectedId);
      if (cModel) return cModel.model;
    }

    return cChatStore.mGetConversationModel(fActiveMessages() as DatabaseMessage[]);
  });

  const cIsActiveModelLoaded = $derived(
    cActiveModelId !== null && cModelsStore.mIsModelLoaded(cActiveModelId)
  );

  const cIsActiveModelLoading = $derived(
    cActiveModelId !== null && cModelsStore.mIsModelOperationInProgress(cActiveModelId)
  );

  // Pull /props on demand so n_ctx surfaces before the first chat request.
  $effect(() => {
    if (cActiveModelId && cIsActiveModelLoaded) {
      const cCached = cModelsStore.mGetModelProps(cActiveModelId);
      if (!cCached) {
        void cModelsStore.mFetchModelProps(cActiveModelId);
      }
    }
  });

  const cContextTotal = $derived.by(() => {
    void cModelsStore.vPropsCacheVersion;
    return cActiveModelId ? cModelsStore.mGetModelContextSize(cActiveModelId) : null;
  });

  const cLiveStats = $derived(fDeriveLiveStats(cProcessingState.processingState));

  const cCurrentRead = $derived.by(() => {
    const cTimings = fLastAssistantTimings(fActiveMessages() as DatabaseMessage[]);
    let vRead = 0;
    if (cTimings) {
      vRead = (cTimings.prompt_n ?? 0) + (cTimings.cache_n ?? 0);
    }
    // live.promptTokens is already the combined reading (prompt + cache),
    // so do not also add live.cacheTokens.
    if (cLiveStats && cLiveStats.promptTokens > 0) {
      vRead = Math.max(vRead, cLiveStats.promptTokens);
    }
    return vRead;
  });

  const cCurrentFresh = $derived.by(() => {
    const cTimings = fLastAssistantTimings(fActiveMessages() as DatabaseMessage[]);
    const cFresh = cTimings?.prompt_n ?? 0;
    return Math.max(cFresh, cLiveStats?.freshTokens ?? 0);
  });

  const cCurrentCache = $derived.by(() => {
    const cTimings = fLastAssistantTimings(fActiveMessages() as DatabaseMessage[]);
    const cCached = cTimings?.cache_n ?? 0;
    if (cLiveStats && cLiveStats.promptTokens > 0) {
      return Math.max(cCached, cLiveStats.cacheTokens);
    }
    return cCached;
  });

  const cCurrentOutput = $derived.by(() => {
    if (cLiveStats && cLiveStats.outputTokens > 0) return cLiveStats.outputTokens;
    const cTimings = fLastAssistantTimings(fActiveMessages() as DatabaseMessage[]);
    return cTimings?.predicted_n ?? 0;
  });

  const cKvTotal = $derived(cCurrentRead + cCurrentOutput);
  const cContextUsed = $derived(cCurrentRead + cCurrentOutput);

  const cCumulative = $derived.by(() => {
    const cMessages = fActiveMessages() as DatabaseMessage[];

    // Agentic sessions stamp the same agentic.llm totals onto every
    // assistant message; cache_n is never per-turn so cache_total stays 0.
    const cAgenticMessages = cMessages.filter(
      (pM) => pM.role === MessageRole.ASSISTANT && pM.timings?.agentic?.llm?.predicted_n != null
    );

    if (cAgenticMessages.length > 0) {
      const cLlm = cAgenticMessages[cAgenticMessages.length - 1].timings!.agentic!.llm;
      const cOutput = cLlm.predicted_n ?? 0;
      const cOutputMs = cLlm.predicted_ms ?? 0;
      const cAverageTokensPerSecond =
        cOutputMs > 0 && cOutput > 0 ? (cOutput / cOutputMs) * 1000 : null;
      return {
        read: cLlm.prompt_n ?? 0,
        output: cOutput,
        cacheTotal: 0,
        averageTokensPerSecond: cAverageTokensPerSecond
      };
    }

    let vRead = 0;
    let vOutput = 0;
    let vOutputMs = 0;
    let vCacheTotal = 0;
    for (const cM of cMessages) {
      if (cM.role !== MessageRole.ASSISTANT || !cM.timings) continue;
      vRead += cM.timings.prompt_n ?? 0;
      vCacheTotal += cM.timings.cache_n ?? 0;
      vOutput += cM.timings.predicted_n ?? 0;
      vOutputMs += cM.timings.predicted_ms ?? 0;
    }
    const cAverageTokensPerSecond =
      vOutputMs > 0 && vOutput > 0 ? (vOutput / vOutputMs) * 1000 : null;
    return {
      read: vRead,
      output: vOutput,
      cacheTotal: vCacheTotal,
      averageTokensPerSecond: cAverageTokensPerSecond
    };
  });

  const cContextPercent = $derived.by(() => {
    if (cContextTotal === null || cContextTotal <= 0) return null;
    return Math.round((cContextUsed / cContextTotal) * 100);
  });

  const cColorLevel = $derived(fColorLevelFromPercent(cContextPercent));

  // Drop lines the surrounding Context / Output / speed rows already render.
  const cTransientDetails = $derived(
    fFilterTransientDetails(cProcessingState.getTechnicalDetails())
  );

  const cHasAnyUsage = $derived(
    cCumulative.read > 0 ||
      cCumulative.output > 0 ||
      cCurrentRead > 0 ||
      cCurrentOutput > 0 ||
      cCumulative.averageTokensPerSecond !== null ||
      cTransientDetails.length > 0
  );

  async function fLoadModel() {
    if (!cActiveModelId || cIsActiveModelLoading) return;
    try {
      await cModelsStore.mLoadModel(cActiveModelId);
    } catch {
      // toast already surfaced by modelsStore.loadModel
    }
  }

  return {
    get activeModelId() {
      return cActiveModelId;
    },
    get isActiveModelLoaded() {
      return cIsActiveModelLoaded;
    },
    get isActiveModelLoading() {
      return cIsActiveModelLoading;
    },
    get contextTotal() {
      return cContextTotal;
    },
    get contextUsed() {
      return cContextUsed;
    },
    get currentRead() {
      return cCurrentRead;
    },
    get currentFresh() {
      return cCurrentFresh;
    },
    get currentCache() {
      return cCurrentCache;
    },
    get currentOutput() {
      return cCurrentOutput;
    },
    get kvTotal() {
      return cKvTotal;
    },
    get cumulativeRead() {
      return cCumulative.read;
    },
    get cumulativeOutput() {
      return cCumulative.output;
    },
    get cumulativeCacheTotal() {
      return cCumulative.cacheTotal;
    },
    get averageTokensPerSecond() {
      return cCumulative.averageTokensPerSecond;
    },
    get contextPercent() {
      return cContextPercent;
    },
    get colorLevel() {
      return cColorLevel;
    },
    get transientDetails() {
      return cTransientDetails;
    },
    get hasAnyUsage() {
      return cHasAnyUsage;
    },
    loadModel: fLoadModel,
    startMonitoring: () => cProcessingState.startMonitoring()
  };
}
