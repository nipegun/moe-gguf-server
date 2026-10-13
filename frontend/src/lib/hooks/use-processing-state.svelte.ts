import { fTranslate } from '../i18n';
import { fActiveProcessingState } from '$lib/stores/chat.svelte';
import { cStatsUnits } from '$lib/constants';
import type { ApiProcessingState, LiveProcessingStats, LiveGenerationStats } from '$lib/types';

export interface UseProcessingStateReturn {
  readonly processingState: ApiProcessingState | null;
  getProcessingDetails(): string[];
  getTechnicalDetails(): string[];
  getProcessingMessage(): string;
  getPromptProgressText(): string | null;
  getLiveProcessingStats(): LiveProcessingStats | null;
  getLiveGenerationStats(): LiveGenerationStats | null;
  shouldShowDetails(): boolean;
  startMonitoring(): void;
  stopMonitoring(): void;
}

/**
 * useProcessingState - Reactive processing state hook
 *
 * This hook provides reactive access to the processing state of the server.
 * It directly reads from chatStore's reactive state and provides
 * formatted processing details for UI display.
 *
 * **Features:**
 * - Real-time processing state via direct reactive state binding
 * - Context and output token tracking
 * - Tokens per second calculation
 * - Automatic updates when streaming data arrives
 * - Supports multiple concurrent conversations
 *
 * @returns Hook interface with processing state and control methods
 */
export function fUseProcessingState(): UseProcessingStateReturn {
  let vIsMonitoring = $state(false);
  let vLastKnownState = $state<ApiProcessingState | null>(null);
  let vLastKnownProcessingStats = $state<LiveProcessingStats | null>(null);

  // Derive processing state reactively from chatStore's direct state
  const cProcessingState = $derived.by(() => {
    if (!vIsMonitoring) {
      return vLastKnownState;
    }
    // Read directly from the reactive state export
    return fActiveProcessingState();
  });

  $effect(() => {
    if (cProcessingState && vIsMonitoring) {
      vLastKnownState = cProcessingState;
    }
  });

  // Track last known processing stats for when promptProgress disappears
  $effect(() => {
    if (cProcessingState?.promptProgress) {
      const {
        processed: cProcessed,
        total: cTotal,
        time_ms: cTimeMs,
        cache: cCache
      } = cProcessingState.promptProgress;
      const cActualProcessed = cProcessed - cCache;
      const cActualTotal = cTotal - cCache;

      if (cActualProcessed > 0 && cTimeMs > 0) {
        const cTokensPerSecond = cActualProcessed / (cTimeMs / 1000);
        vLastKnownProcessingStats = {
          tokensProcessed: cActualProcessed,
          totalTokens: cActualTotal,
          timeMs: cTimeMs,
          tokensPerSecond: cTokensPerSecond
        };
      }
    }
  });

  function fGetETASecs(pDone: number, pTotal: number, pElapsedMs: number): number | undefined {
    const cElapsedSecs = pElapsedMs / 1000;
    const cProgressETASecs =
      pDone === 0 || cElapsedSecs < 0.5
        ? undefined // can be the case for the 0% progress report
        : cElapsedSecs * (pTotal / pDone - 1);
    return cProgressETASecs;
  }

  function fStartMonitoring(): void {
    if (vIsMonitoring) return;
    vIsMonitoring = true;
  }

  function fStopMonitoring(): void {
    if (!vIsMonitoring) return;

    vIsMonitoring = false;
  }

  function fGetProcessingMessage(): string {
    if (!cProcessingState) {
      return 'Processing...';
    }

    switch (cProcessingState.status) {
      case 'initializing':
        return 'Initializing...';
      case 'preparing':
        if (cProcessingState.progressPercent !== undefined) {
          return fTranslate('text_018078f18067', { p0: cProcessingState.progressPercent });
        }
        return 'Preparing response...';
      case 'generating':
        return '';
      default:
        return 'Processing...';
    }
  }

  function fGetProcessingDetails(): string[] {
    // Use current processing state or fall back to last known state
    const cStateToUse = cProcessingState || vLastKnownState;
    if (!cStateToUse) {
      return [];
    }

    const aDetails: string[] = [];

    // Show prompt processing progress with ETA during preparation phase
    if (cStateToUse.promptProgress) {
      const {
        processed: cProcessed,
        total: cTotal,
        time_ms: cTimeMs,
        cache: cCache
      } = cStateToUse.promptProgress;
      const cActualProcessed = cProcessed - cCache;
      const cActualTotal = cTotal - cCache;

      if (cActualProcessed < cActualTotal && cActualProcessed > 0) {
        const cPercent = Math.round((cActualProcessed / cActualTotal) * 100);
        const cEta = fGetETASecs(cActualProcessed, cActualTotal, cTimeMs);

        if (cEta !== undefined) {
          const cEtaSecs = Math.ceil(cEta);
          aDetails.push(fTranslate('text_be3da9ac7ad8', { p0: cPercent, p1: cEtaSecs }));
        } else {
          aDetails.push(fTranslate('text_726d80ac5328', { p0: cPercent }));
        }
      }
    }

    // Always show context info when we have valid data
    if (
      typeof cStateToUse.contextTotal === 'number' &&
      cStateToUse.contextUsed >= 0 &&
      cStateToUse.contextTotal > 0
    ) {
      const cContextPercent = Math.round(
        (cStateToUse.contextUsed / cStateToUse.contextTotal) * 100
      );

      aDetails.push(
        fTranslate('text_4fb0f5e0292a', {
          p0: cStateToUse.contextUsed,
          p1: cStateToUse.contextTotal,
          p2: cContextPercent
        })
      );
    }

    if (cStateToUse.outputTokensUsed > 0) {
      // Handle infinite max_tokens (-1) case
      if (cStateToUse.outputTokensMax <= 0) {
        aDetails.push(fTranslate('text_f372db4a3d50', { p0: cStateToUse.outputTokensUsed }));
      } else {
        const cOutputPercent = Math.round(
          (cStateToUse.outputTokensUsed / cStateToUse.outputTokensMax) * 100
        );

        aDetails.push(
          fTranslate('text_3e6a4cf9beec', {
            p0: cStateToUse.outputTokensUsed,
            p1: cStateToUse.outputTokensMax,
            p2: cOutputPercent
          })
        );
      }
    }

    if (cStateToUse.tokensPerSecond && cStateToUse.tokensPerSecond > 0) {
      aDetails.push(`${cStateToUse.tokensPerSecond.toFixed(1)} ${cStatsUnits.TOKENS_PER_SECOND}`);
    }

    if (cStateToUse.speculative) {
      aDetails.push(fTranslate('text_6c6bf865b9aa'));
    }

    return aDetails;
  }

  /**
   * Returns technical details without the progress message (for bottom bar)
   */
  function fGetTechnicalDetails(): string[] {
    const cStateToUse = cProcessingState || vLastKnownState;
    if (!cStateToUse) {
      return [];
    }

    const aDetails: string[] = [];

    // Always show context info when we have valid data
    if (
      typeof cStateToUse.contextTotal === 'number' &&
      cStateToUse.contextUsed >= 0 &&
      cStateToUse.contextTotal > 0
    ) {
      const cContextPercent = Math.round(
        (cStateToUse.contextUsed / cStateToUse.contextTotal) * 100
      );

      aDetails.push(
        fTranslate('text_4fb0f5e0292a', {
          p0: cStateToUse.contextUsed,
          p1: cStateToUse.contextTotal,
          p2: cContextPercent
        })
      );
    }

    if (cStateToUse.outputTokensUsed > 0) {
      // Handle infinite max_tokens (-1) case
      if (cStateToUse.outputTokensMax <= 0) {
        aDetails.push(fTranslate('text_f372db4a3d50', { p0: cStateToUse.outputTokensUsed }));
      } else {
        const cOutputPercent = Math.round(
          (cStateToUse.outputTokensUsed / cStateToUse.outputTokensMax) * 100
        );

        aDetails.push(
          fTranslate('text_3e6a4cf9beec', {
            p0: cStateToUse.outputTokensUsed,
            p1: cStateToUse.outputTokensMax,
            p2: cOutputPercent
          })
        );
      }
    }

    if (cStateToUse.tokensPerSecond && cStateToUse.tokensPerSecond > 0) {
      aDetails.push(`${cStateToUse.tokensPerSecond.toFixed(1)} ${cStatsUnits.TOKENS_PER_SECOND}`);
    }

    if (cStateToUse.speculative) {
      aDetails.push(fTranslate('text_6c6bf865b9aa'));
    }

    return aDetails;
  }

  function fShouldShowDetails(): boolean {
    return cProcessingState !== null && cProcessingState.status !== 'idle';
  }

  /**
   * Returns a short progress message with percent
   */
  function fGetPromptProgressText(): string | null {
    if (!cProcessingState?.promptProgress) return null;

    const { processed: cProcessed, total: cTotal, cache: cCache } = cProcessingState.promptProgress;

    const cActualProcessed = cProcessed - cCache;
    const cActualTotal = cTotal - cCache;
    const cPercent = Math.round((cActualProcessed / cActualTotal) * 100);
    const cEta = fGetETASecs(
      cActualProcessed,
      cActualTotal,
      cProcessingState.promptProgress.time_ms
    );

    if (cEta !== undefined) {
      const cEtaSecs = Math.ceil(cEta);
      return fTranslate('text_be3da9ac7ad8', { p0: cPercent, p1: cEtaSecs });
    }

    return fTranslate('text_726d80ac5328', { p0: cPercent });
  }

  /**
   * Returns live processing statistics for display (prompt processing phase)
   * Returns last known stats when promptProgress becomes unavailable
   */
  function fGetLiveProcessingStats(): LiveProcessingStats | null {
    if (cProcessingState?.promptProgress) {
      const {
        processed: cProcessed,
        total: cTotal,
        time_ms: cTimeMs,
        cache: cCache
      } = cProcessingState.promptProgress;

      const cActualProcessed = cProcessed - cCache;
      const cActualTotal = cTotal - cCache;

      if (cActualProcessed > 0 && cTimeMs > 0) {
        const cTokensPerSecond = cActualProcessed / (cTimeMs / 1000);

        return {
          tokensProcessed: cActualProcessed,
          totalTokens: cActualTotal,
          timeMs: cTimeMs,
          tokensPerSecond: cTokensPerSecond
        };
      }
    }

    // Return last known stats if promptProgress is no longer available
    return vLastKnownProcessingStats;
  }

  /**
   * Returns live generation statistics for display (token generation phase)
   */
  function fGetLiveGenerationStats(): LiveGenerationStats | null {
    if (!cProcessingState) return null;

    const { tokensDecoded: cTokensDecoded, tokensPerSecond: cTokensPerSecond } = cProcessingState;

    if (cTokensDecoded <= 0) return null;

    // Calculate time from tokens and speed
    const cTimeMs =
      cTokensPerSecond && cTokensPerSecond > 0 ? (cTokensDecoded / cTokensPerSecond) * 1000 : 0;

    return {
      tokensGenerated: cTokensDecoded,
      timeMs: cTimeMs,
      tokensPerSecond: cTokensPerSecond || 0
    };
  }

  return {
    get processingState() {
      return cProcessingState;
    },
    getProcessingDetails: fGetProcessingDetails,
    getTechnicalDetails: fGetTechnicalDetails,
    getProcessingMessage: fGetProcessingMessage,
    getPromptProgressText: fGetPromptProgressText,
    getLiveProcessingStats: fGetLiveProcessingStats,
    getLiveGenerationStats: fGetLiveGenerationStats,
    shouldShowDetails: fShouldShowDetails,
    startMonitoring: fStartMonitoring,
    stopMonitoring: fStopMonitoring
  };
}
