/**
 * Model load progress helpers for the /models/sse surfaces
 * (selector row and chat message), and the queue notice of a request
 * that waits for a free slot.
 */

import { dModelLoadStageLabels, cModelLoadTailShare } from '$lib/constants';
import { fTranslate } from '$lib/i18n';
import type { ApiQueueStatus } from '$lib/types';

/**
 * Notice shown while a request of the interface waits behind others, or null when it does not
 * wait. When a request sent directly to the API runs or waits ahead, the notice says so.
 */
export function fQueueStatusText(pStatus: ApiQueueStatus | null): string | null {
  if (!pStatus || pStatus.ahead <= 0) return null;

  return pStatus.ahead_api > 0
    ? fTranslate('text_5e0c9a7d2b41', { p0: pStatus.ahead })
    : fTranslate('text_8f3b6c1e9d07', { p0: pStatus.ahead });
}

/**
 * Human label for a model load stage.
 */
export function fModelLoadStageLabel(pStage: ApiModelLoadStage): string {
  return dModelLoadStageLabels[pStage];
}

/**
 * Overall load fraction (0.0 -> 1.0) across the declared stage plan.
 * text_model fills [0, 1 - tail], each later phase owns one tail slice.
 */
export function fModelLoadFraction(pProgress: ModelLoadProgress | null): number {
  if (!pProgress) return 0;

  const { stages: cStages, current: cCurrent, value: cValue } = pProgress;
  const cTailCount = Math.max(cStages.length - 1, 0);
  const cTextCeiling = 1 - cTailCount * cModelLoadTailShare;
  const cIdx = cStages.indexOf(cCurrent);

  if (cIdx <= 0) {
    return cValue * cTextCeiling;
  }

  return cTextCeiling + (cIdx - 1 + cValue) * cModelLoadTailShare;
}

/**
 * Single line describing load progress: active stage label and overall percent.
 * Returns null when there is no progress to show.
 */
export function fModelLoadProgressText(pProgress: ModelLoadProgress | null): string | null {
  if (!pProgress) return null;

  const cLabel = fModelLoadStageLabel(pProgress.current);
  return `${cLabel} ${Math.round(fModelLoadFraction(pProgress) * 100)}%`;
}
