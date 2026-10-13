import { fTranslate } from '../i18n';
import { ReasoningEffort } from '$lib/enums';
import type { ReasoningEffortLevel } from '$lib/types';

/**
 * Reasoning effort UI labels.
 * Keys match the ReasoningEffort enum values for type-safe lookups.
 */
export const dReasoningEffortLabels: Record<string, string> = {
  [ReasoningEffort.DEFAULT]: 'Default',
  [ReasoningEffort.OFF]: 'Off',
  [ReasoningEffort.LOW]: 'Low',
  [ReasoningEffort.MEDIUM]: 'Medium',
  [ReasoningEffort.HIGH]: 'High',
  [ReasoningEffort.MAX]: 'Max'
};

export const aReasoningEffortLevels: ReasoningEffortLevel[] = [
  { value: ReasoningEffort.DEFAULT, label: fTranslate('text_21b111cbfe6e') },
  { value: ReasoningEffort.OFF, label: fTranslate('text_ca7981b46ecf') },
  { value: ReasoningEffort.LOW, label: fTranslate('text_f793de205ead') },
  { value: ReasoningEffort.MEDIUM, label: fTranslate('text_8e588cd18774') },
  { value: ReasoningEffort.HIGH, label: fTranslate('text_c4ebc6d4a583') },
  { value: ReasoningEffort.MAX, label: fTranslate('text_a1a5936d3b0f'), hasInfo: true }
];
