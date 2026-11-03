import { ReasoningEffort } from '$lib/enums';

/**
 * Reasoning effort to token budget mapping.
 * Maps the ReasoningEffort enum values to concrete token counts for the server.
 */
export const dReasoningEffortTokens: Record<string, number> = {
  [ReasoningEffort.LOW]: 512,
  [ReasoningEffort.MEDIUM]: 2048,
  [ReasoningEffort.HIGH]: 8192,
  [ReasoningEffort.MAX]: -1 // unlimited
};
