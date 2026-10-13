// actions accepted by the realtime inference control endpoint (API_CHAT.CONTROL)
// kept separate from the endpoint paths since these are protocol level verbs
export const cControlAction = {
  END_REASONING: 'reasoning_end'
} as const;

export type ControlAction = (typeof cControlAction)[keyof typeof cControlAction];
