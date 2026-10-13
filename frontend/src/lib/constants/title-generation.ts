import { fTranslate } from '../i18n';
/* Title generation constants */
export const cTitleGeneration = {
  MIN_LENGTH: 3,
  FALLBACK: fTranslate('text_0d33235199fb'),
  DEFAULT_PROMPT:
    'Based on the following interaction, generate a short, concise title (maximum 6-8 words) that captures the main topic. Return ONLY the title text, nothing else. Do not use quotes.\n\nUser: {{USER}}\n\nAssistant: {{ASSISTANT}}\n\nTitle:',
  PREFIX_PATTERN: /^(Title:|Subject:|Topic:)\s*/i,
  QUOTE_PATTERN: /^["]|["]$/g
} as const;
