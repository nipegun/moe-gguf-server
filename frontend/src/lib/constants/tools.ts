import { fTranslate } from '../i18n';
import { ToolSource } from '$lib/enums/tools.enums';

export const cToolGroupLabels = {
  [ToolSource.BUILTIN]: 'Built-in',
  [ToolSource.CUSTOM]: 'JSON Schema',
  [ToolSource.FRONTEND]: 'Browser'
} as const;

export const cToolServerLabels = {
  [ToolSource.BUILTIN]: fTranslate('text_92a726148df7'),
  [ToolSource.CUSTOM]: fTranslate('text_2304e587b23b'),
  [ToolSource.FRONTEND]: fTranslate('text_c78befadef32')
} as const;
