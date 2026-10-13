import { getContext, setContext } from 'svelte';
import { cContextKeyChatSettingsConfig } from '$lib/constants';

export interface ChatSettingsConfigContext {
  readonly localConfig: SettingsConfigType;
  handleConfigChange: (pKey: string, pValue: string | boolean) => void;
  handleThemeChange: (pTheme: string) => void;
}

const cChatSettingsConfigKey = Symbol.for(cContextKeyChatSettingsConfig);

export function fSetChatSettingsConfigContext(
  pCtx: ChatSettingsConfigContext
): ChatSettingsConfigContext {
  return setContext(cChatSettingsConfigKey, pCtx);
}

export function fGetChatSettingsConfigContext(): ChatSettingsConfigContext {
  return getContext(cChatSettingsConfigKey);
}
