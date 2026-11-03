import { cSettingsFallbackExitRoute } from '$lib/constants';

let vUrl = $state<string>(cSettingsFallbackExitRoute);

export const dSettingsReferrer = {
  get url() {
    return vUrl;
  },
  set url(pValue: string) {
    vUrl = pValue;
  }
};
