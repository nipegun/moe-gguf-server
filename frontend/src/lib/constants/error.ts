import { fTranslate } from '../i18n';
export const dErrorMessages = {
  NETWORK: {
    GENERIC: fTranslate('text_fa67448b4462'),
    NXDOMAIN: fTranslate('text_51f6f3d1a801'),
    REFUSED: fTranslate('text_448ea20860db'),
    TIMEOUT: fTranslate('text_7aa26ae72ac1'),
    UNREACHABLE: fTranslate('text_a28e8330bb9f')
  },
  HTTP: {
    GENERIC: fTranslate('text_cfce761befa8'),
    ACCESS_DENIED: fTranslate('text_cc11d415d932'),
    INTERNAL_ERROR: fTranslate('text_9051661bb355'),
    NOT_FOUND: fTranslate('text_e3ebaa16dd9d'),
    TEMPORARILY_UNAVAILABLE: fTranslate('text_797397887a90')
  }
};

export const dHttpCodeToString: Record<string, string> = {
  401: dErrorMessages.HTTP.ACCESS_DENIED,
  403: dErrorMessages.HTTP.ACCESS_DENIED,
  500: dErrorMessages.HTTP.INTERNAL_ERROR,
  503: dErrorMessages.HTTP.TEMPORARILY_UNAVAILABLE
};
