import dGermanDe from './locales/de-DE.json';
import dEnglishUs from './locales/en-US.json';
import dEnglishGb from './locales/en-GB.json';
import dSpanishAr from './locales/es-AR.json';
import dSpanishEs from './locales/es-ES.json';
import dFrenchFr from './locales/fr-FR.json';
import dHebrewIl from './locales/he-IL.json';
import dHindiIn from './locales/hi-IN.json';
import dItalianIt from './locales/it-IT.json';
import dJapaneseJp from './locales/ja-JP.json';
import dKoreanKr from './locales/ko-KR.json';
import dPortugueseBr from './locales/pt-BR.json';
import dPortuguesePt from './locales/pt-PT.json';
import dRussianRu from './locales/ru-RU.json';
import dChineseCn from './locales/zh-CN.json';

export const aLanguages = [
  'de-DE',
  'en-GB',
  'en-US',
  'es-AR',
  'es-ES',
  'fr-FR',
  'he-IL',
  'hi-IN',
  'it-IT',
  'ja-JP',
  'ko-KR',
  'pt-BR',
  'pt-PT',
  'ru-RU',
  'zh-CN'
] as const;
export type Language = (typeof aLanguages)[number];

/** The server publishes the saved interface language in /api/config.js, before the modules load. */
declare global {
  var cUiLanguage: string | null | undefined;
}

const dCatalogs: Record<Language, Record<string, string>> = {
  'de-DE': dGermanDe,
  'en-GB': dEnglishGb,
  'en-US': dEnglishUs,
  'es-AR': dSpanishAr,
  'es-ES': dSpanishEs,
  'fr-FR': dFrenchFr,
  'he-IL': dHebrewIl,
  'hi-IN': dHindiIn,
  'it-IT': dItalianIt,
  'ja-JP': dJapaneseJp,
  'ko-KR': dKoreanKr,
  'pt-BR': dPortugueseBr,
  'pt-PT': dPortuguesePt,
  'ru-RU': dRussianRu,
  'zh-CN': dChineseCn
};

// The language comes from ?lang= in the URL, else from the server's settings document
// (~/.mgs/config/ui-settings.json), which /api/config.js publishes before the modules load.
function fInitialLanguage(): Language {
  if (typeof window === 'undefined') return 'en-US';
  let vLanguage = new URL(window.location.href).searchParams.get('lang');
  vLanguage ??= typeof globalThis.cUiLanguage === 'string' ? globalThis.cUiLanguage : null;
  return aLanguages.includes(vLanguage as Language) ? (vLanguage as Language) : 'en-US';
}

const cLanguage = fInitialLanguage();

export function fGetLanguage(): Language {
  return cLanguage;
}

export function fTranslate(pKey: string, pValues: Record<string, unknown> = {}): string {
  const vMessage = dCatalogs[cLanguage][pKey] ?? dCatalogs['en-US'][pKey] ?? pKey;
  return vMessage.replace(/\{(p\d+)\}/g, (pMatch, pName: string) =>
    Object.prototype.hasOwnProperty.call(pValues, pName) ? String(pValues[pName]) : pMatch
  );
}

export async function fSetLanguage(pLanguage: string): Promise<void> {
  if (!aLanguages.includes(pLanguage as Language) || typeof window === 'undefined') return;
  // Imported here, not at the top: this module is imported before every store
  const { cUiSettingsStore } = await import('$lib/stores/ui-settings.svelte');
  cUiSettingsStore.mSet('language', pLanguage);
  const cSaved = cUiSettingsStore.vIsLoaded && (await cUiSettingsStore.mFlush());
  const vUrl = new URL(window.location.href);
  if (cSaved) {
    // the server now publishes the new language
    vUrl.searchParams.delete('lang');
  } else {
    // not saved: keep the choice in the URL for this session
    vUrl.searchParams.set('lang', pLanguage);
  }
  // Reload so imported configuration labels and already-open dialogs agree.
  if (vUrl.toString() === window.location.href) {
    window.location.reload();
  } else {
    window.location.assign(vUrl.toString());
  }
}
