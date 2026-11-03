/**
 * cTheme - Color mode of the interface
 *
 * The preference (system, light or dark) is the theme setting, kept by the server with the other
 * settings in ~/.mgs/config/ui-settings.json; nothing of it is kept in the browser. /api/config.js
 * publishes the saved value as globalThis.cUiTheme before the modules load, and the inline script of
 * src/app.html puts it on <html> before the first paint. This store takes over from there: it
 * starts from the same value (the default of the settings registry when none is saved), follows
 * prefers-color-scheme while the preference is "system" and applies every change at once.
 *
 * The color mode in effect is the dark class of <html> (the dark variant of app.css) and its
 * color-scheme style, which the native controls and scrollbars follow.
 */

import { browser } from '$app/environment';
import { cMediaQueries, cSettingsKeys, dSettingConfigDefault } from '$lib/constants';
import { ColorMode } from '$lib/enums';

/** The server publishes the saved theme setting in /api/config.js (null when none is saved). */
declare global {
  var cUiTheme: string | null | undefined;
}

const aColorModes: string[] = [ColorMode.SYSTEM, ColorMode.LIGHT, ColorMode.DARK];

const cDarkClass = 'dark';

// While the color mode changes, no transition runs: every color switches at once
const cNoTransitionsCss =
  '*,*::before,*::after{-webkit-transition:none!important;transition:none!important}';

/** A value of the theme setting, or the default of the settings registry when it is not one. */
function fToColorMode(pValue: unknown): ColorMode {
  return typeof pValue === 'string' && aColorModes.includes(pValue)
    ? (pValue as ColorMode)
    : (dSettingConfigDefault[cSettingsKeys.THEME] as ColorMode);
}

function fWithoutTransitions(pChange: () => void): void {
  const cStyle = document.createElement('style');
  cStyle.textContent = cNoTransitionsCss;
  document.head.appendChild(cStyle);
  pChange();
  // the styles are computed with the transitions off before they come back
  window.getComputedStyle(document.documentElement).getPropertyValue('color-scheme');
  setTimeout(() => cStyle.remove(), 1);
}

class ThemeStore {
  /** True while the operating system prefers the dark color scheme */
  vIsSystemDark = $state(browser && window.matchMedia(cMediaQueries.PREFERS_DARK).matches);

  /** Color mode chosen in the settings: system, light or dark */
  vPreference = $state<ColorMode>(fToColorMode(browser ? globalThis.cUiTheme : null));

  /** Color mode in effect: light or dark */
  vCurrent = $derived<ColorMode>(
    this.vPreference === ColorMode.DARK ||
      (this.vPreference === ColorMode.SYSTEM && this.vIsSystemDark)
      ? ColorMode.DARK
      : ColorMode.LIGHT
  );

  constructor() {
    if (browser) {
      window.matchMedia(cMediaQueries.PREFERS_DARK).addEventListener('change', (pEvent) => {
        this.vIsSystemDark = pEvent.matches;
        this.mApply();
      });
      this.mApply();
    }
  }

  /**
   * Sets the preference (a value of the theme setting: system, light or dark) and applies it.
   * Saving the setting is up to the settings store.
   */
  mSetPreference(pPreference: string): void {
    this.vPreference = fToColorMode(pPreference);
    if (browser) {
      this.mApply();
    }
  }

  /** Puts the color mode in effect on <html>, only when it changes */
  private mApply(): void {
    const cIsDark = this.vCurrent === ColorMode.DARK;
    const cColorScheme = cIsDark ? ColorMode.DARK : ColorMode.LIGHT;
    const cRoot = document.documentElement;

    if (
      cRoot.classList.contains(cDarkClass) === cIsDark &&
      cRoot.style.colorScheme === cColorScheme
    ) {
      return;
    }

    fWithoutTransitions(() => {
      cRoot.classList.toggle(cDarkClass, cIsDark);
      cRoot.style.colorScheme = cColorScheme;
    });
  }
}

export const cTheme = new ThemeStore();
