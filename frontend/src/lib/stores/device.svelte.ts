import { browser } from '$app/environment';
import { cMediaQueries } from '$lib/constants';

/**
 * iOS UA token detection.
 *
 * iPadOS 13+ ships a desktop macOS UA, so 'iPad' is no longer present in it;
 * a Macintosh UA combined with touch support is treated as an iPad instead.
 * Third-party iOS browsers (Chrome, Firefox, Edge) and in-app WKWebViews all
 * run on WKWebView and emit their own tokens (CriOS/FxiOS/EdgiOS/GSA) instead
 * of the trailing 'Safari/' the Safari app keeps.
 */
const cUaPatterns = {
  IOS_PHONE: /iPhone|iPod/,
  MACINTOSH: /Macintosh/,
  SAFARI: /Safari/,
  WEBVIEW_IOS: /CriOS|FxiOS|EdgiOS|GSA/
} as const;

interface DeviceContext {
  /** Any iOS/iPadOS device, regardless of which app or browser embeds the page. */
  isIOSDevice: boolean;
  /** The Safari browser app on iOS, excluding other iOS browsers and WKWebViews. */
  isIOSSafari: boolean;
  /** Any WKWebView context on iOS: in-app browsers, embedded web views, and the
   *  third-party iOS browsers (all of which share the WKWebView engine). */
  isWKWebView: boolean;
  /** PWA standalone mode: the page was launched from the home screen icon. */
  isStandalone: boolean;
}

const dServerDefault: DeviceContext = {
  isIOSDevice: false,
  isIOSSafari: false,
  isWKWebView: false,
  isStandalone: false
};

function fDetect(): DeviceContext {
  if (!browser) return dServerDefault;

  const cUa = navigator.userAgent;
  const cIsTouch = navigator.maxTouchPoints > 0;

  const cIsIOSDevice =
    cUaPatterns.IOS_PHONE.test(cUa) || (cUaPatterns.MACINTOSH.test(cUa) && cIsTouch);

  // Safari keeps 'Safari/' in the UA; non-Safari iOS browsers emit their own
  // token instead. WKWebView typically omits 'Safari/' entirely.
  const cHasSafariToken = cUaPatterns.SAFARI.test(cUa) && !cUaPatterns.WEBVIEW_IOS.test(cUa);
  const cIsIOSSafari = cIsIOSDevice && cHasSafariToken;
  const cIsWKWebView = cIsIOSDevice && !cHasSafariToken;

  // navigator.standalone is the legacy iOS-only flag (deprecated but still
  // present); display-mode: standalone is the modern standard (Safari 16.4+).
  const cIsStandalone =
    window.matchMedia(cMediaQueries.DISPLAY_MODE_STANDALONE).matches ||
    (navigator as Navigator & { standalone?: boolean }).standalone === true;

  return {
    isIOSDevice: cIsIOSDevice,
    isIOSSafari: cIsIOSSafari,
    isWKWebView: cIsWKWebView,
    isStandalone: cIsStandalone
  };
}

export const cDevice = $state<DeviceContext>(fDetect());

if (browser) {
  // isStandalone can change at runtime (e.g. user installs the PWA while the
  // tab is open); the UA-derived flags are static for the session.
  const cMql = window.matchMedia(cMediaQueries.DISPLAY_MODE_STANDALONE);

  cMql.addEventListener('change', (pE) => {
    cDevice.isStandalone = pE.matches;
  });
}
