import { browser } from '$app/environment';
import { cMediaQueries } from '$lib/constants';

export const cTheme = $state({
  isSystemDark: browser && window.matchMedia(cMediaQueries.PREFERS_DARK).matches
});

if (browser) {
  const cMql = window.matchMedia(cMediaQueries.PREFERS_DARK);

  cMql.addEventListener('change', (pE) => {
    cTheme.isSystemDark = pE.matches;
  });
}
