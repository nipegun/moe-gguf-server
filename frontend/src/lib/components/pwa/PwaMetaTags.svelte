<script lang="ts">
  import { cAppleMetaTags, cMediaQueries, cThemeColors } from '$lib/constants/pwa';
  import { cAppName } from '$lib/constants';

  let { appName: vAppName = cAppName } = $props();
</script>

<svelte:head>
  <!-- Theme color for light/dark modes -->
  <meta name="theme-color" content={cThemeColors.LIGHT} media={cMediaQueries.PREFERS_LIGHT} />
  <meta name="theme-color" content={cThemeColors.DARK} media={cMediaQueries.PREFERS_DARK} />

  <!-- Apple mobile web app meta tags -->
  <meta
    name={cAppleMetaTags.MOBILE_WEB_APP_CAPABLE.name}
    content={cAppleMetaTags.MOBILE_WEB_APP_CAPABLE.content}
  />
  <meta
    name={cAppleMetaTags.STATUS_BAR_STYLE.name}
    content={cAppleMetaTags.STATUS_BAR_STYLE.content}
  />
  <meta name={cAppleMetaTags.MOBILE_WEB_APP_TITLE.name} content={vAppName} />
</svelte:head>
