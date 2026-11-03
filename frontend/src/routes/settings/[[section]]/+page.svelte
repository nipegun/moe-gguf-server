<script lang="ts">
  import { SettingsChat } from '$lib/components/app/settings';
  import { page } from '$app/state';
  import { afterNavigate, replaceState } from '$app/navigation';
  import { RouterService } from '$lib/services';
  import { cSettingsSectionSlugs } from '$lib/constants';

  afterNavigate(() => {
    if (!page.params.section) {
      replaceState(RouterService.mSettings(cSettingsSectionSlugs.GENERAL), {});
    }
  });
</script>

<SettingsChat initialSection={(page.params as Record<string, string | undefined>).section} />
