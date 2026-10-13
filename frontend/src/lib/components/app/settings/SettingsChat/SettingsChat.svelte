<script lang="ts">
  import LanguageSelector from '../LanguageSelector.svelte';
  import { fTranslate } from '../../../../i18n';

  import {
    SettingsChatDesktopSidebar,
    SettingsChatFields,
    SettingsChatImportExportTab,
    SettingsChatMobileHeader,
    SettingsChatToolsTab,
    SettingsFooter
  } from '$lib/components/app/settings';
  import { fConfig, cSettingsStore } from '$lib/stores/settings.svelte';
  import {
    cNumericFields,
    cPositiveIntegerFields,
    aSettingsChatSections,
    cSettingsSectionTitles
  } from '$lib/constants';
  import type { SettingsSection } from '$lib/types';
  import { RouterService } from '$lib/services/router.service';
  import { setMode } from 'mode-watcher';
  import { ColorMode } from '$lib/enums/ui.enums';
  import { fade } from 'svelte/transition';
  import { goto } from '$app/navigation';
  import { Button } from '$lib/components/ui/button';
  import { RefreshCw } from '@lucide/svelte';
  import { page } from '$app/state';
  import { setChatSettingsConfigContext } from '$lib/contexts';
  import { dSettingsReferrer } from '$lib/stores/settings-referrer.svelte';
  import { cModelsStore } from '$lib/stores/models.svelte';
  import { fIsRouterMode } from '$lib/stores/server.svelte';
  interface Props {
    initialSection?: string;
    getSectionHref?: (pSection: SettingsSection) => string;
  }

  let { initialSection: vInitialSection, getSectionHref: vGetSectionHref }: Props = $props();

  let vActiveSlug = $derived(
    vInitialSection ?? (page.params as Record<string, string | undefined>).section ?? 'general'
  );

  let vCurrentSection = $derived(
    aSettingsChatSections.find((pSection) => pSection.slug === vActiveSlug) ||
      aSettingsChatSections[0]
  );

  let vLocalConfig: SettingsConfigType = $state({ ...fConfig() });

  let vMobileHeader: { fUpdateCarousel: () => void } | undefined;

  let vFetchInitiated = false;

  $effect(() => {
    if (fIsRouterMode() && vCurrentSection.fields && !vFetchInitiated) {
      vFetchInitiated = true;

      void cModelsStore
        .mFetch()
        .then(() => cModelsStore.mFetchRouterModels())
        .then(() => cModelsStore.mFetchModalitiesForLoadedModels())
        .then(() => cModelsStore.mEnsureFirstModelSelected());
    }
  });

  function fHandleThemeChange(pNewTheme: string) {
    vLocalConfig.theme = pNewTheme;
    setMode(pNewTheme as ColorMode);
  }

  function fHandleConfigChange(pKey: string, pValue: string | boolean) {
    vLocalConfig[pKey] = pValue;
  }

  function fHandleReset() {
    vLocalConfig = { ...fConfig() };
    setMode(vLocalConfig.theme as ColorMode);
    vMobileHeader?.fUpdateCarousel();
  }

  function fHandleSave() {
    if (
      vLocalConfig.customJson &&
      typeof vLocalConfig.customJson === 'string' &&
      vLocalConfig.customJson.trim()
    ) {
      try {
        JSON.parse(vLocalConfig.customJson);
      } catch (vError) {
        alert(fTranslate('text_f3201c77465e'));
        console.error(vError);
        return;
      }
    }

    const dProcessedConfig = { ...vLocalConfig };

    for (const cField of cNumericFields) {
      if (dProcessedConfig[cField] !== undefined && dProcessedConfig[cField] !== '') {
        const cNumValue = Number(dProcessedConfig[cField]);
        if (!isNaN(cNumValue)) {
          if ((cPositiveIntegerFields as readonly string[]).includes(cField)) {
            dProcessedConfig[cField] = Math.max(1, Math.round(cNumValue));
          } else {
            dProcessedConfig[cField] = cNumValue;
          }
        } else {
          alert(fTranslate('text_2d3b86345740', { p0: cField }));
          return;
        }
      }
    }

    cSettingsStore.mUpdateMultipleConfig(dProcessedConfig);
    goto(dSettingsReferrer.url);
  }

  export function fReset() {
    vLocalConfig = { ...fConfig() };
  }

  setChatSettingsConfigContext({
    get localConfig() {
      return vLocalConfig;
    },
    handleConfigChange: fHandleConfigChange,
    handleThemeChange: fHandleThemeChange
  });
</script>

<div class="mx-auto flex h-full w-full flex-col md:pl-8" in:fade={{ duration: 150 }}>
  <div class="flex flex-1 flex-col gap-4 md:flex-row">
    <SettingsChatDesktopSidebar
      sections={aSettingsChatSections}
      isActive={(section: SettingsSection) => section.slug === vActiveSlug}
      getHref={vGetSectionHref ??
        ((section: SettingsSection) => RouterService.mSettings(section.slug))}
    />

    <SettingsChatMobileHeader
      sections={aSettingsChatSections}
      isActive={(section: SettingsSection) => section.slug === vActiveSlug}
      getHref={vGetSectionHref ??
        ((section: SettingsSection) => RouterService.mSettings(section.slug))}
      bind:this={vMobileHeader}
    />

    <div class="mx-auto max-w-3xl flex-1">
      <div class="space-y-6 p-4 md:p-6 md:pt-28">
        <div class="grid">
          <div class="mb-6 flex items-center gap-2 border-b border-border/30 pb-6 md:flex">
            <vCurrentSection.icon class="h-5 w-5" />
            <h3 class="text-lg font-semibold">{vCurrentSection.title}</h3>
          </div>

          {#if vCurrentSection.title === cSettingsSectionTitles.TOOLS}
            <SettingsChatToolsTab />
          {:else if vCurrentSection.title === cSettingsSectionTitles.IMPORT_EXPORT}
            <SettingsChatImportExportTab />
          {:else if vCurrentSection.fields}
            <div class="space-y-6">
              {#if vCurrentSection.title === cSettingsSectionTitles.GENERAL}
                <LanguageSelector />
              {/if}
              <SettingsChatFields
                fields={vCurrentSection.fields}
                localConfig={vLocalConfig}
                onConfigChange={fHandleConfigChange}
                onThemeChange={fHandleThemeChange}
              />

              {#if vCurrentSection.title === cSettingsSectionTitles.GENERAL}
                <div class="flex justify-end">
                  <Button variant="outline" onclick={() => window.location.reload()}>
                    <RefreshCw class="h-3 w-3" />
                    {fTranslate('text_10f0b454208d')}
                  </Button>
                </div>
              {/if}
            </div>
          {/if}
        </div>

        <div class="mt-8 border-t border-border/30 pt-6">
          <p class="text-xs text-muted-foreground">{fTranslate('text_d1955162085a')}</p>
        </div>
      </div>

      <SettingsFooter onReset={fHandleReset} onSave={fHandleSave} />
    </div>
  </div>
</div>
