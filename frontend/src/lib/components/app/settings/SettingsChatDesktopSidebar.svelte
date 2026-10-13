<script lang="ts">
  import { fTranslate } from '../../../i18n';

  import { cIconClassDefault } from '$lib/constants/css-classes';
  import { Settings } from '@lucide/svelte';
  import type { SettingsSection, SettingsSectionTitle } from '$lib/types';

  interface Props {
    sections: SettingsSection[];
    isActive: (pSection: SettingsSection) => boolean;
    getHref?: (pSection: SettingsSection) => string;
    onSectionChange?: (pSection: SettingsSectionTitle) => void;
  }

  let {
    sections: vSections,
    isActive: vIsActive,
    getHref: vGetHref,
    onSectionChange: vOnSectionChange
  }: Props = $props();
</script>

<div class="sticky top-2 hidden w-64 flex-col self-start bg-background py-4 md:flex gap-6">
  <div class="flex items-center gap-2 py-2">
    <Settings class="h-5 w-5 md:h-6 md:w-6" />

    <h1 class="text-xl font-semibold md:text-2xl">{fTranslate('text_74a883a037bc')}</h1>
  </div>

  <nav class="space-y-1">
    {#each vSections as section (section.title)}
      {#if vGetHref}
        <a
          class="flex w-full cursor-pointer items-center gap-3 rounded-lg px-3 py-2 text-start text-sm no-underline transition-colors hover:bg-accent {vIsActive(
            section
          )
            ? 'bg-accent text-accent-foreground'
            : 'text-muted-foreground'}"
          href={vGetHref(section)}
        >
          <section.icon class={cIconClassDefault} />
          <span class="ml-2">{section.title}</span>
        </a>
      {:else}
        <button
          class="flex w-full cursor-pointer items-center gap-3 rounded-lg px-3 py-2 text-start text-sm transition-colors hover:bg-accent {vIsActive(
            section
          )
            ? 'bg-accent text-accent-foreground'
            : 'text-muted-foreground'}"
          onclick={() => vOnSectionChange?.(section.title)}
        >
          <section.icon class={cIconClassDefault} />
          <span class="ml-2">{section.title}</span>
        </button>
      {/if}
    {/each}
  </nav>
</div>
