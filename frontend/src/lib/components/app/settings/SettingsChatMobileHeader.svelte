<script lang="ts">
  import { fTranslate } from '../../../i18n';

  import { cIconClassDefault } from '$lib/constants/css-classes';
  import { Settings, ChevronLeft, ChevronRight } from '@lucide/svelte';
  import { onMount, tick } from 'svelte';
  import type { SettingsSection, SettingsSectionTitle } from '$lib/types';
  import { fUseScrollCarousel } from '$lib/hooks/use-scroll-carousel.svelte';

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

  const cCarousel = fUseScrollCarousel();

  onMount(async () => {
    await tick();
    if (cCarousel.scrollContainer) {
      const cActiveTab = cCarousel.scrollContainer.querySelector('[data-active="true"]');
      if (cActiveTab instanceof HTMLElement) {
        cCarousel.mScrollToCenter(cActiveTab);
      }
    }
  });

  export function fUpdateCarousel() {
    setTimeout(cCarousel.mUpdateScrollButtons, 100);
  }
</script>

<div class="sticky top-0 z-10 flex flex-col bg-background md:hidden">
  <div class="flex items-center gap-2 px-4 pt-4 pb-2 md:pt-6">
    <Settings class="h-5 w-5 md:h-6 md:w-6" />

    <h1 class="text-xl font-semibold md:text-2xl">{fTranslate('text_74a883a037bc')}</h1>
  </div>

  <div class="border-b border-border/30 py-2">
    <div class="relative flex items-center" style="scroll-padding: 1rem;">
      <button
        class="absolute left-2 z-10 flex h-6 w-6 items-center justify-center rounded-full bg-muted shadow-md backdrop-blur-sm transition-opacity hover:bg-accent {cCarousel.canScrollLeft
          ? 'opacity-100'
          : 'pointer-events-none opacity-0'}"
        onclick={cCarousel.mScrollLeft}
        aria-label={fTranslate('text_35dfa1045cb2')}
      >
        <ChevronLeft class={cIconClassDefault} />
      </button>

      <div
        class="scrollbar-hide overflow-x-auto py-2"
        bind:this={cCarousel.scrollContainer}
        onscroll={cCarousel.mUpdateScrollButtons}
      >
        <div class="flex min-w-max gap-2">
          {#each vSections as vSection (vSection.title)}
            {#if vGetHref}
              <a
                class="flex cursor-pointer items-center gap-2 rounded-lg px-3 py-2 text-sm whitespace-nowrap no-underline transition-colors first:ml-4 last:mr-4 hover:bg-accent {vIsActive(
                  vSection
                )
                  ? 'bg-accent text-accent-foreground'
                  : 'text-muted-foreground'}"
                data-active={vIsActive(vSection)}
                href={vGetHref(vSection)}
                onclick={(pE: MouseEvent) => {
                  cCarousel.mScrollToCenter(pE.currentTarget as HTMLElement);
                }}
              >
                <vSection.icon class="{cIconClassDefault} flex-shrink-0" />
                <span>{vSection.title}</span>
              </a>
            {:else}
              <button
                class="flex cursor-pointer items-center gap-2 rounded-lg px-3 py-2 text-sm whitespace-nowrap transition-colors first:ml-4 last:mr-4 hover:bg-accent {vIsActive(
                  vSection
                )
                  ? 'bg-accent text-accent-foreground'
                  : 'text-muted-foreground'}"
                data-active={vIsActive(vSection)}
                onclick={(pE: MouseEvent) => {
                  vOnSectionChange?.(vSection.title);
                  cCarousel.mScrollToCenter(pE.currentTarget as HTMLElement);
                }}
              >
                <vSection.icon class="{cIconClassDefault} flex-shrink-0" />
                <span>{vSection.title}</span>
              </button>
            {/if}
          {/each}
        </div>
      </div>

      <button
        class="absolute right-2 z-10 flex h-6 w-6 items-center justify-center rounded-full bg-muted shadow-md backdrop-blur-sm transition-opacity hover:bg-accent {cCarousel.canScrollRight
          ? 'opacity-100'
          : 'pointer-events-none opacity-0'}"
        onclick={cCarousel.mScrollRight}
        aria-label={fTranslate('text_366bd578350a')}
      >
        <ChevronRight class={cIconClassDefault} />
      </button>
    </div>
  </div>
</div>
