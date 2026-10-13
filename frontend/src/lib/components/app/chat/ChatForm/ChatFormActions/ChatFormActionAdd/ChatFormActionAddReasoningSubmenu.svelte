<script lang="ts">
  import { fTranslate } from '../../../../../../i18n';

  import { cIconClassDefault } from '$lib/constants/css-classes';
  import { Lightbulb, LightbulbOff, Check, Info } from '@lucide/svelte';
  import * as DropdownMenu from '$lib/components/ui/dropdown-menu';
  import * as Tooltip from '$lib/components/ui/tooltip';
  import { fUseReasoningMenu } from '$lib/hooks/use-reasoning-menu.svelte';

  const cReasoning = fUseReasoningMenu();
</script>

{#if cReasoning.modelSupportsThinking}
  <DropdownMenu.Sub>
    <DropdownMenu.SubTrigger class="flex cursor-pointer items-center gap-2">
      {#if cReasoning.thinkingEnabled}
        <Lightbulb class="{cIconClassDefault} shrink-0 text-amber-400" />
      {:else if cReasoning.isOff}
        <LightbulbOff class="{cIconClassDefault} shrink-0 text-muted-foreground" />
      {:else}
        <Lightbulb class="{cIconClassDefault} shrink-0 text-muted-foreground" />
      {/if}

      <span
        class="text-sm inline-flex gap-2 whitespace-nowrap {!cReasoning.thinkingEnabled
          ? 'text-muted-foreground'
          : ''}"
      >
        {fTranslate('text_d8211e24e83d')}
        <span class="text-muted-foreground">
          {cReasoning.currentLabel}
        </span>
      </span>
    </DropdownMenu.SubTrigger>

    <!-- sized by its content so a level name never shares one narrow column with its hint -->
    <DropdownMenu.SubContent
      class="w-max max-w-[min(24rem,calc(100vw-1rem))] min-w-60 bg-popover p-1.5 text-popover-foreground shadow-md outline-none"
    >
      {#each cReasoning.levels as level (level.value)}
        {@const tokenLabel = cReasoning.tokenLabel(level)}
        <DropdownMenu.Item
          class="flex w-full cursor-pointer items-center gap-3 rounded-md px-2 py-1.75 text-left text-sm transition-colors hover:bg-accent {cReasoning.isSelected(
            level
          )
            ? 'bg-accent'
            : ''}"
          onclick={() => cReasoning.select(level)}
        >
          {#if cReasoning.isSelected(level)}
            <Check class="{cIconClassDefault} shrink-0 text-foreground" />
          {:else}
            <div class="{cIconClassDefault} shrink-0"></div>
          {/if}

          <span class="shrink-0 whitespace-nowrap">{level.label}</span>

          {#if tokenLabel}
            <span class="ml-auto min-w-0 text-right text-[11px] text-muted-foreground opacity-60">
              {tokenLabel}
            </span>
          {:else}
            <span class="ml-auto"></span>
          {/if}

          {#if level.hasInfo}
            <Tooltip.Root>
              <Tooltip.Trigger>
                <Info class="h-3.5 w-3.5 shrink-0 text-muted-foreground" />
              </Tooltip.Trigger>
              <Tooltip.Content side="left">
                <p>{fTranslate('text_20c2a9fdd6b0')}</p>
              </Tooltip.Content>
            </Tooltip.Root>
          {/if}
        </DropdownMenu.Item>
      {/each}
    </DropdownMenu.SubContent>
  </DropdownMenu.Sub>
{/if}
