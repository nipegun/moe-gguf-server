<script lang="ts">
  import { Card } from '$lib/components/ui/card';
  import type { DatabaseMessageExtraMcpPrompt } from '$lib/types';
  import { cMcpStore } from '$lib/stores/mcp.svelte';
  import { SvelteMap } from 'svelte/reactivity';
  import { McpPromptVariant } from '$lib/enums';
  import { TruncatedText } from '$lib/components/app/misc';
  import * as Tooltip from '$lib/components/ui/tooltip';

  interface ContentPart {
    text: string;
    argKey: string | null;
  }

  interface Props {
    class?: string;
    prompt: DatabaseMessageExtraMcpPrompt;
    variant?: McpPromptVariant;
    isLoading?: boolean;
    loadError?: string;
  }

  let {
    class: vClassName = '',
    prompt: vPrompt,
    variant: vVariant = McpPromptVariant.MESSAGE,
    isLoading: vIsLoading = false,
    loadError: vLoadError
  }: Props = $props();

  let vHoveredArgKey = $state<string | null>(null);
  let vArgumentEntries = $derived(Object.entries(vPrompt.arguments ?? {}));
  let vHasArguments = $derived(vPrompt.arguments && Object.keys(vPrompt.arguments).length > 0);
  let vHasContent = $derived(vPrompt.content && vPrompt.content.trim().length > 0);

  let vContentParts = $derived.by((): ContentPart[] => {
    if (!vPrompt.content || !vHasArguments) {
      return [{ text: vPrompt.content || '', argKey: null }];
    }

    const aParts: ContentPart[] = [];
    let vRemaining = vPrompt.content;

    const cValueToKey = new SvelteMap<string, string>();
    for (const [cKey, cValue] of vArgumentEntries) {
      if (cValue && cValue.trim()) {
        cValueToKey.set(cValue, cKey);
      }
    }

    const cSortedValues = [...cValueToKey.keys()].sort((pA, pB) => pB.length - pA.length);

    while (vRemaining.length > 0) {
      let vEarliestMatch: { index: number; value: string; key: string } | null = null;

      for (const cValue of cSortedValues) {
        const cIndex = vRemaining.indexOf(cValue);
        if (cIndex !== -1 && (vEarliestMatch === null || cIndex < vEarliestMatch.index)) {
          vEarliestMatch = { index: cIndex, value: cValue, key: cValueToKey.get(cValue)! };
        }
      }

      if (vEarliestMatch) {
        if (vEarliestMatch.index > 0) {
          aParts.push({ text: vRemaining.slice(0, vEarliestMatch.index), argKey: null });
        }

        aParts.push({ text: vEarliestMatch.value, argKey: vEarliestMatch.key });
        vRemaining = vRemaining.slice(vEarliestMatch.index + vEarliestMatch.value.length);
      } else {
        aParts.push({ text: vRemaining, argKey: null });

        break;
      }
    }

    return aParts;
  });

  let vShowArgBadges = $derived(vHasArguments && !vIsLoading && !vLoadError);
  let vIsAttachment = $derived(vVariant === McpPromptVariant.ATTACHMENT);
  let vTextSizeClass = $derived(vIsAttachment ? 'text-xs' : 'text-md');
  let vPaddingClass = $derived(vIsAttachment ? 'px-3 py-2' : 'px-3.75 py-2.5');
  let vMaxHeightStyle = $derived(
    vIsAttachment ? 'max-height: 6rem;' : 'max-height: var(--max-message-height);'
  );

  const cServerFavicon = $derived(cMcpStore.mGetServerFavicon(vPrompt.serverName));
  const cServerDisplayName = $derived(cMcpStore.mGetServerDisplayName(vPrompt.serverName));
</script>

<div class="flex flex-col gap-2 {vClassName}">
  <div class="flex items-center justify-between gap-2">
    <div class="inline-flex flex-wrap items-center gap-1.25 text-xs text-muted-foreground">
      <Tooltip.Root>
        <Tooltip.Trigger>
          {#if cServerFavicon}
            <img
              src={cServerFavicon}
              alt=""
              class="h-3.5 w-3.5 shrink-0 rounded-sm"
              onerror={(e) => {
                (e.currentTarget as HTMLImageElement).style.display = 'none';
              }}
            />
          {/if}
        </Tooltip.Trigger>

        <Tooltip.Content>
          <span>{cServerDisplayName}</span>
        </Tooltip.Content>
      </Tooltip.Root>

      <TruncatedText text={vPrompt.name} />
    </div>

    {#if vShowArgBadges}
      <div class="flex flex-wrap justify-end gap-1">
        {#each vArgumentEntries as [key, value] (key)}
          <Tooltip.Root>
            <Tooltip.Trigger>
              <!-- svelte-ignore a11y_no_static_element_interactions -->
              <span
                class="rounded-sm bg-purple-200/60 px-1.5 py-0.5 text-[10px] leading-none text-purple-700 transition-opacity dark:bg-purple-800/40 dark:text-purple-300 {vHoveredArgKey &&
                vHoveredArgKey !== key
                  ? 'opacity-30'
                  : ''}"
                onmouseenter={() => (vHoveredArgKey = key)}
                onmouseleave={() => (vHoveredArgKey = null)}
              >
                {key}
              </span>
            </Tooltip.Trigger>

            <Tooltip.Content>
              <span class="max-w-xs break-all">{value}</span>
            </Tooltip.Content>
          </Tooltip.Root>
        {/each}
      </div>
    {/if}
  </div>

  {#if vLoadError}
    <Card
      class="relative overflow-hidden rounded-[1.125rem] border border-destructive/50 bg-destructive/10 backdrop-blur-md"
    >
      <div
        class="overflow-y-auto {vPaddingClass}"
        style="{vMaxHeightStyle} overflow-wrap: anywhere; word-break: break-word;"
      >
        <span class="{vTextSizeClass} text-destructive">{vLoadError}</span>
      </div>
    </Card>
  {:else if vIsLoading}
    <Card
      class="relative overflow-hidden rounded-[1.125rem] border border-purple-200 bg-purple-500/10 px-1 py-2 backdrop-blur-md dark:border-purple-800 dark:bg-purple-500/20"
    >
      <div
        class="overflow-y-auto {vPaddingClass}"
        style="{vMaxHeightStyle} overflow-wrap: anywhere; word-break: break-word;"
      >
        <div class="space-y-2">
          <div class="h-3 w-3/4 animate-pulse rounded bg-foreground/20"></div>

          <div class="h-3 w-full animate-pulse rounded bg-foreground/20"></div>

          <div class="h-3 w-5/6 animate-pulse rounded bg-foreground/20"></div>
        </div>
      </div>
    </Card>
  {:else if vHasContent}
    <Card
      class="relative overflow-hidden rounded-[1.125rem] border border-purple-200 bg-purple-500/10 py-0 text-foreground backdrop-blur-md dark:border-purple-800 dark:bg-purple-500/20"
    >
      <div
        class="overflow-y-auto {vPaddingClass}"
        style="{vMaxHeightStyle} overflow-wrap: anywhere; word-break: break-word;"
      >
        <span class="{vTextSizeClass} whitespace-pre-wrap">
          <!-- This formatting is needed to keep the text in proper shape -->
          <!-- svelte-ignore a11y_no_static_element_interactions -->
          {#each vContentParts as part, i (i)}{#if part.argKey}<span
                class="rounded-sm bg-purple-300/50 px-0.5 text-purple-900 transition-opacity dark:bg-purple-700/50 dark:text-purple-100 {vHoveredArgKey &&
                vHoveredArgKey !== part.argKey
                  ? 'opacity-30'
                  : ''}"
                onmouseenter={() => (vHoveredArgKey = part.argKey)}
                onmouseleave={() => (vHoveredArgKey = null)}>{part.text}</span
              >{:else}<span class="transition-opacity {vHoveredArgKey ? 'opacity-30' : ''}"
                >{part.text}</span
              >{/if}{/each}</span
        >
      </div>
    </Card>
  {/if}
</div>
