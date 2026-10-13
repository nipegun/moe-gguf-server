<script lang="ts">
  import { fTranslate } from '../../../../../../i18n';

  import { ChevronDown, ShieldQuestion } from '@lucide/svelte';
  import { ChatMessageActionCard } from '$lib/components/app';
  import { Button, buttonVariants } from '$lib/components/ui/button';
  import * as ButtonGroup from '$lib/components/ui/button-group';
  import { cn } from '$lib/components/ui/utils';
  import * as DropdownMenu from '$lib/components/ui/dropdown-menu';
  import { ToolSource, ToolPermissionDecision } from '$lib/enums';
  import { cToolServerLabels } from '$lib/constants';
  import { cToolsStore } from '$lib/stores/tools.svelte';

  interface Props {
    toolName: string;
    serverLabel: string;
    onDecision: (pDecision: ToolPermissionDecision) => void;
  }

  let { toolName: vToolName, serverLabel: vServerLabel, onDecision: vOnDecision }: Props = $props();
</script>

<ChatMessageActionCard icon={ShieldQuestion}>
  {#snippet message()}
    {fTranslate('text_44f5e65bfcaf')}
    <span class="font-semibold">{vToolName}</span>{#if vServerLabel}
      {fTranslate('text_75857a458999')} <span class="font-semibold">{vServerLabel}</span>{/if}?
  {/snippet}

  {#snippet actions()}
    <DropdownMenu.Root>
      <ButtonGroup.Root class="overflow-hidden rounded-md shadow-sm">
        <Button
          variant="secondary"
          size="sm"
          class="!rounded-r-none !shadow-none"
          onclick={() => vOnDecision(ToolPermissionDecision.ONCE)}
        >
          {fTranslate('text_168511d24d9e')}
        </Button>

        <ButtonGroup.Separator />

        <DropdownMenu.Trigger
          class={cn(
            buttonVariants({ variant: 'secondary', size: 'sm' }),
            'inline-flex cursor-pointer items-center !rounded-l-none !shadow-none !px-2'
          )}
          aria-label={fTranslate('text_5680c021458b')}
        >
          <ChevronDown class="h-3.5 w-3.5" />
        </DropdownMenu.Trigger>
      </ButtonGroup.Root>

      <DropdownMenu.Content align="start" class="min-w-[8rem]">
        <DropdownMenu.Item onclick={() => vOnDecision(ToolPermissionDecision.ALWAYS)}>
          {fTranslate('text_977618bd8bc7')}
          <pre>{vToolName}</pre>
          {fTranslate('text_7c9bbe5ec9b3')}
        </DropdownMenu.Item>
        {#if vServerLabel}
          <DropdownMenu.Item onclick={() => vOnDecision(ToolPermissionDecision.ALWAYS_SERVER)}>
            {fTranslate('text_c3b44cb1bd9a')}
            {vServerLabel}
          </DropdownMenu.Item>
        {:else}
          {@const source = cToolsStore.mGetToolSource(vToolName)}
          {@const providerName =
            source === ToolSource.BUILTIN
              ? cToolServerLabels[ToolSource.BUILTIN]
              : source === ToolSource.CUSTOM
                ? cToolServerLabels[ToolSource.CUSTOM]
                : 'MCP Tools'}
          <DropdownMenu.Item onclick={() => vOnDecision(ToolPermissionDecision.ALWAYS_SERVER)}>
            {fTranslate('text_41b73824de28')}
            {providerName}
          </DropdownMenu.Item>
        {/if}
      </DropdownMenu.Content>
    </DropdownMenu.Root>

    <Button
      variant="destructive"
      size="sm"
      onclick={() => vOnDecision(ToolPermissionDecision.DENY)}
    >
      {fTranslate('text_05a2d7332eb9')}
    </Button>
  {/snippet}
</ChatMessageActionCard>
