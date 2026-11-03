<script lang="ts">
  import { fTranslate } from '../../../../../../i18n';

  import { ChatMessageMcpPromptContent, ActionIcon } from '$lib/components/app';
  import { X } from '@lucide/svelte';
  import type { DatabaseMessageExtraMcpPrompt } from '$lib/types';
  import { McpPromptVariant } from '$lib/enums';

  interface Props {
    class?: string;
    isLoading?: boolean;
    loadError?: string;
    onRemove?: () => void;
    prompt: DatabaseMessageExtraMcpPrompt;
    readonly?: boolean;
  }

  let {
    class: vClassName = '',
    isLoading: vIsLoading = false,
    loadError: vLoadError,
    onRemove: vOnRemove,
    prompt: vPrompt,
    readonly: vReadonly = false
  }: Props = $props();
</script>

<div class="group relative {vClassName}">
  <ChatMessageMcpPromptContent
    isLoading={vIsLoading}
    loadError={vLoadError}
    prompt={vPrompt}
    variant={McpPromptVariant.ATTACHMENT}
  />

  {#if !vReadonly && vOnRemove}
    <div
      class="absolute top-10 right-2 flex items-center justify-center opacity-0 transition-opacity group-hover:opacity-100"
    >
      <ActionIcon
        icon={X}
        tooltip={fTranslate('text_c3812fc4acb8')}
        stopPropagationOnClick
        onclick={() => vOnRemove?.()}
      />
    </div>
  {/if}
</div>
