<script lang="ts">
  import { fTranslate } from '../../../../../../i18n';

  import { cIconClassDefault } from '$lib/constants/css-classes';
  import { Plus, File, MessageSquare, Zap, FolderOpen } from '@lucide/svelte';
  import * as DropdownMenu from '$lib/components/ui/dropdown-menu';
  import * as Tooltip from '$lib/components/ui/tooltip';
  import { buttonVariants } from '$lib/components/ui/button';
  import { cn } from '$lib/components/ui/utils';
  import {
    aAttachmentFileItems,
    cAttachmentTooltipText,
    cTooltipDelayDuration
  } from '$lib/constants';
  import {
    ChatFormActionAddToolsSubmenu,
    ChatFormActionAddMcpServersSubmenu,
    ChatFormActionAddReasoningSubmenu
  } from '$lib/components/app';
  import { fUseAttachmentMenu } from '$lib/hooks/use-attachment-menu.svelte';

  interface Props {
    class?: string;
    disabled?: boolean;
    hasAudioModality?: boolean;
    hasVideoModality?: boolean;
    hasVisionModality?: boolean;
    hasMcpPromptsSupport?: boolean;
    hasMcpResourcesSupport?: boolean;
    onFileUpload?: () => void;
    onSystemPromptClick?: () => void;
    onMcpPromptClick?: () => void;
    onMcpSettingsClick?: () => void;
    onMcpResourcesClick?: () => void;
  }

  let {
    class: vClassName = '',
    disabled: vDisabled = false,
    hasAudioModality: vHasAudioModality = false,
    hasVideoModality: vHasVideoModality = false,
    hasVisionModality: vHasVisionModality = false,
    hasMcpPromptsSupport: vHasMcpPromptsSupport = false,
    hasMcpResourcesSupport: vHasMcpResourcesSupport = false,
    onFileUpload: vOnFileUpload,
    onSystemPromptClick: vOnSystemPromptClick,
    onMcpPromptClick: vOnMcpPromptClick,
    onMcpSettingsClick: vOnMcpSettingsClick,
    onMcpResourcesClick: vOnMcpResourcesClick
  }: Props = $props();

  let vDropdownOpen = $state(false);
  // The system message action moves focus to the message editor, so the menu
  // must not restore focus to the trigger on close
  let vSuppressCloseAutoFocus = false;

  function fHandleMcpSettingsClick() {
    vDropdownOpen = false;
    vOnMcpSettingsClick?.();
  }

  const cAttachmentMenu = fUseAttachmentMenu(
    () => ({
      hasVisionModality: vHasVisionModality,
      hasAudioModality: vHasAudioModality,
      hasVideoModality: vHasVideoModality,
      hasMcpPromptsSupport: vHasMcpPromptsSupport,
      hasMcpResourcesSupport: vHasMcpResourcesSupport
    }),
    () => ({
      onFileUpload: vOnFileUpload,
      onSystemPromptClick: vOnSystemPromptClick,
      onMcpPromptClick: vOnMcpPromptClick,
      onMcpResourcesClick: vOnMcpResourcesClick
    }),
    () => {
      vDropdownOpen = false;
    }
  );
</script>

<div class="flex items-center gap-1 {vClassName}">
  <DropdownMenu.Root bind:open={vDropdownOpen}>
    <!-- ignoreNonKeyboardFocus prevents the tooltip from flashing when the
         menu closes and focus returns to the trigger -->
    <Tooltip.Root ignoreNonKeyboardFocus>
      <Tooltip.Trigger>
        {#snippet child({ props })}
          <DropdownMenu.Trigger
            {...props}
            class={cn(
              buttonVariants({ variant: 'secondary' }),
              'file-upload-button h-8 w-8 cursor-pointer rounded-full p-0'
            )}
            disabled={vDisabled}
          >
            <span class="sr-only">{cAttachmentTooltipText}</span>

            <Plus class={cIconClassDefault} />
          </DropdownMenu.Trigger>
        {/snippet}
      </Tooltip.Trigger>

      <Tooltip.Content>
        <p>{cAttachmentTooltipText}</p>
      </Tooltip.Content>
    </Tooltip.Root>

    <DropdownMenu.Content
      align="start"
      class="w-max max-w-[min(20rem,calc(100vw-1rem))] min-w-52"
      onCloseAutoFocus={(e) => {
        if (vSuppressCloseAutoFocus) {
          vSuppressCloseAutoFocus = false;
          e.preventDefault();
        }
      }}
    >
      <ChatFormActionAddReasoningSubmenu />

      <DropdownMenu.Separator />

      <DropdownMenu.Sub>
        <DropdownMenu.SubTrigger class="flex cursor-pointer items-center gap-2">
          <File class={cIconClassDefault} />

          <span>{fTranslate('text_01792a33c228')}</span>
        </DropdownMenu.SubTrigger>

        <DropdownMenu.SubContent class="w-48">
          {#each aAttachmentFileItems as item (item.id)}
            {@const enabled = cAttachmentMenu.isItemEnabled(item.enabledWhen)}
            {#if enabled}
              <DropdownMenu.Item
                class="{item.class ?? ''} flex cursor-pointer items-center gap-2"
                onclick={() => cAttachmentMenu.callbacks[item.action]()}
              >
                <item.icon class={cIconClassDefault} />

                <span>{item.label}</span>
              </DropdownMenu.Item>
            {:else if item.disabledTooltip}
              <Tooltip.Root delayDuration={cTooltipDelayDuration}>
                <Tooltip.Trigger tabindex={-1}>
                  {#snippet child({ props })}
                    <div {...props} class="cursor-default">
                      <DropdownMenu.Item
                        class="{item.class ?? ''} flex items-center gap-2"
                        disabled
                      >
                        <item.icon class={cIconClassDefault} />

                        <span>{item.label}</span>
                      </DropdownMenu.Item>
                    </div>
                  {/snippet}
                </Tooltip.Trigger>

                <Tooltip.Content side="right">
                  <p>{item.disabledTooltip}</p>
                </Tooltip.Content>
              </Tooltip.Root>
            {/if}
          {/each}
        </DropdownMenu.SubContent>
      </DropdownMenu.Sub>

      <DropdownMenu.Item
        class="flex cursor-pointer items-center gap-2"
        onclick={() => {
          vSuppressCloseAutoFocus = true;
          vOnSystemPromptClick?.();
        }}
      >
        <MessageSquare class={cIconClassDefault} />

        <span>{fTranslate('text_8e5a3143184d')}</span>
      </DropdownMenu.Item>

      <ChatFormActionAddToolsSubmenu />

      <ChatFormActionAddMcpServersSubmenu onMcpSettingsClick={fHandleMcpSettingsClick} />

      {#if vHasMcpPromptsSupport}
        <DropdownMenu.Separator />

        <DropdownMenu.Item
          class="flex cursor-pointer items-center gap-2"
          onclick={vOnMcpPromptClick}
        >
          <Zap class={cIconClassDefault} />

          <span>{fTranslate('text_6829fa6ca0ec')}</span>
        </DropdownMenu.Item>
      {/if}

      {#if vHasMcpResourcesSupport}
        <DropdownMenu.Item
          class="flex cursor-pointer items-center gap-2"
          onclick={vOnMcpResourcesClick}
        >
          <FolderOpen class={cIconClassDefault} />

          <span>{fTranslate('text_77a814db2847')}</span>
        </DropdownMenu.Item>
      {/if}
    </DropdownMenu.Content>
  </DropdownMenu.Root>
</div>
