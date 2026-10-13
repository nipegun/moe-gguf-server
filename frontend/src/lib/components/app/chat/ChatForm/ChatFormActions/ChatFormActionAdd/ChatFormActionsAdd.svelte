<script lang="ts">
  import { cIsMobile } from '$lib/stores/viewport.svelte';
  import ChatFormActionAddDropdown from './ChatFormActionAddDropdown.svelte';
  import ChatFormActionAddSheet from './ChatFormActionAddSheet.svelte';
  import ChatFormActionAddButton from './ChatFormActionAddButton.svelte';

  interface Props {
    disabled?: boolean;
    hasAudioModality?: boolean;
    hasVideoModality?: boolean;
    hasMcpPromptsSupport?: boolean;
    hasMcpResourcesSupport?: boolean;
    hasVisionModality?: boolean;
    onFileUpload?: () => void;
    onMcpPromptClick?: () => void;
    onMcpResourcesClick?: () => void;
    onMcpSettingsClick?: () => void;
    onSystemPromptClick?: () => void;
  }

  let {
    disabled: vDisabled = false,
    hasAudioModality: vHasAudioModality = false,
    hasVideoModality: vHasVideoModality = false,
    hasMcpPromptsSupport: vHasMcpPromptsSupport = false,
    hasMcpResourcesSupport: vHasMcpResourcesSupport = false,
    hasVisionModality: vHasVisionModality = false,
    onFileUpload: vOnFileUpload,
    onMcpPromptClick: vOnMcpPromptClick,
    onMcpResourcesClick: vOnMcpResourcesClick,
    onMcpSettingsClick: vOnMcpSettingsClick,
    onSystemPromptClick: vOnSystemPromptClick
  }: Props = $props();
</script>

{#if cIsMobile.current}
  <ChatFormActionAddSheet
    disabled={vDisabled}
    hasAudioModality={vHasAudioModality}
    hasVideoModality={vHasVideoModality}
    hasVisionModality={vHasVisionModality}
    hasMcpPromptsSupport={vHasMcpPromptsSupport}
    hasMcpResourcesSupport={vHasMcpResourcesSupport}
    onFileUpload={vOnFileUpload}
    onSystemPromptClick={vOnSystemPromptClick}
    onMcpPromptClick={vOnMcpPromptClick}
    onMcpResourcesClick={vOnMcpResourcesClick}
  >
    {#snippet trigger({ disabled, onclick })}
      <ChatFormActionAddButton {disabled} {onclick} />
    {/snippet}
  </ChatFormActionAddSheet>
{:else}
  <ChatFormActionAddDropdown
    disabled={vDisabled}
    hasAudioModality={vHasAudioModality}
    hasVideoModality={vHasVideoModality}
    hasVisionModality={vHasVisionModality}
    hasMcpPromptsSupport={vHasMcpPromptsSupport}
    hasMcpResourcesSupport={vHasMcpResourcesSupport}
    onFileUpload={vOnFileUpload}
    onMcpPromptClick={vOnMcpPromptClick}
    onMcpResourcesClick={vOnMcpResourcesClick}
    onMcpSettingsClick={vOnMcpSettingsClick}
    onSystemPromptClick={vOnSystemPromptClick}
  />
{/if}
