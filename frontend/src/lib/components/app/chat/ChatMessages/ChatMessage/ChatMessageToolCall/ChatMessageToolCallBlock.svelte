<script lang="ts">
  import { BuiltInTool } from '$lib/enums';
  import {
    extractSearchQuery,
    extractSearchResults,
    isWebSearchToolName,
    type AgenticSection
  } from '$lib/utils';
  import type { DatabaseMessageExtra } from '$lib/types';
  import ChatMessageToolCallBlockDefault from './ChatMessageToolCallBlockDefault.svelte';
  import ChatMessageToolCallBlockEditFile from './ChatMessageToolCallBlockEditFile.svelte';
  import ChatMessageToolCallBlockExecShellCommand from './ChatMessageToolCallBlockExecShellCommand.svelte';
  import ChatMessageToolCallBlockFileGlobSearch from './ChatMessageToolCallBlockFileGlobSearch.svelte';
  import ChatMessageToolCallBlockGetDatetime from './ChatMessageToolCallBlockGetDatetime.svelte';
  import ChatMessageToolCallBlockGrepSearch from './ChatMessageToolCallBlockGrepSearch.svelte';
  import ChatMessageToolCallBlockLocalShell from './ChatMessageToolCallBlockLocalShell.svelte';
  import ChatMessageToolCallBlockReadFile from './ChatMessageToolCallBlockReadFile.svelte';
  import ChatMessageToolCallBlockRunJavascript from './ChatMessageToolCallBlockRunJavascript.svelte';
  import ChatMessageToolCallBlockSearchResults from './ChatMessageToolCallBlockSearchResults.svelte';
  import ChatMessageToolCallBlockWriteFile from './ChatMessageToolCallBlockWriteFile.svelte';
  import { fIsLocalShellTool } from './parsers/local-shell-run';

  interface Props {
    section: AgenticSection;
    attachments?: DatabaseMessageExtra[];
    open: boolean;
    isStreaming: boolean;
    isExecuting?: boolean;
    onToggle?: () => void;
  }

  let {
    section: vSection,
    attachments: vAttachments,
    open: vOpen,
    isStreaming: vIsStreaming,
    isExecuting: vIsExecuting,
    onToggle: vOnToggle
  }: Props = $props();

  const cIsLocalShellCall = $derived(fIsLocalShellTool(vSection.toolName));
  const cSearchResults = $derived(extractSearchResults(vSection.toolResult));
  const cSearchQuery = $derived(extractSearchQuery(vSection.toolArgs));
  const cIsSearchCall = $derived(
    cSearchResults.length > 0 || (cSearchQuery.length > 0 && isWebSearchToolName(vSection.toolName))
  );
</script>

{#if cIsLocalShellCall}
  <ChatMessageToolCallBlockLocalShell
    section={vSection}
    open={vOpen}
    isStreaming={vIsStreaming}
    onToggle={vOnToggle}
  />
{:else if cIsSearchCall}
  <ChatMessageToolCallBlockSearchResults
    section={vSection}
    open={vOpen}
    isStreaming={vIsStreaming}
    onToggle={vOnToggle}
  />
{:else if vSection.toolName === BuiltInTool.GET_DATETIME}
  <ChatMessageToolCallBlockGetDatetime section={vSection} isStreaming={vIsStreaming} />
{:else if vSection.toolName === BuiltInTool.READ_FILE}
  <ChatMessageToolCallBlockReadFile
    section={vSection}
    open={vOpen}
    isStreaming={vIsStreaming}
    onToggle={vOnToggle}
  />
{:else if vSection.toolName === BuiltInTool.EDIT_FILE}
  <ChatMessageToolCallBlockEditFile
    section={vSection}
    open={vOpen}
    isStreaming={vIsStreaming}
    onToggle={vOnToggle}
  />
{:else if vSection.toolName === BuiltInTool.WRITE_FILE}
  <ChatMessageToolCallBlockWriteFile
    section={vSection}
    open={vOpen}
    isStreaming={vIsStreaming}
    onToggle={vOnToggle}
  />
{:else if vSection.toolName === BuiltInTool.EXEC_SHELL_COMMAND}
  <ChatMessageToolCallBlockExecShellCommand
    section={vSection}
    open={vOpen}
    isStreaming={vIsStreaming}
    isExecuting={vIsExecuting}
    attachments={vAttachments}
    onToggle={vOnToggle}
  />
{:else if vSection.toolName === BuiltInTool.FILE_GLOB_SEARCH}
  <ChatMessageToolCallBlockFileGlobSearch
    section={vSection}
    open={vOpen}
    isStreaming={vIsStreaming}
    onToggle={vOnToggle}
  />
{:else if vSection.toolName === BuiltInTool.GREP_SEARCH}
  <ChatMessageToolCallBlockGrepSearch
    section={vSection}
    open={vOpen}
    isStreaming={vIsStreaming}
    onToggle={vOnToggle}
  />
{:else if vSection.toolName === BuiltInTool.RUN_JAVASCRIPT}
  <ChatMessageToolCallBlockRunJavascript
    section={vSection}
    open={vOpen}
    isStreaming={vIsStreaming}
    onToggle={vOnToggle}
  />
{:else}
  <ChatMessageToolCallBlockDefault
    section={vSection}
    open={vOpen}
    isStreaming={vIsStreaming}
    attachments={vAttachments}
    onToggle={vOnToggle}
  />
{/if}
