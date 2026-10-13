<script lang="ts">
  import { fTranslate } from '../../../../../../i18n';

  // Block for the tools of the local shell MCP server (bash.run,
  // zsh.run, sh.run and psh.run). The title is the tool name and the
  // body holds up to three plain monospace boxes, styled like the Input
  // box of the other tools: stdin with the bare command (filled in
  // while the model writes it), then stdout and stderr, each one only
  // when that stream produced something. The borders are white (stdin;
  // dark grey on the light theme), green (stdout) and red (stderr) for
  // every tool of that server. The exit code goes next to the title of
  // the output boxes, or next to stdin when there is no output.
  // The sudo password prompts of the command are answered in
  // DialogLocalShellAskpass, mounted in the root layout.

  import { SyntaxHighlightedCode } from '$lib/components/app';
  import { FileTypeText } from '$lib/enums';
  import { cMaxHeightCodeBlock } from '$lib/constants';
  import type { AgenticSection } from '$lib/utils';
  import {
    fParseLocalShellCommand,
    fParseLocalShellResult,
    type LocalShellRunResult
  } from './parsers/local-shell-run';
  import ToolCallBlock from './ToolCallBlock.svelte';

  interface Props {
    section: AgenticSection;
    open: boolean;
    isStreaming: boolean;
    onToggle?: () => void;
  }

  let {
    section: vSection,
    open: vOpen,
    isStreaming: vIsStreaming,
    onToggle: vOnToggle
  }: Props = $props();

  const vCommand = $derived(fParseLocalShellCommand(vSection));
  const dRun = $derived(fParseLocalShellResult(vSection.toolResult));
  // Any other result (an argument error, a denied permission, a failed
  // request) is error text without an exit code.
  const vErrorText = $derived(!dRun && vSection.toolResult ? vSection.toolResult : '');
  const vStdout = $derived(dRun?.stdout ?? '');
  const vStderr = $derived(dRun ? dRun.stderr : vErrorText);
  const vStatus = $derived(fStatus(dRun));

  // " (code 2)", " (code 137, timed out)"... or "" before the result arrives.
  function fStatus(pRun: LocalShellRunResult | null): string {
    if (!pRun) return '';
    const aLParts: string[] = [];
    if (pRun.exitCode !== null)
      aLParts.push(fTranslate('text_cbfccaa6cfac', { p0: pRun.exitCode }));
    if (pRun.timedOut) aLParts.push(fTranslate('text_3dcd80f1b15f'));
    if (pRun.cancelled) aLParts.push(fTranslate('text_f12fde684b2e'));
    return aLParts.length > 0 ? ` (${aLParts.join(', ')})` : '';
  }
</script>

<ToolCallBlock
  section={vSection}
  open={vOpen}
  isStreaming={vIsStreaming}
  meta={null}
  title={vSection.toolName ?? ''}
  onToggle={vOnToggle}
>
  {#snippet children(_meta, ctx)}
    {#if vCommand}
      <div class="mb-1.5 flex min-w-0 items-center gap-2 text-xs text-muted-foreground/70">
        <span class="min-w-0">stdin{vStdout || vStderr ? '' : vStatus}:</span>
      </div>
      <SyntaxHighlightedCode
        code={vCommand}
        language={FileTypeText.PLAIN_TEXT}
        maxHeight={cMaxHeightCodeBlock}
        borderColor="var(--local-shell-stdin-border)"
        streaming={ctx.isCodeStreaming}
      />
    {/if}
    {#if vStdout}
      <div
        class="{vCommand
          ? 'mt-4'
          : ''} mb-1.5 flex min-w-0 items-center gap-2 text-xs text-muted-foreground/70"
      >
        <span class="min-w-0">stdout{vStatus}:</span>
      </div>
      <SyntaxHighlightedCode
        code={vStdout}
        language={FileTypeText.PLAIN_TEXT}
        maxHeight={cMaxHeightCodeBlock}
        borderColor="var(--local-shell-stdout-border)"
      />
    {/if}
    {#if vStderr}
      <div
        class="{vCommand || vStdout
          ? 'mt-4'
          : ''} mb-1.5 flex min-w-0 items-center gap-2 text-xs text-muted-foreground/70"
      >
        <span class="min-w-0">stderr{vStatus}:</span>
      </div>
      <SyntaxHighlightedCode
        code={vStderr}
        language={FileTypeText.PLAIN_TEXT}
        maxHeight={cMaxHeightCodeBlock}
        borderColor="var(--local-shell-stderr-border)"
      />
    {/if}
  {/snippet}
</ToolCallBlock>
