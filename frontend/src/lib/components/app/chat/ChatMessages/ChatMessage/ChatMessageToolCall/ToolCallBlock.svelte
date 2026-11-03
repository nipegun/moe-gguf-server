<script lang="ts" generics="TMeta">
  // Generic chrome shell shared by every per-tool block under
  // `ChatMessageToolCall/`. Owns:
  //   - the collapsible wrapper (defaults to CollapsibleContentBlock;
  //     `exec_shell_command` swaps in CollapsibleTerminalBlock via the
  //     `wrapper` prop);
  //   - the icon, spinner state, and MCP favicon fallback chain;
  //   - the status subtitle pill.
  // Components supply only their `meta`, a title snippet, and a body
  // snippet - everything around them is this single source of truth.

  import { Loader2, Wrench } from '@lucide/svelte';
  import { CollapsibleContentBlock } from '$lib/components/app';
  import { cIconClassDefault, cIconClassSpin } from '$lib/constants/css-classes';
  import { AgenticSectionType } from '$lib/enums';
  import { fGetBuiltinToolUi } from '$lib/constants/built-in-tools';
  import { cMcpStore } from '$lib/stores/mcp.svelte';
  import type { Component, Snippet } from 'svelte';
  import type { AgenticSection, BuiltinToolUiEntry } from '$lib/utils';

  type ToolCallBlockMetaWithError = TMeta & { errorMessage?: string };

  interface ToolCallCtx {
    isStreaming: boolean;
    isPending: boolean;
    isStreamingCall: boolean;
    isCodeStreaming: boolean;
  }

  interface Props {
    section: AgenticSection;
    open: boolean;
    isStreaming: boolean;
    /**
     * The per-tool meta, including any `errorMessage` field that the
     * shared chrome uses to compute the status pill subtitle.
     */
    meta: ToolCallBlockMetaWithError | null | undefined;
    /**
     * True while the tool's process is actively producing output
     * chunks after its args finished streaming (used by
     * `exec_shell_command`'s stdout feed).
     */
    extraLiveStreaming?: boolean;
    /**
     * Swap the title-row icon for a spinning `Loader2` while the
     * spinner is showing. Only meaningful for tools where "live"
     * is interesting (e.g. exec_shell_command showing the in-flight
     * process). Other tools leave it off and render the spinner
     * inline within the body.
     */
    spinIconWhenActive?: boolean;
    /**
     * Wrapper component that renders the title row and the body
     * children. Defaults to CollapsibleContentBlock;
     * `exec_shell_command` uses CollapsibleTerminalBlock for its
     * terminal-style frame.
     */
    wrapper?: typeof CollapsibleContentBlock;
    title?: string;
    titleSnippet?: Snippet;
    onToggle?: () => void;
    children: Snippet<[TMeta | null | undefined, ToolCallCtx]>;
  }

  let {
    section: vSection,
    open: vOpen,
    isStreaming: vIsStreaming,
    meta: vMeta,
    extraLiveStreaming: vExtraLiveStreaming = false,
    spinIconWhenActive: vSpinIconWhenActive = false,
    wrapper: Wrapper = CollapsibleContentBlock,
    title: vTitle,
    titleSnippet: vTitleSnippet,
    onToggle: vOnToggle,
    children: vChildren
  }: Props = $props();

  const cIsPending = $derived(vSection.type === AgenticSectionType.TOOL_CALL_PENDING);
  const cIsStreamingCall = $derived(vSection.type === AgenticSectionType.TOOL_CALL_STREAMING);
  const cShowSpinner = $derived(
    cIsPending || (cIsStreamingCall && vIsStreaming) || vExtraLiveStreaming
  );
  const cIsCodeStreaming = $derived(vIsStreaming && (cIsPending || cIsStreamingCall));

  const cToolUi: BuiltinToolUiEntry | null = $derived(fGetBuiltinToolUi(vSection.toolName));
  const cToolIcon: Component = $derived(
    vSpinIconWhenActive && cShowSpinner ? Loader2 : (cToolUi?.icon ?? Wrench)
  );
  const cToolIconClass = $derived(
    vSpinIconWhenActive && cShowSpinner ? cIconClassSpin : cIconClassDefault
  );
  // Drop the MCP favicon while the spinner is on so the title row
  // signals "in flight" without being overwritten by server branding.
  const cMcpServerFavicon = $derived(
    cShowSpinner ? null : cMcpStore.mGetServerFaviconForTool(vSection.toolName)
  );
  const cIconUrl = $derived(
    cShowSpinner || (cToolUi?.icon ?? null) || !cMcpServerFavicon ? null : cMcpServerFavicon
  );

  function fSubtitleFor(pErrorMessage?: string): string | undefined {
    if (vExtraLiveStreaming) return 'streaming...';
    if (cShowSpinner) return 'executing...';
    if (pErrorMessage) return 'failed';
    if (cIsStreamingCall && !vIsStreaming) return 'incomplete';
    return undefined;
  }

  const cSubtitle = $derived(fSubtitleFor(vMeta?.errorMessage));
</script>

<Wrapper
  open={vOpen}
  class="my-2"
  icon={cToolIcon}
  iconClass={cToolIconClass}
  iconUrl={cIconUrl}
  title={vTitle}
  titleSnippet={vTitleSnippet}
  subtitle={cSubtitle}
  onToggle={vOnToggle}
>
  {@render vChildren(vMeta, {
    isStreaming: vIsStreaming,
    isPending: cIsPending,
    isStreamingCall: cIsStreamingCall,
    isCodeStreaming: cIsCodeStreaming
  })}
</Wrapper>
