<script lang="ts">
  import { fTranslate } from '../../../../i18n';

  import { remark } from 'remark';
  import remarkBreaks from 'remark-breaks';
  import remarkGfm from 'remark-gfm';
  import remarkMath from 'remark-math';
  import rehypeHighlight from 'rehype-highlight';
  import { all as lowlightAll } from 'lowlight';
  import remarkRehype from 'remark-rehype';
  import rehypeKatex from 'rehype-katex';
  import rehypeStringify from 'rehype-stringify';
  import type { Root as HastRoot, RootContent as HastRootContent } from 'hast';
  import type { Root as MdastRoot } from 'mdast';
  import { browser } from '$app/environment';
  import { onDestroy, tick } from 'svelte';
  import { SvelteMap } from 'svelte/reactivity';
  import { fRehypeRestoreTableHtml } from './plugins/rehype/table-html-restorer';
  import { fRehypeEnhanceLinks } from './plugins/rehype/enhance-links';
  import { fRehypeEnhanceCodeBlocks } from './plugins/rehype/enhance-code-blocks';
  import { fRehypeEnhanceMermaidBlocks } from './plugins/rehype/enhance-mermaid-blocks';
  import { cRehypeMermaidPre } from './plugins/rehype/mermaid-pre';
  import { cRehypeSvgPre } from './plugins/rehype/svg-pre';
  import { fRehypeEnhanceSvgBlocks } from './plugins/rehype/enhance-svg-blocks';
  import { fRehypeResolveAttachmentImages } from './plugins/rehype/resolve-attachment-images';
  import { fRehypeRtlSupport } from './plugins/rehype/rehype-rtl-support';
  import { fRemarkLiteralHtml } from './plugins/remark/literal-html';
  import {
    fGetHastNodeId,
    fGetMdastNodeHash,
    fIsAppendMode,
    fGetCodeInfoFromTarget
  } from './markdown-utils';
  import {
    preprocessLaTeX,
    createImageErrorFallback,
    escapeHtml,
    copyCodeToClipboard,
    copyToClipboard
  } from '$lib/utils';
  import {
    cImageNotErrorBoundSelector,
    cDataErrorBoundAttr,
    cDataErrorHandledAttr,
    cBoolTrueString,
    cSettingsKeys,
    cCodeBlockHeaderClass,
    cMermaidWrapperClass,
    cMermaidBlockClass,
    cMermaidLanguage,
    cMermaidSyntaxAttr,
    cMermaidRenderedAttr,
    cSvgWrapperClass,
    cSvgBlockClass,
    cSvgLanguage,
    cXmlLanguage,
    cSvgTagPrefix,
    cSvgSourceAttr,
    cSvgRenderedAttr,
    cSvgInlineShadowStyle,
    cToggleSourceBtnClass,
    cDiagramViewModeAttr,
    cDiagramViewRendered,
    cDiagramViewSource
  } from '$lib/constants';
  import { ColorMode, UrlProtocol } from '$lib/enums';
  import { FileTypeText } from '$lib/enums/files.enums';
  import { highlightCode, detectIncompleteCodeBlock, type IncompleteCodeBlock } from '$lib/utils';
  import { fSanitizeSvg } from '$lib/utils/sanitize-svg';
  import { fMountSvgShadow } from '$lib/utils/svg-shadow';
  import '$styles/katex-custom.scss';
  import githubDarkCss from 'highlight.js/styles/github-dark.css?inline';
  import githubLightCss from 'highlight.js/styles/github.css?inline';
  import { mode } from 'mode-watcher';
  import {
    CodeBlockActions,
    DialogCodePreview,
    DialogMermaidPreview,
    ActionIconCopyToClipboard
  } from '$lib/components/app';
  import { fCreateAutoScrollController } from '$lib/hooks/use-auto-scroll.svelte';
  import type { DatabaseMessageExtra } from '$lib/types/database';
  import { fConfig } from '$lib/stores/settings.svelte';

  interface Props {
    attachments?: DatabaseMessageExtra[];
    content: string;
    class?: string;
    disableMath?: boolean;
  }

  interface MarkdownBlock {
    id: string;
    html: string;
    contentHash?: string;
  }

  let {
    content: vContent,
    attachments: vAttachments,
    class: vClassName = '',
    disableMath: vDisableMath = false
  }: Props = $props();

  let vContainerRef = $state<HTMLDivElement>();
  let vRenderedBlocks = $state<MarkdownBlock[]>([]);
  let vUnstableBlockHtml = $state('');
  let vIncompleteCodeBlock = $state<IncompleteCodeBlock | null>(null);
  const cStreamingSvgCode = $derived.by(() => {
    const cBlock = vIncompleteCodeBlock;
    if (!cBlock) return null;
    if (cBlock.language === cSvgLanguage) return cBlock.code;
    if (cBlock.language === cXmlLanguage && cBlock.code.trimStart().startsWith(cSvgTagPrefix))
      return cBlock.code;
    return null;
  });
  const cLiveSvgHtml = $derived(cStreamingSvgCode !== null ? fSanitizeSvg(cStreamingSvgCode) : '');

  // Derived rather than called inline in the template so it only recomputes when
  // the block actually changes. Auto-detection is disabled while streaming: it
  // costs ~38ms a call and re-guesses the language on every chunk.
  const cStreamingCodeHtml = $derived(
    vIncompleteCodeBlock
      ? highlightCode(vIncompleteCodeBlock.code, vIncompleteCodeBlock.language || 'text', false)
      : ''
  );
  let vPreviewDialogOpen = $state(false);
  let vPreviewCode = $state('');
  let vPreviewLanguage = $state('text');
  let vMermaidPreviewOpen = $state(false);
  let vMermaidPreviewSvgHtml = $state('');
  let vSvgPreviewLive = $state(false);
  let vStreamingSvgHost = $state<HTMLDivElement | null>(null);

  // While the zoom dialog is open on a streaming svg, mirror the live render into it
  $effect(() => {
    if (vSvgPreviewLive && cLiveSvgHtml) vMermaidPreviewSvgHtml = cLiveSvgHtml;
  });

  // Mount the streaming svg into its shadow host on every chunk so it renders live
  $effect(() => {
    if (vStreamingSvgHost) fMountSvgShadow(vStreamingSvgHost, cLiveSvgHtml, cSvgInlineShadowStyle);
  });

  let vStreamingCodeScrollContainer = $state<HTMLDivElement>();

  // Auto-scroll controller for streaming code block content
  const cStreamingAutoScroll = fCreateAutoScrollController();

  let vPendingMarkdown: string | null = null;
  let vIsProcessing = false;

  // Per-instance transform cache, avoids re-transforming stable blocks during streaming
  // Garbage collected when component is destroyed (on conversation change)
  const cTransformCache = new SvelteMap<string, string>();
  let vPreviousContent = '';

  const cThemeStyleId = `highlight-theme-${(window.idxThemeStyle = (window.idxThemeStyle ?? 0) + 1)}`;

  let vProcessor = $derived(() => {
    void vAttachments;
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    let vProc: any = remark().use(remarkGfm); // GitHub Flavored Markdown

    if (!vDisableMath) {
      vProc = vProc.use(remarkMath); // Parse $inline$ and $$block$$ math
    }

    vProc = vProc
      .use(remarkBreaks) // Convert line breaks to <br>
      .use(fRemarkLiteralHtml) // Treat raw HTML as literal text with preserved indentation
      .use(remarkRehype); // Convert Markdown AST to rehype

    if (!vDisableMath) {
      vProc = vProc.use(rehypeKatex); // Render math using KaTeX
    }

    return vProc
      .use(rehypeHighlight, {
        languages: lowlightAll,
        aliases: { [FileTypeText.XML]: [FileTypeText.SVELTE, FileTypeText.VUE] }
      }) // Add syntax highlighting
      .use(fRehypeRestoreTableHtml) // Restore limited HTML (e.g., <br>, <ul>) inside Markdown tables
      .use(fRehypeEnhanceLinks) // Add target="_blank" to links
      .use(cRehypeMermaidPre) // Convert mermaid blocks to <pre class="mermaid">
      .use(cRehypeSvgPre) // Convert svg blocks to <pre class="svg-block">
      .use(fRehypeEnhanceCodeBlocks) // Wrap code blocks with header and actions
      .use(fRehypeEnhanceMermaidBlocks) // Wrap mermaid blocks with header and actions
      .use(fRehypeEnhanceSvgBlocks) // Wrap svg blocks with header and actions
      .use(fRehypeResolveAttachmentImages, { attachments: vAttachments })
      .use(fRehypeRtlSupport) // Add bidirectional text support
      .use(rehypeStringify, { allowDangerousHtml: true }); // Convert to HTML string
  });

  /**
   * Removes click event listeners from copy and preview buttons.
   * Called on component destroy.
   */
  function fCleanupEventListeners() {
    if (!vContainerRef) return;

    const cCopyButtons = vContainerRef.querySelectorAll<HTMLButtonElement>('.copy-code-btn');
    const cPreviewButtons = vContainerRef.querySelectorAll<HTMLButtonElement>('.preview-code-btn');

    for (const cButton of cCopyButtons) {
      cButton.removeEventListener('click', fHandleCopyClick);
    }

    for (const cButton of cPreviewButtons) {
      cButton.removeEventListener('click', fHandlePreviewClick);
    }
  }

  /**
   * Removes this component's highlight.js theme style from the document head.
   * Called on component destroy to clean up injected styles.
   */
  function fCleanupHighlightTheme() {
    if (!browser) return;

    const cExistingTheme = document.getElementById(cThemeStyleId);
    cExistingTheme?.remove();
  }

  /**
   * Loads the appropriate highlight.js theme based on dark/light mode.
   * Injects a scoped style element into the document head.
   * @param pIsDark - Whether to load the dark theme (true) or light theme (false)
   */
  function fLoadHighlightTheme(pIsDark: boolean) {
    if (!browser) return;

    const cExistingTheme = document.getElementById(cThemeStyleId);
    cExistingTheme?.remove();

    const cStyle = document.createElement('style');
    cStyle.id = cThemeStyleId;
    cStyle.textContent = pIsDark ? githubDarkCss : githubLightCss;

    document.head.appendChild(cStyle);
  }

  /**
   * Transforms a single MDAST node to HTML string with caching.

  /**
   * Transforms a single MDAST node to HTML string with caching.
   * Runs the full remark/rehype plugin pipeline (GFM, math, syntax highlighting, etc.)
   * on an isolated single-node tree, then stringifies the resulting HAST to HTML.
   * Results are cached by node position hash for streaming performance.
   * @param pProcessorInstance - The remark/rehype processor instance
   * @param pNode - The MDAST node to transform
   * @param pIndex - Node index for hash fallback
   * @returns Object containing the HTML string and cache hash
   */
  async function fTransformMdastNode(
    pProcessorInstance: ReturnType<typeof vProcessor>,
    pNode: unknown,
    pIndex: number
  ): Promise<{ html: string; hash: string }> {
    const cHash = fGetMdastNodeHash(pNode, pIndex);

    const cCached = cTransformCache.get(cHash);
    if (cCached) {
      return { html: cCached, hash: cHash };
    }

    const dSingleNodeRoot = { type: 'root', children: [pNode] };
    const cTransformedRoot = (await pProcessorInstance.run(
      dSingleNodeRoot as MdastRoot
    )) as HastRoot;
    const cHtml = pProcessorInstance.stringify(cTransformedRoot);

    cTransformCache.set(cHash, cHtml);

    return { html: cHtml, hash: cHash };
  }

  /**
   * Handles click events on copy buttons within code blocks.
   * Copies the raw code content to the clipboard.
   * @param pEvent - The click event from the copy button
   */
  async function fHandleCopyClick(pEvent: Event) {
    pEvent.preventDefault();
    pEvent.stopPropagation();

    const cTarget = pEvent.currentTarget as HTMLButtonElement | null;

    if (!cTarget) {
      return;
    }

    const cInfo = fGetCodeInfoFromTarget(cTarget);

    if (!cInfo) {
      return;
    }

    try {
      await copyCodeToClipboard(cInfo.rawCode);
    } catch (vError) {
      console.error('Failed to copy code:', vError);
    }
  }

  /**
   * Handles preview dialog open state changes.
   * Clears preview content when dialog is closed.
   * @param pOpen - Whether the dialog is being opened or closed
   */
  function fHandlePreviewDialogOpenChange(pOpen: boolean) {
    vPreviewDialogOpen = pOpen;

    if (!pOpen) {
      vPreviewCode = '';
      vPreviewLanguage = 'text';
    }
  }

  /**
   * Handles click events on preview buttons within HTML code blocks.
   * Opens a preview dialog with the rendered HTML content.
   * @param pEvent - The click event from the preview button
   */
  function fHandlePreviewClick(pEvent: Event) {
    pEvent.preventDefault();
    pEvent.stopPropagation();

    const cTarget = pEvent.currentTarget as HTMLButtonElement | null;

    if (!cTarget) {
      return;
    }

    const cInfo = fGetCodeInfoFromTarget(cTarget);

    if (!cInfo) {
      return;
    }

    vPreviewCode = cInfo.rawCode;
    vPreviewLanguage = cInfo.language;
    vPreviewDialogOpen = true;
  }

  /**
   * Processes markdown content into stable and unstable HTML blocks.
   * Uses incremental rendering: stable blocks are cached, unstable block is re-rendered.
   * Incomplete code blocks are rendered using SyntaxHighlightedCode to maintain interactivity.
   * @param pMarkdown - The raw markdown string to process
   */
  async function fProcessMarkdown(pMarkdown: string) {
    // Early exit if content unchanged (can happen with rapid coalescing)
    if (pMarkdown === vPreviousContent) {
      return;
    }

    if (!pMarkdown) {
      vRenderedBlocks = [];
      vUnstableBlockHtml = '';
      vIncompleteCodeBlock = null;
      vPreviousContent = '';
      return;
    }

    // Check for incomplete code block at the end of content
    const cIncompleteBlock = detectIncompleteCodeBlock(pMarkdown);

    if (cIncompleteBlock) {
      // Process only the prefix (content before the incomplete code block)
      const cPrefixMarkdown = pMarkdown.slice(0, cIncompleteBlock.openingIndex);

      if (cPrefixMarkdown.trim()) {
        const cNormalizedPrefix = preprocessLaTeX(cPrefixMarkdown);
        const cProcessorInstance = vProcessor();
        const cAst = cProcessorInstance.parse(cNormalizedPrefix) as MdastRoot;
        const cMdastChildren = (cAst as { children?: unknown[] }).children ?? [];
        const aNextBlocks: MarkdownBlock[] = [];

        // Check if we're in append mode for cache reuse
        const cAppendMode = fIsAppendMode(cPrefixMarkdown, vPreviousContent);
        const cPreviousBlockCount = cAppendMode ? vRenderedBlocks.length : 0;

        // All prefix blocks are now stable since code block is separate
        for (let vIndex = 0; vIndex < cMdastChildren.length; vIndex++) {
          const cChild = cMdastChildren[vIndex];

          // In append mode, reuse previous blocks if unchanged
          if (cAppendMode && vIndex < cPreviousBlockCount) {
            const cPrevBlock = vRenderedBlocks[vIndex];
            const cCurrentHash = fGetMdastNodeHash(cChild, vIndex);

            if (cPrevBlock?.contentHash === cCurrentHash) {
              aNextBlocks.push(cPrevBlock);

              continue;
            }
          }

          // Transform this block (with caching)
          const { html: cHtml, hash: cHash } = await fTransformMdastNode(
            cProcessorInstance,
            cChild,
            vIndex
          );
          const cId = fGetHastNodeId(
            { position: (cChild as { position?: unknown }).position } as HastRootContent,
            vIndex
          );

          aNextBlocks.push({ id: cId, html: cHtml, contentHash: cHash });
        }

        vRenderedBlocks = aNextBlocks;
      } else {
        vRenderedBlocks = [];
      }

      vPreviousContent = cPrefixMarkdown;
      vUnstableBlockHtml = '';
      vIncompleteCodeBlock = cIncompleteBlock;

      return;
    }

    // No incomplete code block - use standard processing
    vIncompleteCodeBlock = null;

    const cNormalized = preprocessLaTeX(pMarkdown);
    const cProcessorInstance = vProcessor();
    const cAst = cProcessorInstance.parse(cNormalized) as MdastRoot;
    const cMdastChildren = (cAst as { children?: unknown[] }).children ?? [];
    const cStableCount = Math.max(cMdastChildren.length - 1, 0);
    const aNextBlocks: MarkdownBlock[] = [];

    // Check if we're in append mode for cache reuse
    const cAppendMode = fIsAppendMode(pMarkdown, vPreviousContent);
    const cPreviousBlockCount = cAppendMode ? vRenderedBlocks.length : 0;

    for (let vIndex = 0; vIndex < cStableCount; vIndex++) {
      const cChild = cMdastChildren[vIndex];

      // In append mode, reuse previous blocks if unchanged
      if (cAppendMode && vIndex < cPreviousBlockCount) {
        const cPrevBlock = vRenderedBlocks[vIndex];
        const cCurrentHash = fGetMdastNodeHash(cChild, vIndex);
        if (cPrevBlock?.contentHash === cCurrentHash) {
          aNextBlocks.push(cPrevBlock);

          continue;
        }
      }

      // Transform this block (with caching)
      const { html: cHtml, hash: cHash } = await fTransformMdastNode(
        cProcessorInstance,
        cChild,
        vIndex
      );
      const cId = fGetHastNodeId(
        { position: (cChild as { position?: unknown }).position } as HastRootContent,
        vIndex
      );

      aNextBlocks.push({ id: cId, html: cHtml, contentHash: cHash });
    }

    let vUnstableHtml = '';

    if (cMdastChildren.length > cStableCount) {
      const cUnstableChild = cMdastChildren[cStableCount];
      const dSingleNodeRoot = { type: 'root', children: [cUnstableChild] };
      const cTransformedRoot = (await cProcessorInstance.run(
        dSingleNodeRoot as MdastRoot
      )) as HastRoot;

      vUnstableHtml = cProcessorInstance.stringify(cTransformedRoot);
    }

    vRenderedBlocks = aNextBlocks;
    vPreviousContent = pMarkdown;
    await tick(); // Force DOM sync before updating unstable HTML block
    vUnstableBlockHtml = vUnstableHtml;
  }

  /**
   * Attaches click event listeners to copy and preview buttons in code blocks.
   * Uses data-listener-bound attribute to prevent duplicate bindings.
   */
  function fSetupCodeBlockActions() {
    if (!vContainerRef) return;

    const cWrappers = vContainerRef.querySelectorAll<HTMLElement>('.code-block-wrapper');

    for (const cWrapper of cWrappers) {
      const cCopyButton = cWrapper.querySelector<HTMLButtonElement>('.copy-code-btn');
      const cPreviewButton = cWrapper.querySelector<HTMLButtonElement>('.preview-code-btn');

      if (cCopyButton && cCopyButton.dataset.listenerBound !== 'true') {
        cCopyButton.dataset.listenerBound = 'true';
        cCopyButton.addEventListener('click', fHandleCopyClick);
      }

      if (cPreviewButton && cPreviewButton.dataset.listenerBound !== 'true') {
        cPreviewButton.dataset.listenerBound = 'true';
        cPreviewButton.addEventListener('click', fHandlePreviewClick);
      }
    }
  }

  /**
   * Attaches error handlers to images to show fallback UI when loading fails (e.g., CORS).
   * Uses data-error-bound attribute to prevent duplicate bindings.
   */
  function fSetupImageErrorHandlers() {
    if (!vContainerRef) return;

    const cImages = vContainerRef.querySelectorAll<HTMLImageElement>(cImageNotErrorBoundSelector);

    for (const cImg of cImages) {
      cImg.dataset[cDataErrorBoundAttr] = cBoolTrueString;
      cImg.addEventListener('error', fHandleImageError);
    }
  }

  /**
   * Opens the mermaid diagram in a full-screen preview dialog with zoom/pan support.
   * Also handles copy and preview button clicks for mermaid blocks.
   * Uses event delegation: a single handler on the container.
   */
  async function fHandleMermaidClick(pEvent: MouseEvent) {
    const cTarget = pEvent.target as HTMLElement;

    // Toggle a diagram block between its rendered view and its source view.
    // Shared by mermaid and svg, css drives the visibility from the wrapper mode.
    const cToggleBtn = cTarget.closest(`.${cToggleSourceBtnClass}`);
    if (cToggleBtn) {
      pEvent.preventDefault();
      pEvent.stopPropagation();

      const cWrapper = cToggleBtn.closest(`.${cMermaidWrapperClass}, .${cSvgWrapperClass}`);
      if (!cWrapper) return;

      const cIsSource = cWrapper.getAttribute(cDiagramViewModeAttr) === cDiagramViewSource;
      const cNext = cIsSource ? cDiagramViewRendered : cDiagramViewSource;
      cWrapper.setAttribute(cDiagramViewModeAttr, cNext);
      cToggleBtn.setAttribute('aria-pressed', String(!cIsSource));
      return;
    }

    // Check if clicking on copy or preview button in mermaid block
    const cCopyBtn = cTarget.closest(`.${cMermaidWrapperClass} .copy-code-btn`);
    const cPreviewBtn = cTarget.closest(`.${cMermaidWrapperClass} .preview-code-btn`);

    if (cCopyBtn || cPreviewBtn) {
      const cWrapper = cTarget.closest(`.${cMermaidWrapperClass}`);
      if (!cWrapper) return;

      const cPreElement = cWrapper.querySelector<HTMLElement>(
        `pre.${cMermaidBlockClass}[${cMermaidSyntaxAttr}]`
      );
      if (!cPreElement) return;

      const cMermaidSyntax = cPreElement.getAttribute(cMermaidSyntaxAttr) ?? '';

      if (cCopyBtn) {
        pEvent.preventDefault();
        pEvent.stopPropagation();
        try {
          await copyToClipboard(cMermaidSyntax);
        } catch (vError) {
          console.error('Failed to copy mermaid syntax:', vError);
        }
        return;
      }

      if (cPreviewBtn) {
        pEvent.preventDefault();
        pEvent.stopPropagation();
        const cSvg = cPreElement.querySelector('svg');
        if (!cSvg) return;
        vMermaidPreviewSvgHtml = cSvg.outerHTML;
        vSvgPreviewLive = false;
        vMermaidPreviewOpen = true;
        return;
      }
    }

    // Check if clicking on copy or preview button in svg block
    const cSvgCopyBtn = cTarget.closest(`.${cSvgWrapperClass} .copy-code-btn`);
    const cSvgPreviewBtn = cTarget.closest(`.${cSvgWrapperClass} .preview-code-btn`);

    if (cSvgCopyBtn || cSvgPreviewBtn) {
      const cWrapper = cTarget.closest(`.${cSvgWrapperClass}`);
      if (!cWrapper) return;

      const cPreElement = cWrapper.querySelector<HTMLElement>(
        `pre.${cSvgBlockClass}[${cSvgSourceAttr}]`
      );
      if (!cPreElement) return;

      if (cSvgCopyBtn) {
        pEvent.preventDefault();
        pEvent.stopPropagation();
        try {
          await copyToClipboard(cPreElement.getAttribute(cSvgSourceAttr) ?? '');
        } catch (vError) {
          console.error('Failed to copy svg source:', vError);
        }
        return;
      }

      if (cSvgPreviewBtn) {
        pEvent.preventDefault();
        pEvent.stopPropagation();
        vMermaidPreviewSvgHtml = fSanitizeSvg(cPreElement.getAttribute(cSvgSourceAttr) ?? '');
        vSvgPreviewLive = false;
        vMermaidPreviewOpen = true;
        return;
      }
    }

    // A click on the header chrome targets the action buttons, never the
    // diagram. Guard so a header click can not fall through to the click to
    // zoom branches below, whatever the scroll position or stacking.
    if (cTarget.closest(`.${cCodeBlockHeaderClass}`)) return;

    // Open preview when clicking the svg block itself. A final block carries its
    // source, a streaming block does not and is mirrored live into the dialog.
    const cSvgEl = cTarget.closest(`.${cSvgBlockClass}`);
    if (cSvgEl) {
      const cSource = cSvgEl.getAttribute(cSvgSourceAttr);
      if (cSource !== null) {
        vMermaidPreviewSvgHtml = fSanitizeSvg(cSource);
        vSvgPreviewLive = false;
      } else {
        vSvgPreviewLive = true;
      }
      vMermaidPreviewOpen = true;
      return;
    }

    // Otherwise, open preview when clicking on the mermaid diagram itself
    const cMermaidEl = cTarget.closest(`.${cMermaidBlockClass}`);
    if (!cMermaidEl) return;

    const cSvg = cMermaidEl.querySelector('svg');
    if (!cSvg) return;

    vMermaidPreviewSvgHtml = cSvg.outerHTML;
    vSvgPreviewLive = false;
    vMermaidPreviewOpen = true;
  }

  /**
   * Handles mermaid preview dialog open state changes.
   * Cleans up SVG content when dialog is closed.
   */
  function fHandleMermaidPreviewOpenChange(pOpen: boolean) {
    vMermaidPreviewOpen = pOpen;
    if (!pOpen) {
      vMermaidPreviewSvgHtml = '';
      vSvgPreviewLive = false;
    }
  }

  /**
   * Renders mermaid diagrams that haven't been rendered yet.
   * Called after each markdown content update.
   * Marks nodes immediately to prevent duplicate renders during streaming.
   * Reads mode.current before await to ensure reactive tracking.
   */
  async function fRenderMermaidDiagrams() {
    if (!vContainerRef) return;

    const cNodes = vContainerRef.querySelectorAll(
      `pre.${cMermaidBlockClass}:not([${cMermaidRenderedAttr}])`
    );
    if (cNodes.length === 0) return;

    // Mark nodes immediately to prevent duplicate renders if called again during streaming.
    // This avoids needing a guard that would block node discovery.
    cNodes.forEach((pNode) => pNode.setAttribute(cMermaidRenderedAttr, 'true'));

    // Read mode before await so Svelte tracks it reactively.
    const cIsDark = mode.current === ColorMode.DARK;

    // lazy load the mermaid dependecy only when needed to reduce bundle size.
    const { default: cMermaid } = await import('mermaid');

    cMermaid.initialize({
      startOnLoad: false,
      theme: cIsDark ? 'dark' : 'default',
      securityLevel: 'strict',
      flowchart: {
        useMaxWidth: false,
        htmlLabels: true
      },
      sequence: {
        useMaxWidth: false
      },
      gantt: {
        useMaxWidth: false
      }
    });

    try {
      await cMermaid.run({
        nodes: Array.from(cNodes) as unknown as NodeListOf<HTMLElement>
      });
    } catch (vError) {
      console.error('Failed to render mermaid diagram:', vError);
    }
  }

  /**
   * Renders svg diagrams that haven't been rendered yet.
   * Sanitizes the source before injecting and marks each node so it renders once.
   * An empty sanitize result keeps the raw source as escaped text.
   */
  function fRenderSvgDiagrams() {
    if (!vContainerRef) return;

    const cNodes = vContainerRef.querySelectorAll<HTMLElement>(
      `pre.${cSvgBlockClass}:not([${cSvgRenderedAttr}])`
    );
    if (cNodes.length === 0) return;

    cNodes.forEach((pNode) => {
      pNode.setAttribute(cSvgRenderedAttr, 'true');

      const cSource = pNode.getAttribute(cSvgSourceAttr) ?? pNode.textContent ?? '';
      const cClean = fSanitizeSvg(cSource);

      if (cClean) {
        pNode.textContent = '';
        const cHost = document.createElement('div');
        pNode.appendChild(cHost);
        fMountSvgShadow(cHost, cClean, cSvgInlineShadowStyle);
      }
    });
  }

  /**
   * Handles image load errors by replacing the image with a fallback UI.
   * Shows a placeholder with a link to open the image in a new tab.
   */
  function fHandleImageError(pEvent: Event) {
    const cImg = pEvent.target as HTMLImageElement;
    if (!cImg || !cImg.src) return;

    // Don't handle data URLs or already-handled images
    if (
      cImg.src.startsWith(UrlProtocol.DATA) ||
      cImg.dataset[cDataErrorHandledAttr] === cBoolTrueString
    )
      return;
    cImg.dataset[cDataErrorHandledAttr] = cBoolTrueString;

    const cSrc = cImg.src;
    // Create fallback element
    const cFallback = document.createElement('div');
    cFallback.className = 'image-load-error';
    cFallback.appendChild(createImageErrorFallback(cSrc));

    // Replace image with fallback
    cImg.parentNode?.replaceChild(cFallback, cImg);
  }

  /**
   * Queues markdown for processing with coalescing support.
   * Only processes the latest markdown when multiple updates arrive quickly.
   * Uses requestAnimationFrame to yield to browser paint between batches.
   * @param pMarkdown - The markdown content to render
   */
  async function fUpdateRenderedBlocks(pMarkdown: string) {
    vPendingMarkdown = pMarkdown;

    if (vIsProcessing) {
      return;
    }

    vIsProcessing = true;
    let vFailedMarkdown = pMarkdown;

    try {
      while (vPendingMarkdown !== null) {
        const cNextMarkdown = vPendingMarkdown;
        vPendingMarkdown = null;
        vFailedMarkdown = cNextMarkdown;

        await fProcessMarkdown(cNextMarkdown);

        // Yield to browser for paint. During this, new chunks coalesce
        // into pendingMarkdown, so we always render the latest state.
        if (vPendingMarkdown !== null) {
          await new Promise((pResolve) => requestAnimationFrame(pResolve));
        }
      }
    } catch (vError) {
      // Fail closed: the pipeline that neutralizes literal HTML did not finish,
      // so the original Markdown is shown as escaped plain text, never as markup
      console.error('Failed to process markdown:', vError);
      vRenderedBlocks = [];
      vIncompleteCodeBlock = null;
      vPreviousContent = '';
      vUnstableBlockHtml = escapeHtml(vFailedMarkdown).replace(/\n/g, '<br>');
    } finally {
      vIsProcessing = false;
    }
  }

  $effect(() => {
    const cCurrentMode = mode.current;
    const cIsDark = cCurrentMode === ColorMode.DARK;

    fLoadHighlightTheme(cIsDark);
  });

  $effect(() => {
    fUpdateRenderedBlocks(vContent);
  });

  $effect(() => {
    const cHasRenderedBlocks = vRenderedBlocks.length > 0;
    const cHasUnstableBlock = Boolean(vUnstableBlockHtml);

    if ((cHasRenderedBlocks || cHasUnstableBlock) && vContainerRef) {
      fSetupCodeBlockActions();
      fSetupImageErrorHandlers();
      fRenderMermaidDiagrams();
      fRenderSvgDiagrams();
    }
  });

  // Auto-scroll for streaming code block
  $effect(() => {
    cStreamingAutoScroll.mSetContainer(vStreamingCodeScrollContainer);
  });

  $effect(() => {
    cStreamingAutoScroll.mUpdateInterval(vIncompleteCodeBlock !== null);
  });

  onDestroy(() => {
    fCleanupEventListeners();
    fCleanupHighlightTheme();
    cStreamingAutoScroll.mDestroy();
  });
</script>

<!-- svelte-ignore a11y_click_events_have_key_events -->
<!-- svelte-ignore a11y_no_static_element_interactions -->
<div
  bind:this={vContainerRef}
  onclick={fHandleMermaidClick}
  class="markdown-content {vClassName}{fConfig()[cSettingsKeys.FULL_HEIGHT_CODE_BLOCKS]
    ? ' full-height-code-blocks'
    : ''}"
>
  {#each vRenderedBlocks as block (block.id)}
    <div class="markdown-block" data-block-id={block.id}>
      {@html block.html}
    </div>
  {/each}

  {#if vUnstableBlockHtml}
    <div class="markdown-block markdown-block--unstable" data-block-id="unstable">
      <!-- eslint-disable-next-line no-at-html-tags -->
      {@html vUnstableBlockHtml}
    </div>
  {/if}

  {#if vIncompleteCodeBlock}
    {#if vIncompleteCodeBlock.language === cMermaidLanguage}
      <div class="mermaid-block-wrapper streaming-mermaid-block">
        <div class="code-block-header">
          <span class="code-language">{fTranslate('text_0fbccedd6152')}</span>
          <div class="code-block-actions">
            <ActionIconCopyToClipboard
              text={vIncompleteCodeBlock.code}
              canCopy={false}
              ariaLabel="Diagram incomplete"
            />
          </div>
        </div>
        <div class="mermaid-loading-placeholder">
          <span class="mermaid-loading-text">{fTranslate('text_f7f714a03395')}</span>
        </div>
      </div>
    {:else if cStreamingSvgCode !== null}
      <div class="svg-block-wrapper streaming-svg-block">
        <div class="code-block-header">
          <span class="code-language">{fTranslate('text_acdb1373d176')}</span>
          <div class="code-block-actions">
            <ActionIconCopyToClipboard
              text={vIncompleteCodeBlock.code}
              canCopy={false}
              ariaLabel="Diagram incomplete"
            />
          </div>
        </div>
        {#if cLiveSvgHtml}
          <div class="svg-scroll-container">
            <div class={cSvgBlockClass}>
              <div bind:this={vStreamingSvgHost}></div>
            </div>
          </div>
        {:else}
          <div class="mermaid-loading-placeholder">
            <span class="mermaid-loading-text">{fTranslate('text_6702e91a7860')}</span>
          </div>
        {/if}
      </div>
    {:else}
      <div class="code-block-wrapper streaming-code-block relative">
        <div class="code-block-header">
          <span class="code-language">{vIncompleteCodeBlock.language || 'text'}</span>
          <CodeBlockActions
            code={vIncompleteCodeBlock.code}
            language={vIncompleteCodeBlock.language || 'text'}
            disabled
            onPreview={(code, lang) => {
              vPreviewCode = code;
              vPreviewLanguage = lang;
              vPreviewDialogOpen = true;
            }}
          />
        </div>

        <div
          bind:this={vStreamingCodeScrollContainer}
          class="streaming-code-scroll-container"
          onscroll={() => cStreamingAutoScroll.mHandleScroll()}
        >
          <pre class="streaming-code-pre"><code
              class="hljs language-{vIncompleteCodeBlock.language || 'text'}"
              >{@html cStreamingCodeHtml}</code
            ></pre>
        </div>
      </div>
    {/if}
  {/if}
</div>

<DialogCodePreview
  open={vPreviewDialogOpen}
  code={vPreviewCode}
  language={vPreviewLanguage}
  onOpenChange={fHandlePreviewDialogOpenChange}
/>

<DialogMermaidPreview
  open={vMermaidPreviewOpen}
  svgHtml={vMermaidPreviewSvgHtml}
  onOpenChange={fHandleMermaidPreviewOpenChange}
/>

<style>
  @import './markdown-content.css';
</style>
