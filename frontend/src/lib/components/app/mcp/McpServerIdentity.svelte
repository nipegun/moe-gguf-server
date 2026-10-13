<script lang="ts">
  import { fTranslate } from '../../../i18n';

  import { ExternalLink } from '@lucide/svelte';
  import { Badge } from '$lib/components/ui/badge';
  import { McpLogo } from '$lib/components/app/mcp';
  import { TruncatedText } from '$lib/components/app/misc';
  import { sanitizeExternalUrl } from '$lib/utils';
  import type { MCPServerInfo } from '$lib/types';

  interface Props {
    displayName?: string;
    faviconUrl?: string | null;
    serverInfo?: MCPServerInfo;
    iconClass?: string;
    iconRounded?: string;
    showVersion?: boolean;
    showWebsite?: boolean;
    nameClass?: string;
  }

  let {
    displayName: vDisplayName,
    faviconUrl: vFaviconUrl = null,
    serverInfo: vServerInfo,
    iconClass: vIconClass = 'h-5 w-5',
    iconRounded: vIconRounded = 'rounded-sm',
    showVersion: vShowVersion = true,
    showWebsite: vShowWebsite = true,
    nameClass: vNameClass
  }: Props = $props();

  let vSafeWebsiteUrl = $derived(
    vServerInfo?.websiteUrl ? sanitizeExternalUrl(vServerInfo.websiteUrl) : null
  );
</script>

<span class="flex min-w-0 items-center gap-1.5">
  {#if vFaviconUrl}
    <img src={vFaviconUrl} alt="" class={['shrink-0 text-foreground', vIconRounded, vIconClass]} />
  {:else}
    <McpLogo class={['shrink-0 text-foreground', vIconRounded, vIconClass].join(' ')} />
  {/if}

  <TruncatedText text={vDisplayName ?? ''} class={vNameClass ?? ''} />

  {#if vShowVersion && vServerInfo?.version}
    <Badge variant="secondary" class="h-4 max-w-24 min-w-0 shrink px-1 text-[10px]">
      <TruncatedText text={`v${vServerInfo.version}`} />
    </Badge>
  {/if}

  {#if vShowWebsite && vSafeWebsiteUrl}
    <a
      href={vSafeWebsiteUrl}
      target="_blank"
      rel="noopener noreferrer"
      class="shrink-0 text-muted-foreground hover:text-foreground"
      aria-label={fTranslate('text_345173567bc6')}
      onclick={(e) => e.stopPropagation()}
    >
      <ExternalLink class="h-3 w-3" />
    </a>
  {/if}
</span>
