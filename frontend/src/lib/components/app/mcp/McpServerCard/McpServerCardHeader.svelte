<script lang="ts">
  import { Switch } from '$lib/components/ui/switch';
  import { Badge } from '$lib/components/ui/badge';
  import { McpCapabilitiesBadges, McpServerIdentity } from '$lib/components/app/mcp';
  import { dMcpTransportLabels, dMcpTransportIcons } from '$lib/constants';
  import { MCPTransportType } from '$lib/enums';
  import type { MCPServerInfo, MCPCapabilitiesInfo } from '$lib/types';

  interface Props {
    displayName: string;
    faviconUrl?: string | null;
    enabled: boolean;
    disabled?: boolean;
    onToggle: (pEnabled: boolean) => void;
    serverInfo?: MCPServerInfo;
    capabilities?: MCPCapabilitiesInfo;
    transportType?: MCPTransportType;
  }

  let {
    displayName: vDisplayName,
    faviconUrl: vFaviconUrl,
    enabled: vEnabled,
    disabled: vDisabled = false,
    onToggle: vOnToggle,
    serverInfo: vServerInfo,
    capabilities: vCapabilities,
    transportType: vTransportType
  }: Props = $props();
</script>

<div class="space-y-3">
  <div class="flex items-start justify-between gap-3">
    <div class="flex min-w-0 flex-col gap-3">
      <div class="inline-flex items-center gap-2">
        <McpServerIdentity
          displayName={vDisplayName}
          faviconUrl={vFaviconUrl}
          serverInfo={vServerInfo}
          iconClass="h-5 w-5"
          iconRounded="rounded"
          nameClass="leading-6 font-medium"
        />
      </div>

      {#if vCapabilities || vTransportType}
        <div class="flex flex-wrap items-center gap-1.5">
          {#if vTransportType}
            {@const TransportIcon = dMcpTransportIcons[vTransportType]}
            <Badge variant="outline" class="h-5 gap-1 px-1.5 text-[10px]">
              {#if TransportIcon}
                <TransportIcon class="h-3 w-3" />
              {/if}

              {dMcpTransportLabels[vTransportType] || vTransportType}
            </Badge>
          {/if}

          {#if vCapabilities}
            <McpCapabilitiesBadges capabilities={vCapabilities} />
          {/if}
        </div>
      {/if}
    </div>

    <div class="flex shrink-0 items-center pl-2">
      <Switch checked={vEnabled} disabled={vDisabled} onCheckedChange={vOnToggle} />
    </div>
  </div>
</div>
