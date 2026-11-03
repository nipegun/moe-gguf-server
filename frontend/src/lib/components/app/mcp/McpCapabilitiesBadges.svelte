<script lang="ts">
  import { fTranslate } from '../../../i18n';

  import { Wrench, Database, MessageSquare, FileText, Sparkles, ListChecks } from '@lucide/svelte';
  import type { MCPCapabilitiesInfo } from '$lib/types';
  import { Badge } from '$lib/components/ui/badge';

  interface Props {
    capabilities?: MCPCapabilitiesInfo;
  }

  let { capabilities: vCapabilities }: Props = $props();
</script>

{#if vCapabilities}
  {#if vCapabilities.server.tools}
    <Badge variant="outline" class="h-5 gap-1 bg-green-50 px-1.5 text-[10px] dark:bg-green-950">
      <Wrench class="h-3 w-3 text-green-600 dark:text-green-400" />
      {fTranslate('text_ea93d6a262ec')}
    </Badge>
  {/if}

  {#if vCapabilities.server.resources}
    <Badge variant="outline" class="h-5 gap-1 bg-blue-50 px-1.5 text-[10px] dark:bg-blue-950">
      <Database class="h-3 w-3 text-blue-600 dark:text-blue-400" />
      {fTranslate('text_e89b30aa1dc3')}
    </Badge>
  {/if}

  {#if vCapabilities.server.prompts}
    <Badge variant="outline" class="h-5 gap-1 bg-purple-50 px-1.5 text-[10px] dark:bg-purple-950">
      <MessageSquare class="h-3 w-3 text-purple-600 dark:text-purple-400" />
      {fTranslate('text_5f2e66e65397')}
    </Badge>
  {/if}

  {#if vCapabilities.server.logging}
    <Badge variant="outline" class="h-5 gap-1 bg-orange-50 px-1.5 text-[10px] dark:bg-orange-950">
      <FileText class="h-3 w-3 text-orange-600 dark:text-orange-400" />
      {fTranslate('text_d3ef01b4a9c9')}
    </Badge>
  {/if}

  {#if vCapabilities.server.completions}
    <Badge variant="outline" class="h-5 gap-1 bg-cyan-50 px-1.5 text-[10px] dark:bg-cyan-950">
      <Sparkles class="h-3 w-3 text-cyan-600 dark:text-cyan-400" />
      {fTranslate('text_ab1451c522d1')}
    </Badge>
  {/if}

  {#if vCapabilities.server.tasks}
    <Badge variant="outline" class="h-5 gap-1 bg-pink-50 px-1.5 text-[10px] dark:bg-pink-950">
      <ListChecks class="h-3 w-3 text-pink-600 dark:text-pink-400" />
      {fTranslate('text_b3a60e61a523')}
    </Badge>
  {/if}
{/if}
