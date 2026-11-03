<script lang="ts">
  import { fTranslate } from '../../../../../i18n';

  import { Loader2 } from '@lucide/svelte';
  import { Button } from '$lib/components/ui/button';

  interface Props {
    modelId: string | null;
    isLoading: boolean;
    onLoad: () => void;
  }

  let { modelId: vModelId, isLoading: vIsLoading, onLoad: vOnLoad }: Props = $props();
</script>

{#if vModelId !== null && !vIsLoading}
  <div class="flex flex-col gap-2 border-t border-border/50 pt-2 text-xs text-muted-foreground">
    <span>{fTranslate('text_2a90ea4a5f0a')}</span>
    <Button size="sm" variant="secondary" class="self-start" onclick={vOnLoad}
      >{fTranslate('text_889fad74ece5')}</Button
    >
  </div>
{:else if vIsLoading}
  <div class="flex items-center gap-2 border-t border-border/50 pt-2 text-xs text-muted-foreground">
    <Loader2 class="h-3.5 w-3.5 animate-spin" />
    <span>{fTranslate('text_fae76fba2b4f')}</span>
  </div>
{/if}
