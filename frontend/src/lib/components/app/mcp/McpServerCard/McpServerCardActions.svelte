<script lang="ts">
  import { fTranslate } from '../../../../i18n';

  import { Trash2, RefreshCw, Pencil } from '@lucide/svelte';
  import { Button } from '$lib/components/ui/button';

  interface Props {
    isHealthChecking: boolean;
    onEdit: () => void;
    onRefresh: () => void;
    onDelete: () => void;
  }

  let {
    isHealthChecking: vIsHealthChecking,
    onEdit: vOnEdit,
    onRefresh: vOnRefresh,
    onDelete: vOnDelete
  }: Props = $props();
</script>

<div class="flex shrink-0 items-center gap-1">
  <Button
    variant="ghost"
    size="icon"
    class="h-7 w-7"
    onclick={vOnEdit}
    aria-label={fTranslate('text_464c4ffd019e')}
  >
    <Pencil class="h-3.5 w-3.5" />
  </Button>

  <Button
    variant="ghost"
    size="icon"
    class="h-7 w-7"
    onclick={vOnRefresh}
    disabled={vIsHealthChecking}
    aria-label={fTranslate('text_0e9161011702')}
  >
    <RefreshCw class="h-3.5 w-3.5" />
  </Button>

  <Button
    variant="ghost"
    size="icon"
    class="hover:text-destructive-foreground h-7 w-7 text-destructive hover:bg-destructive/10"
    onclick={vOnDelete}
    aria-label={fTranslate('text_e2d0a54968ea')}
  >
    <Trash2 class="h-3.5 w-3.5" />
  </Button>
</div>
