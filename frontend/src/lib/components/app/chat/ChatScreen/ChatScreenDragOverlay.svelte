<script>
  import { fTranslate } from '../../../../i18n';

  import { Upload } from '@lucide/svelte';
</script>

<div
  class="pointer-events-none fixed inset-0 z-50 flex items-center justify-center bg-black/50 backdrop-blur-sm"
>
  <div
    class="flex flex-col items-center justify-center rounded-2xl border-2 border-dashed border-border bg-background p-12 shadow-lg"
  >
    <Upload class="mb-4 h-12 w-12 text-muted-foreground" />

    <p class="text-lg font-medium text-foreground">{fTranslate('text_21298c62c8e5')}</p>

    <p class="text-sm text-muted-foreground">{fTranslate('text_3c53fc7884f1')}</p>
  </div>
</div>
