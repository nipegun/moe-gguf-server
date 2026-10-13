<script lang="ts">
  import { fTranslate } from '../../../../../../i18n';

  interface Props {
    currentItem: { name?: string } | null;
    displayPreview: string | undefined;
  }

  let { currentItem: vCurrentItem, displayPreview: vDisplayPreview }: Props = $props();
</script>

{#if vDisplayPreview}
  <div class="flex flex-1 items-center justify-center">
    <img
      src={vDisplayPreview}
      alt={vCurrentItem?.name || fTranslate('text_5975cf1bba43')}
      class="max-h-[80vh] max-w-[80vw] rounded-lg object-contain shadow-lg"
    />
  </div>
{/if}
