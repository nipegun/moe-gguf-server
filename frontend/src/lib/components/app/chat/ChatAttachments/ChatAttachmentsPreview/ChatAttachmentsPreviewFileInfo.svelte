<script lang="ts">
  interface Props {
    displayName: string;
    fileSize: string;
  }

  let { displayName: vDisplayName, fileSize: vFileSize }: Props = $props();
</script>

<div class="sticky top-0 z-[20] mb-4 rounded-lg bg-black/5 px-4 py-2 text-center backdrop-blur-md">
  <p class="font-medium text-white">{vDisplayName}</p>

  {#if vFileSize}
    <p class="text-xs text-white/60">{vFileSize}</p>
  {/if}
</div>
