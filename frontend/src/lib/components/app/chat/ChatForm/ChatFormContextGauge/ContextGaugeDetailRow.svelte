<script lang="ts">
  interface Props {
    label: string;
    value: string;
    subtitle?: string;
  }

  let { label: vLabel, value: vValue, subtitle: vSubtitle }: Props = $props();
</script>

<div class="grid gap-1.5">
  <div class="flex items-baseline justify-between">
    <span class="text-muted-foreground">{vLabel}</span>
    <span class="font-mono text-muted-foreground">{vValue}</span>
  </div>

  {#if vSubtitle}
    <div class="text-[10px] leading-tight text-muted-foreground/70">{vSubtitle}</div>
  {/if}
</div>
