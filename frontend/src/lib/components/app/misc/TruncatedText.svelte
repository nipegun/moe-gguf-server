<script lang="ts">
  import * as Tooltip from '$lib/components/ui/tooltip';

  interface Props {
    text: string;
    class?: string;
    showTooltip?: boolean;
  }

  let { text: vText, class: vClassName = '', showTooltip: vShowTooltip = true }: Props = $props();

  let vTextElement: HTMLSpanElement | undefined = $state();
  let vIsTruncated = $state(false);

  function fCheckTruncation() {
    if (vTextElement) {
      vIsTruncated = vTextElement.scrollWidth > vTextElement.clientWidth;
    }
  }

  $effect(() => {
    if (vTextElement) {
      fCheckTruncation();

      const cObserver = new ResizeObserver(fCheckTruncation);
      cObserver.observe(vTextElement);

      return () => cObserver.disconnect();
    }
  });
</script>

{#if vIsTruncated && vShowTooltip}
  <Tooltip.Root>
    <Tooltip.Trigger class="{vClassName} min-w-0">
      <span bind:this={vTextElement} class="block truncate">
        {vText}
      </span>
    </Tooltip.Trigger>

    <Tooltip.Content class="z-[9999]">
      <p>{vText}</p>
    </Tooltip.Content>
  </Tooltip.Root>
{:else}
  <span bind:this={vTextElement} class="{vClassName} block min-w-0 truncate">
    {vText}
  </span>
{/if}
