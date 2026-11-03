<script lang="ts">
  /**
   * Speech bubble with the two-digit number of a chat (01 to 99), so that it can be told apart in
   * the collapsed sidebar, where its title is not shown. A chat without a number (the 99 were
   * taken when it was created) shows two dots.
   */
  interface Props {
    number?: number;
    class?: string;
  }

  let { number: vNumber = 0, class: vClassName = 'h-5 w-5' }: Props = $props();

  let vLabel = $derived(vNumber > 0 ? String(vNumber).padStart(2, '0') : '··');
</script>

<svg
  class="shrink-0 {vClassName}"
  viewBox="0 0 24 24"
  fill="none"
  stroke="currentColor"
  stroke-width="1.8"
  stroke-linejoin="round"
  aria-hidden="true"
>
  <path d="M6 2.5h12a4 4 0 0 1 4 4v7a4 4 0 0 1-4 4h-7.5L5 21.5v-4a4 4 0 0 1-3-3.9v-7A4 4 0 0 1 6 2.5z" />
  <text
    x="12"
    y="10.2"
    fill="currentColor"
    stroke="none"
    font-size="10"
    font-weight="700"
    text-anchor="middle"
    dominant-baseline="central"
    font-family="ui-sans-serif, system-ui, sans-serif">{vLabel}</text
  >
</svg>
