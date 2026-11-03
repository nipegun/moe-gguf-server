<script>
  import logoMark from '$lib/assets/logo.svg?raw';
  let { class: vClassName = '', style: vStyle = '' } = $props();
</script>

<div class={vClassName} style={vStyle}>
  {@html logoMark}
</div>

<style>
  div :global(svg) {
    width: var(--size, 1rem);
    height: var(--size, 1rem);
  }
</style>
