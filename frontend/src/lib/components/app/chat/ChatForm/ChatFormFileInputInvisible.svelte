<script lang="ts">
  interface Props {
    class?: string;
    multiple?: boolean;
    onFileSelect?: (pFiles: File[]) => void;
  }

  let {
    class: vClassName = '',
    multiple: vMultiple = true,
    onFileSelect: vOnFileSelect
  }: Props = $props();

  let vFileInputElement: HTMLInputElement | undefined;

  export function fClick() {
    vFileInputElement?.click();
  }

  function fHandleFileSelect(pEvent: Event) {
    const cInput = pEvent.target as HTMLInputElement;

    if (cInput.files) {
      vOnFileSelect?.(Array.from(cInput.files));
    }
  }
</script>

<input
  bind:this={vFileInputElement}
  type="file"
  multiple={vMultiple}
  onchange={fHandleFileSelect}
  class="hidden {vClassName}"
/>
