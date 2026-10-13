<script lang="ts">
  import { fTranslate } from '../../../../i18n';

  import { cIsMobile } from '$lib/stores/viewport.svelte';
  import { autoResizeTextarea } from '$lib/utils';
  import { onMount } from 'svelte';

  interface Props {
    class?: string;
    disabled?: boolean;
    onInput?: () => void;
    onKeydown?: (pEvent: KeyboardEvent) => void;
    onPaste?: (pEvent: ClipboardEvent) => void;
    placeholder?: string;
    value?: string;
  }

  let {
    class: vClassName = '',
    disabled: vDisabled = false,
    onInput: vOnInput,
    onKeydown: vOnKeydown,
    onPaste: vOnPaste,
    placeholder: vPlaceholder = fTranslate('text_da0e101a67e7'),
    value: vValue = $bindable('')
  }: Props = $props();

  let vTextareaElement: HTMLTextAreaElement | undefined;

  onMount(() => {
    if (vTextareaElement) {
      autoResizeTextarea(vTextareaElement);
      vTextareaElement.focus();
    }
  });

  // Expose the textarea element for external access
  export function fGetElement() {
    return vTextareaElement;
  }

  export function fFocus() {
    if (cIsMobile.current) return;

    vTextareaElement?.focus({ preventScroll: true });
  }

  export function fResetHeight() {
    if (vTextareaElement) {
      vTextareaElement.style.height = '1rem';
    }
  }
</script>

<div class="flex-1 {vClassName}">
  <textarea
    dir="auto"
    bind:this={vTextareaElement}
    bind:value={vValue}
    class={[
      'text-md min-h-12 w-full resize-none border-0 bg-transparent p-0 leading-6 outline-none placeholder:text-muted-foreground focus-visible:ring-0 focus-visible:ring-offset-0',
      vDisabled && 'cursor-not-allowed'
    ]}
    style="max-height: var(--max-message-height);"
    disabled={vDisabled}
    onkeydown={vOnKeydown}
    oninput={(event) => {
      autoResizeTextarea(event.currentTarget);
      vOnInput?.();
    }}
    onpaste={vOnPaste}
    placeholder={vPlaceholder}
  ></textarea>
</div>
