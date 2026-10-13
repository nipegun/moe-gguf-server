<script lang="ts">
  import { fTranslate } from '../../../i18n';

  import { cIconClassDefault } from '$lib/constants/css-classes';
  import { Input } from '$lib/components/ui/input';
  import { Search, X } from '@lucide/svelte';

  interface Props {
    autofocus?: boolean;
    value?: string;
    placeholder?: string;
    onInput?: (pValue: string) => void;
    onClose?: () => void;
    onKeyDown?: (pEvent: KeyboardEvent) => void;
    class?: string;
    id?: string;
    ref?: HTMLInputElement | null;
    isCancelAlwaysVisible?: boolean;
  }

  let {
    autofocus: vAutofocus,
    value: vValue = $bindable(''),
    placeholder: vPlaceholder = fTranslate('text_7f55382219f0'),
    onInput: vOnInput,
    onClose: vOnClose,
    onKeyDown: vOnKeyDown,
    class: vClassName,
    id: vId,
    ref: vRef = $bindable(null),
    isCancelAlwaysVisible: vIsCancelAlwaysVisible = false
  }: Props = $props();

  let vShowClearButton = $derived(vIsCancelAlwaysVisible || !!vValue || !!vOnClose);

  function fHandleInput(pEvent: Event) {
    const cTarget = pEvent.target as HTMLInputElement;

    vValue = cTarget.value;
    vOnInput?.(cTarget.value);
  }

  function fHandleClear() {
    if (vValue) {
      vValue = '';
      vOnInput?.('');
      vRef?.focus({ preventScroll: true });
    } else {
      vOnClose?.();
    }
  }
</script>

<div class="relative {vClassName}">
  <Search
    class="absolute top-1/2 left-3 z-10 {cIconClassDefault} -translate-y-1/2 transform text-muted-foreground"
  />

  <Input
    autofocus={vAutofocus}
    id={vId}
    bind:value={vValue}
    bind:ref={vRef}
    class="pl-9 {vShowClearButton ? 'pr-9' : ''}"
    oninput={fHandleInput}
    onkeydown={vOnKeyDown}
    placeholder={vPlaceholder}
    type="search"
  />

  {#if vShowClearButton}
    <button
      type="button"
      class="absolute top-1/2 right-3 -translate-y-1/2 transform cursor-pointer text-muted-foreground transition-colors hover:text-foreground"
      onclick={fHandleClear}
      aria-label={vValue ? fTranslate('text_3b7ea51793e9') : fTranslate('text_7d9eb7acb13e')}
    >
      <X class={cIconClassDefault} />
    </button>
  {/if}
</div>
