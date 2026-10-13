<script lang="ts">
  import { fTranslate } from '../../../../i18n';

  import { SearchInput } from '$lib/components/app';

  interface Props {
    value?: string;
    placeholder?: string;
    onInput?: (pValue: string) => void;
    class?: string;
  }

  let {
    value: vValue = $bindable(''),
    placeholder: vPlaceholder = fTranslate('text_e90dd0173105'),
    onInput: vOnInput,
    class: vClassName
  }: Props = $props();
</script>

<div class="mb-4 px-2 {vClassName}">
  <SearchInput bind:value={vValue} placeholder={vPlaceholder} onInput={vOnInput} />
</div>
