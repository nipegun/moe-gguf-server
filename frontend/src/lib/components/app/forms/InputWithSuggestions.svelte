<script lang="ts">
  import { fTranslate } from '../../../i18n';

  import { fly } from 'svelte/transition';
  import { Input } from '$lib/components/ui/input';
  import { Label } from '$lib/components/ui/label';

  interface Props {
    name: string;
    value: string;
    suggestions?: string[];
    isLoadingSuggestions?: boolean;
    isAutocompleteActive?: boolean;
    autocompleteIndex?: number;
    onInput: (pValue: string) => void;
    onKeydown: (pEvent: KeyboardEvent) => void;
    onBlur: () => void;
    onFocus: () => void;
    onSelectSuggestion: (pValue: string) => void;
  }

  let {
    name: vName,
    value: vValue = '',
    suggestions: vSuggestions = [],
    isLoadingSuggestions: vIsLoadingSuggestions = false,
    isAutocompleteActive: vIsAutocompleteActive = false,
    autocompleteIndex: vAutocompleteIndex = 0,
    onInput: vOnInput,
    onKeydown: vOnKeydown,
    onBlur: vOnBlur,
    onFocus: vOnFocus,
    onSelectSuggestion: vOnSelectSuggestion
  }: Props = $props();
</script>

<div class="relative grid gap-1">
  <Label for="tpl-arg-{vName}" class="mb-1 text-muted-foreground">
    <span>
      {vName}

      <span class="text-destructive">*</span>
    </span>

    {#if vIsLoadingSuggestions}
      <span class="text-xs text-muted-foreground/50">...</span>
    {/if}
  </Label>

  <Input
    id="tpl-arg-{vName}"
    type="text"
    value={vValue}
    oninput={(e) => vOnInput(e.currentTarget.value)}
    onkeydown={vOnKeydown}
    onblur={vOnBlur}
    onfocus={vOnFocus}
    placeholder={fTranslate('text_6675a0593a90', { p0: vName })}
    autocomplete="off"
  />

  {#if vIsAutocompleteActive && vSuggestions.length > 0}
    <div
      class="absolute top-full right-0 left-0 z-10 mt-1 max-h-32 overflow-y-auto rounded-lg border border-border/50 bg-background shadow-lg"
      transition:fly={{ y: -5, duration: 100 }}
    >
      {#each vSuggestions as suggestion, i (suggestion)}
        <button
          type="button"
          onmousedown={() => vOnSelectSuggestion(suggestion)}
          class="w-full px-3 py-1.5 text-left text-sm hover:bg-accent {i === vAutocompleteIndex
            ? 'bg-accent'
            : ''}"
        >
          {suggestion}
        </button>
      {/each}
    </div>
  {/if}
</div>
