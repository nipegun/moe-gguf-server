<script lang="ts">
  import { fTranslate } from '../../../../../../i18n';

  import type { MCPPromptInfo } from '$lib/types';
  import ChatFormPromptPickerArgumentInput from './ChatFormPromptPickerArgumentInput.svelte';
  import { Button } from '$lib/components/ui/button';

  interface Props {
    prompt: MCPPromptInfo;
    promptArgs: Record<string, string>;
    suggestions: Record<string, string[]>;
    loadingSuggestions: Record<string, boolean>;
    activeAutocomplete: string | null;
    autocompleteIndex: number;
    promptError: string | null;
    onArgInput: (pArgName: string, pValue: string) => void;
    onArgKeydown: (pEvent: KeyboardEvent, pArgName: string) => void;
    onArgBlur: (pArgName: string) => void;
    onArgFocus: (pArgName: string) => void;
    onSelectSuggestion: (pArgName: string, pValue: string) => void;
    onSubmit: (pEvent: SubmitEvent) => void;
    onCancel: () => void;
  }

  let {
    prompt: vPrompt,
    promptArgs: vPromptArgs,
    suggestions: vSuggestions,
    loadingSuggestions: vLoadingSuggestions,
    activeAutocomplete: vActiveAutocomplete,
    autocompleteIndex: vAutocompleteIndex,
    promptError: vPromptError,
    onArgInput: vOnArgInput,
    onArgKeydown: vOnArgKeydown,
    onArgBlur: vOnArgBlur,
    onArgFocus: vOnArgFocus,
    onSelectSuggestion: vOnSelectSuggestion,
    onSubmit: vOnSubmit,
    onCancel: vOnCancel
  }: Props = $props();
</script>

<form onsubmit={vOnSubmit} class="space-y-3 pt-4">
  {#each vPrompt.arguments ?? [] as arg (arg.name)}
    <ChatFormPromptPickerArgumentInput
      argument={arg}
      value={vPromptArgs[arg.name] ?? ''}
      suggestions={vSuggestions[arg.name] ?? []}
      isLoadingSuggestions={vLoadingSuggestions[arg.name] ?? false}
      isAutocompleteActive={vActiveAutocomplete === arg.name}
      autocompleteIndex={vActiveAutocomplete === arg.name ? vAutocompleteIndex : 0}
      onInput={(value) => vOnArgInput(arg.name, value)}
      onKeydown={(e) => vOnArgKeydown(e, arg.name)}
      onBlur={() => vOnArgBlur(arg.name)}
      onFocus={() => vOnArgFocus(arg.name)}
      onSelectSuggestion={(value) => vOnSelectSuggestion(arg.name, value)}
    />
  {/each}

  {#if vPromptError}
    <div
      class="flex items-start gap-2 rounded-lg border border-destructive/50 bg-destructive/10 px-3 py-2 text-sm text-destructive"
      role="alert"
    >
      <span class="shrink-0">⚠</span>

      <span>{vPromptError}</span>
    </div>
  {/if}

  <div class="mt-8 flex justify-end gap-2">
    <Button type="button" size="sm" onclick={vOnCancel} variant="secondary"
      >{fTranslate('text_19766ed6ccb2')}</Button
    >

    <Button size="sm" type="submit">{fTranslate('text_0ce2c63c2535')}</Button>
  </div>
</form>
