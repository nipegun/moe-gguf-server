<script lang="ts">
  import { fTranslate } from '../../../i18n';

  import { Button } from '$lib/components/ui/button';
  import { InputWithSuggestions } from '$lib/components/app';
  import { KeyboardKey } from '$lib/enums';
  import { cMcpStore } from '$lib/stores/mcp.svelte';
  import { cMinAutocompleteInputLength } from '$lib/constants';
  import type { MCPResourceTemplateInfo } from '$lib/types';
  import {
    debounce,
    extractTemplateVariables,
    expandTemplate,
    isTemplateComplete
  } from '$lib/utils';

  interface Props {
    template: MCPResourceTemplateInfo;
    onResolve: (pUri: string, pServerName: string) => void;
    onCancel: () => void;
  }

  let { template: vTemplate, onResolve: vOnResolve, onCancel: vOnCancel }: Props = $props();

  const cVariables = $derived(extractTemplateVariables(vTemplate.uriTemplate));

  let vValues = $state<Record<string, string>>({});
  let vSuggestions = $state<Record<string, string[]>>({});
  let vLoadingSuggestions = $state<Record<string, boolean>>({});
  let vActiveAutocomplete = $state<string | null>(null);
  let vAutocompleteIndex = $state(0);

  const cExpandedUri = $derived(expandTemplate(vTemplate.uriTemplate, vValues));
  const cIsComplete = $derived(isTemplateComplete(vTemplate.uriTemplate, vValues));

  const cFetchCompletions = debounce(async (pArgName: string, pValue: string) => {
    if (pValue.length < 1) {
      vSuggestions[pArgName] = [];

      return;
    }

    vLoadingSuggestions[pArgName] = true;

    try {
      const cResult = await cMcpStore.mGetResourceCompletions(
        vTemplate.serverName,
        vTemplate.uriTemplate,
        pArgName,
        pValue
      );

      if (cResult && cResult.values.length > 0) {
        const cFilteredValues = cResult.values.filter((pV) => pV.trim() !== '');

        if (cFilteredValues.length > 0) {
          vSuggestions[pArgName] = cFilteredValues;
          vActiveAutocomplete = pArgName;
          vAutocompleteIndex = 0;
        } else {
          vSuggestions[pArgName] = [];
        }
      } else {
        vSuggestions[pArgName] = [];
      }
    } catch (vError) {
      console.error('[McpResourceTemplateForm] Failed to fetch completions:', vError);
      vSuggestions[pArgName] = [];
    } finally {
      vLoadingSuggestions[pArgName] = false;
    }
  }, 200);

  function fHandleArgInput(pArgName: string, pValue: string) {
    vValues[pArgName] = pValue;
    cFetchCompletions(pArgName, pValue);
  }

  function fSelectSuggestion(pArgName: string, pValue: string) {
    vValues[pArgName] = pValue;
    vSuggestions[pArgName] = [];
    vActiveAutocomplete = null;
  }

  function fHandleArgKeydown(pEvent: KeyboardEvent, pArgName: string) {
    const cArgSuggestions = vSuggestions[pArgName] ?? [];

    if (pEvent.key === KeyboardKey.ESCAPE) {
      pEvent.preventDefault();
      pEvent.stopPropagation();

      if (cArgSuggestions.length > 0 && vActiveAutocomplete === pArgName) {
        vSuggestions[pArgName] = [];
        vActiveAutocomplete = null;
      } else {
        vOnCancel();
      }

      return;
    }

    if (cArgSuggestions.length === 0 || vActiveAutocomplete !== pArgName) return;

    if (pEvent.key === KeyboardKey.ARROW_DOWN) {
      pEvent.preventDefault();
      vAutocompleteIndex = Math.min(vAutocompleteIndex + 1, cArgSuggestions.length - 1);
    } else if (pEvent.key === KeyboardKey.ARROW_UP) {
      pEvent.preventDefault();
      vAutocompleteIndex = Math.max(vAutocompleteIndex - 1, 0);
    } else if (pEvent.key === KeyboardKey.ENTER && cArgSuggestions[vAutocompleteIndex]) {
      pEvent.preventDefault();
      pEvent.stopPropagation();
      fSelectSuggestion(pArgName, cArgSuggestions[vAutocompleteIndex]);
    }
  }

  function fHandleArgBlur(pArgName: string) {
    setTimeout(() => {
      if (vActiveAutocomplete === pArgName) {
        vSuggestions[pArgName] = [];
        vActiveAutocomplete = null;
      }
    }, 150);
  }

  function fHandleArgFocus(pArgName: string) {
    const cValue = vValues[pArgName] ?? '';

    if (cValue.length >= cMinAutocompleteInputLength) {
      cFetchCompletions(pArgName, cValue);
    }
  }

  function fHandleSubmit(pEvent: SubmitEvent) {
    pEvent.preventDefault();

    if (cIsComplete) {
      vOnResolve(cExpandedUri, vTemplate.serverName);
    }
  }
</script>

<form onsubmit={fHandleSubmit} class="space-y-3">
  {#each cVariables as variable (variable.name)}
    <InputWithSuggestions
      name={variable.name}
      value={vValues[variable.name] ?? ''}
      suggestions={vSuggestions[variable.name] ?? []}
      isLoadingSuggestions={vLoadingSuggestions[variable.name] ?? false}
      isAutocompleteActive={vActiveAutocomplete === variable.name}
      autocompleteIndex={vActiveAutocomplete === variable.name ? vAutocompleteIndex : 0}
      onInput={(value) => fHandleArgInput(variable.name, value)}
      onKeydown={(e) => fHandleArgKeydown(e, variable.name)}
      onBlur={() => fHandleArgBlur(variable.name)}
      onFocus={() => fHandleArgFocus(variable.name)}
      onSelectSuggestion={(value) => fSelectSuggestion(variable.name, value)}
    />
  {/each}

  {#if cIsComplete}
    <div class="rounded-md bg-muted/50 px-3 py-2">
      <p class="text-xs text-muted-foreground">{fTranslate('text_7509c7981d15')}</p>

      <p class="mt-0.5 font-mono text-xs break-all">{cExpandedUri}</p>
    </div>
  {/if}

  <div class="flex justify-end gap-2 pt-1">
    <Button type="button" size="sm" variant="secondary" onclick={vOnCancel}
      >{fTranslate('text_19766ed6ccb2')}</Button
    >

    <Button size="sm" type="submit" disabled={!cIsComplete}
      >{fTranslate('text_65f6015df981')}</Button
    >
  </div>
</form>
