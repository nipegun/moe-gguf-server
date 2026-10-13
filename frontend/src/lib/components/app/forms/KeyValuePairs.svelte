<script lang="ts">
  import { fTranslate } from '../../../i18n';

  import { tick } from 'svelte';
  import { Plus, Trash2 } from '@lucide/svelte';
  import { Input } from '$lib/components/ui/input';
  import {
    autoResizeTextarea,
    sanitizeKeyValuePairKey,
    sanitizeKeyValuePairValue
  } from '$lib/utils';
  import { cKeyValuePairKeyMaxLength, cKeyValuePairValueMaxLength } from '$lib/constants';
  import type { KeyValuePair } from '$lib/types';

  interface Props {
    class?: string;
    pairs: KeyValuePair[];
    onPairsChange: (pPairs: KeyValuePair[]) => void;
    keyPlaceholder?: string;
    valuePlaceholder?: string;
    addButtonLabel?: string;
    emptyMessage?: string;
    sectionLabel?: string;
    sectionLabelOptional?: boolean;
  }

  let {
    class: vClassName = '',
    pairs: vPairs,
    onPairsChange: vOnPairsChange,
    keyPlaceholder: vKeyPlaceholder = 'Key',
    valuePlaceholder: vValuePlaceholder = 'Value',
    addButtonLabel: vAddButtonLabel = 'Add',
    emptyMessage: vEmptyMessage = fTranslate('text_b7023ab8ed18'),
    sectionLabel: vSectionLabel,
    sectionLabelOptional: vSectionLabelOptional = true
  }: Props = $props();

  // Pre-allocate the ref array so `bind:ref={keyInputRefs[index]}` never reads `undefined`
  // for in-range indices; the $effect below keeps it in sync when `pairs` grows.
  // svelte-ignore state_referenced_locally
  let aKeyInputRefs: (HTMLInputElement | null)[] = $state(vPairs.map(() => null));

  async function fAddPair() {
    // Capture the target index before mutating so deletions earlier in the
    // list can't make keyInputRefs.length drift past the newly-appended row.
    const cNewIndex = vPairs.length;
    vOnPairsChange([...vPairs, { key: '', value: '' }]);
    await tick();
    aKeyInputRefs[cNewIndex]?.focus();
  }

  function fRemovePair(pIndex: number) {
    vOnPairsChange(vPairs.filter((pIgnored, pI) => pI !== pIndex));
  }

  function fUpdatePairKey(pIndex: number, pRawKey: string) {
    const cKey = sanitizeKeyValuePairKey(pRawKey);
    const aNewPairs = [...vPairs];

    aNewPairs[pIndex] = { ...aNewPairs[pIndex], key: cKey };
    vOnPairsChange(aNewPairs);
  }

  function fTrimPairKey(pIndex: number, pKey: string) {
    const cTrimmed = pKey.trim();
    if (cTrimmed === pKey) return;

    const aNewPairs = [...vPairs];

    aNewPairs[pIndex] = { ...aNewPairs[pIndex], key: cTrimmed };
    vOnPairsChange(aNewPairs);
  }

  function fUpdatePairValue(pIndex: number, pRawValue: string) {
    const cValue = sanitizeKeyValuePairValue(pRawValue);
    const aNewPairs = [...vPairs];

    aNewPairs[pIndex] = { ...aNewPairs[pIndex], value: cValue };
    vOnPairsChange(aNewPairs);
  }

  function fTrimPairValue(pIndex: number, pValue: string) {
    const cTrimmed = pValue.trim();
    if (cTrimmed === pValue) return;

    const aNewPairs = [...vPairs];

    aNewPairs[pIndex] = { ...aNewPairs[pIndex], value: cTrimmed };
    vOnPairsChange(aNewPairs);
  }

  // Keep keyInputRefs aligned with pairs length so bind:ref never sees `undefined`.
  // $effect.pre runs during traversal in tree order, before the {#each} block re-renders,
  // so newly-appended items always have a defined slot when their binding is set up.
  $effect.pre(() => {
    while (aKeyInputRefs.length < vPairs.length) {
      aKeyInputRefs.push(null);
    }
  });
</script>

<div class={vClassName}>
  <div class="mb-2 flex items-center justify-between">
    {#if vSectionLabel}
      <span class="text-xs font-medium select-none">
        {vSectionLabel}
        {#if vSectionLabelOptional}
          <span class="text-muted-foreground">{fTranslate('text_0059798b7f70')}</span>
        {/if}
      </span>
    {/if}

    <button
      type="button"
      class="inline-flex cursor-pointer items-center gap-1 rounded-md px-1.5 py-1 text-xs text-muted-foreground hover:bg-muted hover:text-foreground"
      onclick={fAddPair}
    >
      <Plus class="h-3 w-3" />
      {vAddButtonLabel}
    </button>
  </div>

  {#if vPairs.length > 0}
    <div class="space-y-3">
      {#each vPairs as pair, index (index)}
        <div class="flex items-start gap-2">
          <Input
            bind:ref={aKeyInputRefs[index]}
            type="text"
            placeholder={vKeyPlaceholder}
            value={pair.key}
            maxlength={cKeyValuePairKeyMaxLength}
            oninput={(e) => fUpdatePairKey(index, e.currentTarget.value)}
            onblur={(e) => fTrimPairKey(index, e.currentTarget.value)}
            class="flex-1"
          />

          <textarea
            use:autoResizeTextarea
            placeholder={vValuePlaceholder}
            value={pair.value}
            maxlength={cKeyValuePairValueMaxLength}
            oninput={(e) => {
              fUpdatePairValue(index, e.currentTarget.value);
              autoResizeTextarea(e.currentTarget);
            }}
            onblur={(e) => fTrimPairValue(index, e.currentTarget.value)}
            class="flex-1 resize-none rounded-md border border-input bg-transparent px-3 py-2 text-sm leading-5 placeholder:text-muted-foreground focus-visible:ring-1 focus-visible:ring-ring focus-visible:outline-none"
            rows="1"
          ></textarea>

          <button
            type="button"
            class="mt-1.5 shrink-0 cursor-pointer rounded-md p-1 text-muted-foreground hover:bg-destructive/10 hover:text-destructive"
            onclick={() => fRemovePair(index)}
            aria-label={fTranslate('text_5a89edf2867d')}
          >
            <Trash2 class="h-3.5 w-3.5" />
          </button>
        </div>
      {/each}
    </div>
  {:else}
    <p class="select-none text-xs text-muted-foreground">{vEmptyMessage}</p>
  {/if}
</div>
