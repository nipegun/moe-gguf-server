<script lang="ts">
  import { fTranslate } from '../../../i18n';

  import { Button } from '$lib/components/ui/button';
  import { Checkbox } from '$lib/components/ui/checkbox';
  import SearchInput from '$lib/components/app/forms/SearchInput.svelte';
  import { ScrollArea } from '$lib/components/ui/scroll-area';
  import { SvelteSet } from 'svelte/reactivity';
  import { fUseMarqueeSelection } from '$lib/hooks/use-marquee-selection.svelte';

  interface Props {
    conversations: DatabaseConversation[];
    messageCountMap?: Map<string, number>;
    mode: 'export' | 'import';
    onCancel: () => void;
    onConfirm: (pSelectedConversations: DatabaseConversation[]) => void;
    isOpen?: boolean;
  }

  let {
    conversations: vConversations,
    messageCountMap: vMessageCountMap = new Map(),
    mode: vMode,
    onCancel: vOnCancel,
    onConfirm: vOnConfirm,
    isOpen: vIsOpen = true
  }: Props = $props();

  let vSearchQuery = $state('');
  let vSelectedIds = $state.raw<SvelteSet<string>>(fGetInitialSelectedIds());

  function fGetInitialSelectedIds(): SvelteSet<string> {
    return new SvelteSet(vConversations.map((pC) => pC.id));
  }

  let vFilteredConversations = $derived(
    vConversations.filter((pConv) => {
      const cName = pConv.name || fTranslate('text_31d248c44579');
      return cName.toLowerCase().includes(vSearchQuery.toLowerCase());
    })
  );

  let vOrderedIds = $derived(vFilteredConversations.map((pC) => pC.id));

  let vAllSelected = $derived(
    vFilteredConversations.length > 0 &&
      vFilteredConversations.every((pConv) => vSelectedIds.has(pConv.id))
  );

  let vSomeSelected = $derived(
    vFilteredConversations.some((pConv) => vSelectedIds.has(pConv.id)) && !vAllSelected
  );

  const cMarquee = fUseMarqueeSelection({
    selectedIds: () => vSelectedIds,
    orderedIds: () => vOrderedIds,
    enabled: () => vIsOpen
  });

  function fToggleAll() {
    const cNewSet = new SvelteSet(vSelectedIds);
    if (vAllSelected) {
      vFilteredConversations.forEach((pConv) => cNewSet.delete(pConv.id));
    } else {
      vFilteredConversations.forEach((pConv) => cNewSet.add(pConv.id));
    }
    vSelectedIds = cNewSet;
  }

  function fHandleConfirm() {
    const cSelected = vConversations.filter((pConv) => vSelectedIds.has(pConv.id));
    vOnConfirm(cSelected);
  }

  function fHandleCancel() {
    vSelectedIds = fGetInitialSelectedIds();
    vSearchQuery = '';
    cMarquee.reset();

    vOnCancel();
  }

  export function fReset() {
    vSelectedIds = fGetInitialSelectedIds();
    vSearchQuery = '';
    cMarquee.reset();
  }
</script>

<div class="space-y-4">
  <SearchInput bind:value={vSearchQuery} placeholder={fTranslate('text_e90dd0173105')} />

  <div class="flex items-center justify-between text-sm text-muted-foreground">
    <span>
      {vSelectedIds.size}
      {fTranslate('text_28391d3bc64e')}
      {vConversations.length}
      {fTranslate('text_d7cbbb688b2e')}
      {#if vSearchQuery}
        ({vFilteredConversations.length} {fTranslate('text_db6ec1bfdac0')}
      {/if}
    </span>
  </div>

  <div class="overflow-hidden rounded-md border">
    <ScrollArea class="h-100">
      <table class="w-full">
        <thead class="sticky top-0 z-10 bg-muted">
          <tr class="border-b">
            <th class="w-12 p-3 text-left">
              <Checkbox
                checked={vAllSelected}
                indeterminate={vSomeSelected}
                onCheckedChange={fToggleAll}
              />
            </th>

            <th class="p-3 text-left text-sm font-medium">{fTranslate('text_ca6722cd00cf')}</th>

            <th class="w-32 p-3 text-left text-sm font-medium">{fTranslate('text_04d7b4833927')}</th
            >
          </tr>
        </thead>

        <tbody>
          {#if vFilteredConversations.length === 0}
            <tr>
              <td colspan="3" class="p-8 text-center text-sm text-muted-foreground">
                {#if vSearchQuery}
                  {fTranslate('text_82bf5deb2051')}{vSearchQuery}"
                {:else}
                  {fTranslate('text_5a484c8b8b23')}
                {/if}
              </td>
            </tr>
          {:else}
            {#each vFilteredConversations as conv (conv.id)}
              {@const checked = vSelectedIds.has(conv.id)}
              <tr
                class="cursor-pointer border-b transition-colors hover:bg-muted/50 {checked
                  ? 'bg-muted/75'
                  : ''}"
                data-conversation-row={conv.id}
                onmousedown={(event) => cMarquee.rowMouseDown(conv.id, event)}
                onclick={(event) => cMarquee.rowClick(conv.id, event.shiftKey)}
              >
                <td class="p-3">
                  <Checkbox
                    {checked}
                    onclick={(event) => {
                      event.preventDefault();
                      event.stopPropagation();
                      cMarquee.rowClick(conv.id, event.shiftKey);
                    }}
                  />
                </td>

                <td class="p-3 text-sm">
                  <div
                    class="max-w-68 truncate"
                    title={conv.name || fTranslate('text_31d248c44579')}
                  >
                    {conv.name || fTranslate('text_31d248c44579')}
                  </div>
                </td>

                <td class="p-3 text-sm text-muted-foreground">
                  {vMessageCountMap.get(conv.id) ?? 0}
                </td>
              </tr>
            {/each}
          {/if}
        </tbody>
      </table>
    </ScrollArea>
  </div>

  <div class="flex justify-end gap-2">
    <Button variant="outline" onclick={fHandleCancel}>{fTranslate('text_19766ed6ccb2')}</Button>

    <Button onclick={fHandleConfirm} disabled={vSelectedIds.size === 0}>
      {vMode === 'export' ? fTranslate('text_3664895579f0') : fTranslate('text_2cff9baabf56')} ({vSelectedIds.size})
    </Button>
  </div>
</div>
