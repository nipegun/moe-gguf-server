<script lang="ts">
  import { fTranslate } from '../../../../i18n';

  import { Download, Pin, PinOff, Trash2, X } from '@lucide/svelte';
  import { ActionIcon, DialogConfirmation } from '$lib/components/app';
  import { Checkbox } from '$lib/components/ui/checkbox';
  import { TooltipSide } from '$lib/enums';

  interface Props {
    class?: string;
    selectedCount: number;
    visibleCount: number;
    allVisibleSelected: boolean;
    someVisibleSelected: boolean;
    someSelectedPinned: boolean;
    pinStateIsMixed: boolean;
    onSelectAllToggle: () => void;
    onBulkPinToggle: () => void;
    onBulkExport: () => void;
    onBulkDelete: () => void;
    onClose: () => void;
  }

  let {
    class: vClassName = '',
    selectedCount: vSelectedCount,
    visibleCount: vVisibleCount,
    allVisibleSelected: vAllVisibleSelected,
    someVisibleSelected: vSomeVisibleSelected,
    someSelectedPinned: vSomeSelectedPinned,
    pinStateIsMixed: vPinStateIsMixed,
    onSelectAllToggle: vOnSelectAllToggle,
    onBulkPinToggle: vOnBulkPinToggle,
    onBulkExport: vOnBulkExport,
    onBulkDelete: vOnBulkDelete,
    onClose: vOnClose
  }: Props = $props();

  let vShowDeleteDialog = $state(false);

  function fHandleDeleteClick() {
    vShowDeleteDialog = true;
  }

  function fHandleDeleteConfirm() {
    vShowDeleteDialog = false;
    vOnBulkDelete();
  }

  function fHandleDeleteCancel() {
    vShowDeleteDialog = false;
  }

  const cHasSelection = $derived(vSelectedCount > 0);
  const cIsMasterChecked = $derived(vAllVisibleSelected);
  const cIsMasterIndeterminate = $derived(!vAllVisibleSelected && vSomeVisibleSelected);

  const cPinTooltip = $derived(
    cHasSelection
      ? vPinStateIsMixed
        ? fTranslate('text_4a59b87acd7e')
        : vSomeSelectedPinned
          ? vSelectedCount === 1
            ? fTranslate('text_ee3c71613054')
            : fTranslate('text_bb444562d616')
          : vSelectedCount === 1
            ? fTranslate('text_ff1cee744146')
            : fTranslate('text_81f50d740fb5')
      : fTranslate('text_ff1cee744146')
  );

  const cPinDisabled = $derived(!cHasSelection || vPinStateIsMixed);
</script>

<div
  role="toolbar"
  aria-label={fTranslate('text_5667f96ffb3d')}
  class="flex items-center gap-1.5 rounded-xl border border-border/50 bg-background/50 px-2 py-1.5 shadow-sm backdrop-blur-xl {vClassName}"
>
  <label class="flex min-w-0 cursor-pointer items-center gap-2">
    <Checkbox
      checked={cIsMasterChecked}
      indeterminate={cIsMasterIndeterminate}
      onCheckedChange={vOnSelectAllToggle}
      aria-label={cIsMasterChecked
        ? fTranslate('text_967549497036')
        : fTranslate('text_1fc9a387654d')}
    />

    <span class="truncate text-xs font-medium text-muted-foreground">
      {vSelectedCount} / {vVisibleCount}
      {fTranslate('text_d7cbbb688b2e')}
    </span>
  </label>

  <div class="ml-auto flex items-center gap-0.75">
    <ActionIcon
      icon={vSomeSelectedPinned ? PinOff : Pin}
      tooltip={cPinTooltip}
      tooltipSide={TooltipSide.TOP}
      disabled={cPinDisabled}
      ariaLabel={cPinTooltip}
      size="sm"
      iconSize="h-3.5 w-3.5"
      class="h-7 w-7 rounded-md bg-transparent backdrop-blur-none hover:bg-accent! {cPinDisabled
        ? 'cursor-not-allowed'
        : ''} {!cPinDisabled ? 'opacity-100' : 'opacity-40'}"
      onclick={vOnBulkPinToggle}
    />

    <ActionIcon
      icon={Download}
      tooltip={cHasSelection ? fTranslate('text_3664895579f0') : fTranslate('text_3664895579f0')}
      tooltipSide={TooltipSide.TOP}
      disabled={!cHasSelection}
      ariaLabel="Export selected"
      size="sm"
      iconSize="h-3.5 w-3.5"
      class="h-7 w-7 rounded-md bg-transparent backdrop-blur-none hover:bg-accent! {cHasSelection
        ? 'opacity-100'
        : 'opacity-40'}"
      onclick={vOnBulkExport}
    />

    <ActionIcon
      icon={Trash2}
      tooltip={fTranslate('text_d2ab5d46fed4')}
      tooltipSide={TooltipSide.TOP}
      disabled={!cHasSelection}
      ariaLabel="Delete selected"
      size="sm"
      iconSize="h-3.5 w-3.5 text-destructive"
      class="h-7 w-7 rounded-md bg-transparent backdrop-blur-none hover:bg-destructive/10! dark:hover:bg-destructive/20! disabled:hover:bg-transparent {cHasSelection
        ? 'opacity-100'
        : 'opacity-40'}"
      onclick={fHandleDeleteClick}
    />

    <div class="mx-1 h-4 w-px bg-border" aria-hidden="true"></div>

    <ActionIcon
      icon={X}
      tooltip={fTranslate('text_0a6603b65249')}
      tooltipSide={TooltipSide.TOP}
      ariaLabel="Exit bulk selection mode"
      size="sm"
      iconSize="h-3.5 w-3.5"
      class="h-7 w-7 rounded-md bg-transparent backdrop-blur-none hover:bg-accent!"
      onclick={vOnClose}
    />
  </div>
</div>

<DialogConfirmation
  bind:open={vShowDeleteDialog}
  title={fTranslate('text_d2e31b55c80b', {
    p0: vSelectedCount,
    p1: vSelectedCount === 1 ? '' : 's'
  })}
  description={fTranslate('text_abaa31087e83', {
    p0: vSelectedCount === 1 ? '' : 's',
    p1: vSelectedCount === 1 ? 'its' : 'their'
  })}
  confirmText={vSelectedCount === 1
    ? fTranslate('text_e2d0a54968ea')
    : fTranslate('text_0ae262c71c03', { p0: vSelectedCount })}
  cancelText={fTranslate('text_19766ed6ccb2')}
  variant="destructive"
  icon={Trash2}
  onConfirm={fHandleDeleteConfirm}
  onCancel={fHandleDeleteCancel}
/>
