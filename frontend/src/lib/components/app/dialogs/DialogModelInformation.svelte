<script lang="ts">
  import { fTranslate } from '../../../i18n';

  import * as Dialog from '$lib/components/ui/dialog';
  import * as Table from '$lib/components/ui/table';
  import { BadgesModality, ActionIconCopyToClipboard } from '$lib/components/app';
  import { cServerStore } from '$lib/stores/server.svelte';
  import { cModelsStore, fModelOptions, fModelsLoading } from '$lib/stores/models.svelte';
  import { formatFileSize, formatParameters, formatNumber } from '$lib/utils';
  import type { ApiMgsServerProps } from '$lib/types';

  interface Props {
    open?: boolean;
    onOpenChange?: (pOpen: boolean) => void;
    // when set, fetch props from the child process (router mode)
    modelId?: string | null;
  }

  let {
    open: vOpen = $bindable(),
    onOpenChange: vOnOpenChange,
    modelId: vModelId = null
  }: Props = $props();

  let vIsRouter = $derived(cServerStore.mIsRouterMode);

  // per-model props fetched from the child process
  let vRouterModelProps = $state<ApiMgsServerProps | null>(null);
  let vIsLoadingRouterProps = $state(false);

  // in router mode use per-model props, otherwise use global props
  let vServerProps = $derived(vIsRouter && vModelId ? vRouterModelProps : cServerStore.vProps);

  let vModelName = $derived(vIsRouter && vModelId ? vModelId : cModelsStore.mSingleModelName);
  let vModels = $derived(fModelOptions());
  let vIsLoadingModels = $derived(fModelsLoading());

  // in router mode, find the model option matching modelId
  // in single mode, use the first model as before
  let vFirstModel = $derived.by(() => {
    if (vIsRouter && vModelId) {
      return vModels.find((pM) => pM.model === vModelId) ?? null;
    }
    return vModels[0] ?? null;
  });

  // Get modalities from modelStore using the model ID from the first model
  let vModalities = $derived.by(() => {
    if (!vFirstModel?.id) return [];
    return cModelsStore.mGetModelModalitiesArray(vFirstModel.id);
  });

  // Ensure models are fetched when dialog opens
  $effect(() => {
    if (vOpen && vModels.length === 0) {
      cModelsStore.mFetch();
    }
  });

  // fetch per-model props from child process when dialog opens in router mode
  $effect(() => {
    if (vOpen && vIsRouter && vModelId) {
      vIsLoadingRouterProps = true;
      cModelsStore
        .mFetchModelProps(vModelId)
        .then((pProps) => {
          vRouterModelProps = pProps;
        })
        .catch(() => {
          vRouterModelProps = null;
        })
        .finally(() => {
          vIsLoadingRouterProps = false;
        });
    }
    if (!vOpen) {
      vRouterModelProps = null;
    }
  });
</script>

<Dialog.Root bind:open={vOpen} onOpenChange={vOnOpenChange}>
  <Dialog.Content class="@container z-9999 !max-h-[80dvh] !max-w-[60rem] max-w-full">
    <style>
      @container (max-width: 56rem) {
        .resizable-text-container {
          max-width: calc(100vw - var(--threshold));
        }
      }
    </style>

    <Dialog.Header>
      <Dialog.Title>{fTranslate('text_6d67a9d8ae24')}</Dialog.Title>

      <Dialog.Description>{fTranslate('text_d14c75d7d3b8')}</Dialog.Description>
    </Dialog.Header>

    <div class="space-y-6 py-4">
      {#if vIsLoadingModels || vIsLoadingRouterProps}
        <div class="flex items-center justify-center py-8">
          <div class="text-sm text-muted-foreground">{fTranslate('text_68ad292b7736')}</div>
        </div>
      {:else if vFirstModel}
        {@const modelMeta = vFirstModel.meta}

        {#if vServerProps}
          <Table.Root>
            <Table.Header>
              <Table.Row>
                <Table.Head class="w-[10rem]">{fTranslate('text_5e2c614c23f0')}</Table.Head>

                <Table.Head>
                  <div class="inline-flex items-center gap-2">
                    <span
                      class="resizable-text-container min-w-0 flex-1 truncate"
                      style:--threshold="12rem"
                    >
                      {vModelName}
                    </span>

                    <ActionIconCopyToClipboard
                      text={vModelName || ''}
                      canCopy={!!vModelName}
                      ariaLabel="Copy model name to clipboard"
                    />
                  </div>
                </Table.Head>
              </Table.Row>
            </Table.Header>
            <Table.Body>
              <!-- Model Path -->
              <Table.Row>
                <Table.Cell class="h-10 align-middle font-medium"
                  >{fTranslate('text_a1039763f359')}</Table.Cell
                >

                <Table.Cell
                  class="inline-flex h-10 items-center gap-2 align-middle font-mono text-xs"
                >
                  <span
                    class="resizable-text-container min-w-0 flex-1 truncate"
                    style:--threshold="14rem"
                  >
                    {vServerProps.model_path}
                  </span>

                  <ActionIconCopyToClipboard
                    text={vServerProps.model_path}
                    ariaLabel="Copy model path to clipboard"
                  />
                </Table.Cell>
              </Table.Row>

              <!-- Context Size -->
              {#if vServerProps?.default_generation_settings?.n_ctx}
                <Table.Row>
                  <Table.Cell class="h-10 align-middle font-medium"
                    >{fTranslate('text_db8951096dc3')}</Table.Cell
                  >

                  <Table.Cell
                    >{formatNumber(vServerProps.default_generation_settings.n_ctx)}
                    {fTranslate('text_c51e455b41df')}</Table.Cell
                  >
                </Table.Row>
              {:else}
                <Table.Row>
                  <Table.Cell class="h-10 align-middle font-medium text-red-500"
                    >{fTranslate('text_db8951096dc3')}</Table.Cell
                  >

                  <Table.Cell class="text-red-500">{fTranslate('text_67a926f7008e')}</Table.Cell>
                </Table.Row>
              {/if}

              <!-- Training Context -->
              {#if modelMeta?.n_ctx_train}
                <Table.Row>
                  <Table.Cell class="h-10 align-middle font-medium"
                    >{fTranslate('text_a108e1f789d7')}</Table.Cell
                  >

                  <Table.Cell
                    >{formatNumber(modelMeta.n_ctx_train)}
                    {fTranslate('text_c51e455b41df')}</Table.Cell
                  >
                </Table.Row>
              {/if}

              <!-- Model Size -->
              {#if modelMeta?.size}
                <Table.Row>
                  <Table.Cell class="h-10 align-middle font-medium"
                    >{fTranslate('text_bcae6c15f396')}</Table.Cell
                  >

                  <Table.Cell>{formatFileSize(modelMeta.size)}</Table.Cell>
                </Table.Row>
              {/if}

              <!-- Parameters -->
              {#if modelMeta?.n_params}
                <Table.Row>
                  <Table.Cell class="h-10 align-middle font-medium"
                    >{fTranslate('text_e68b36b17cbd')}</Table.Cell
                  >

                  <Table.Cell>{formatParameters(modelMeta.n_params)}</Table.Cell>
                </Table.Row>
              {/if}

              <!-- Embedding Size -->
              {#if modelMeta?.n_embd}
                <Table.Row>
                  <Table.Cell class="align-middle font-medium"
                    >{fTranslate('text_b53b4922dde8')}</Table.Cell
                  >

                  <Table.Cell>{formatNumber(modelMeta.n_embd)}</Table.Cell>
                </Table.Row>
              {/if}

              <!-- Vocabulary Size -->
              {#if modelMeta?.n_vocab}
                <Table.Row>
                  <Table.Cell class="align-middle font-medium"
                    >{fTranslate('text_86f9679cfa5b')}</Table.Cell
                  >

                  <Table.Cell
                    >{formatNumber(modelMeta.n_vocab)} {fTranslate('text_c51e455b41df')}</Table.Cell
                  >
                </Table.Row>
              {/if}

              <!-- Vocabulary Type -->
              {#if modelMeta?.vocab_type}
                <Table.Row>
                  <Table.Cell class="align-middle font-medium"
                    >{fTranslate('text_c57d0bc2c3d8')}</Table.Cell
                  >
                  <Table.Cell class="align-middle capitalize">{modelMeta.vocab_type}</Table.Cell>
                </Table.Row>
              {/if}

              <!-- Total Slots -->
              <Table.Row>
                <Table.Cell class="align-middle font-medium"
                  >{fTranslate('text_14f89dfcfa6e')}</Table.Cell
                >

                <Table.Cell>{vServerProps.total_slots}</Table.Cell>
              </Table.Row>

              <!-- Modalities -->
              {#if vModalities.length > 0}
                <Table.Row>
                  <Table.Cell class="align-middle font-medium"
                    >{fTranslate('text_870ae34ac3e0')}</Table.Cell
                  >

                  <Table.Cell>
                    <div class="flex flex-wrap gap-1">
                      <BadgesModality modalities={vModalities} />
                    </div>
                  </Table.Cell>
                </Table.Row>
              {/if}

              <!-- Build Info -->
              <Table.Row>
                <Table.Cell class="align-middle font-medium"
                  >{fTranslate('text_642ae0c77e01')}</Table.Cell
                >

                <Table.Cell class="align-middle font-mono text-xs"
                  >{vServerProps.build_info}</Table.Cell
                >
              </Table.Row>

              <!-- Chat Template -->
              {#if vServerProps.chat_template}
                <Table.Row>
                  <Table.Cell class="align-middle font-medium"
                    >{fTranslate('text_3e07c9137dce')}</Table.Cell
                  >

                  <Table.Cell class="py-10">
                    <div class="rounded-md bg-muted p-4">
                      <pre
                        class="font-mono text-xs whitespace-pre-wrap">{vServerProps.chat_template}</pre>
                    </div>
                  </Table.Cell>
                </Table.Row>
              {/if}
            </Table.Body>
          </Table.Root>
        {/if}
      {:else if !vIsLoadingModels}
        <div class="flex items-center justify-center py-8">
          <div class="text-sm text-muted-foreground">{fTranslate('text_a5d9ac0027a3')}</div>
        </div>
      {/if}
    </div>
  </Dialog.Content>
</Dialog.Root>
