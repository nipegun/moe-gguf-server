<script lang="ts">
  import { fTranslate } from '../../../i18n';

  import { Box, Loader2, X } from '@lucide/svelte';
  import { fade } from 'svelte/transition';
  import { onMount } from 'svelte';
  import { browser } from '$app/environment';
  import { goto } from '$app/navigation';
  import { page } from '$app/state';
  import { ActionIcon } from '$lib/components/app';
  import { ServerModelStatus } from '$lib/enums';
  import { cRoutes, cSettingsSectionSlugs } from '$lib/constants';
  import { ModelsService } from '$lib/services/models.service';
  import { RouterService } from '$lib/services/router.service';
  import { cModelsStore, fRouterModels, fSelectedModelName } from '$lib/stores/models.svelte';
  import { cMonitorStore } from '$lib/stores/monitor.svelte';
  import { cServerStore, fIsRouterMode } from '$lib/stores/server.svelte';
  import { fFormatFileSize, fFormatNumber, fFormatParameters } from '$lib/utils';
  import type { ApiModelDataEntry, MonitorModel } from '$lib/types';

  /**
   * Model page: what every loaded model is, the configuration it was loaded with (after --fit and
   * the automatic choices of mgs) and the command line of its process. The models are chosen and
   * loaded in Settings > Models.
   */

  interface Props {
    class?: string;
  }

  let { class: vClassName = '' }: Props = $props();

  interface DetailRow {
    label: string;
    value: string;
    mono?: boolean;
    // on/off state shown before the value, in green while it is active
    state?: { text: string; active: boolean };
  }

  let vReady = $state(false);
  let vIsRouter = $derived(fIsRouterMode());

  // single-model mode (mgs --model): the model of /v1/models, with its load configuration
  let vSingleEntries = $state<ApiModelDataEntry[]>([]);

  let vEntries = $derived<ApiModelDataEntry[]>(vIsRouter ? fRouterModels() : vSingleEntries);
  let vRunningEntries = $derived(vEntries.filter((pEntry) => fIsRunning(pEntry)));

  // the page shows the loaded model: the one selected in the chat when it runs, else the first
  let vEntry = $derived(
    vRunningEntries.find((pEntry) => pEntry.id === fSelectedModelName()) ?? vRunningEntries[0]
  );

  // the load configuration does not change while a model runs, but whether its MoE expert cache was
  // created and how many draft tokens MTP asks for do: they come from the monitor report
  let vHasRunning = $derived(vRunningEntries.length > 0);

  $effect(() => {
    if (!vHasRunning) return;

    return cMonitorStore.mSubscribeReport();
  });

  // ids whose details were already requested while they run, so a refresh is asked only once per load
  const sDetailsRequested = new Set<string>();

  // the status feed only changes the status of a model: once it runs, its load configuration is fetched
  $effect(() => {
    if (!vIsRouter) return;

    for (const cEntry of fRouterModels()) {
      if (!fIsRunning(cEntry)) {
        sDetailsRequested.delete(cEntry.id);
        continue;
      }
      if (!cEntry.load && !sDetailsRequested.has(cEntry.id)) {
        sDetailsRequested.add(cEntry.id);
        void cModelsStore.mFetchRouterModels();
      }
    }
  });

  let vPreviousRouteId = $state<string | null>(null);

  $effect(() => {
    const cCurrentId = page.route.id;
    return () => {
      vPreviousRouteId = cCurrentId;
    };
  });

  function fHandleClose() {
    if (browser && window.history.length > 1 && vPreviousRouteId !== '/model') {
      history.back();
    } else {
      goto(cRoutes.START);
    }
  }

  onMount(() => {
    void fInitialize();
  });

  async function fInitialize() {
    try {
      if (!cServerStore.vProps) {
        await cServerStore.mFetch();
      }

      if (fIsRouterMode()) {
        await cModelsStore.mFetch(true);
        cModelsStore.mSubscribeStatus();
      } else {
        const cList = await ModelsService.mList();
        vSingleEntries = cList.data;
      }
    } catch (vError) {
      console.error('Failed to read the models:', vError);
    } finally {
      vReady = true;
    }
  }

  function fIsRunning(pEntry: ApiModelDataEntry): boolean {
    // outside router mode the only model is always loaded and has no status
    if (!pEntry.status) return true;

    return (
      pEntry.status.value === ServerModelStatus.LOADED ||
      pEntry.status.value === ServerModelStatus.SLEEPING
    );
  }

  function fTokens(pValue: unknown): string {
    return `${fFormatNumber(pValue)} ${fTranslate('text_c51e455b41df')}`;
  }

  function fMonitorModel(pEntry: ApiModelDataEntry): MonitorModel | undefined {
    return cMonitorStore.vReport?.models.find((pModel) => pModel.name === pEntry.id);
  }

  function fStateRow(pLabel: string, pActive: boolean, pText: string, pDetail = ''): DetailRow {
    return { label: pLabel, value: pDetail, state: { text: pText, active: pActive } };
  }

  function fFlashAttnRow(pEntry: ApiModelDataEntry): DetailRow | null {
    const cLoad = pEntry.load ?? {};
    if (cLoad.flash_attn === undefined) return null;

    const cEnabled = cLoad.flash_attn_enabled ?? cLoad.flash_attn === 'enabled';
    return fStateRow(
      fTranslate('text_6cdfc6da156b'),
      cEnabled,
      cEnabled ? fTranslate('text_130011756125') : fTranslate('text_ca7981b46ecf'),
      cLoad.flash_attn === 'auto' ? fTranslate('text_e2c959889d5f') : ''
    );
  }

  // MTP with the number of draft tokens it asks for in every step: measured while generating when
  // no result was saved, and none when drafting did not pay off on this computer
  function fMtpRow(pEntry: ApiModelDataEntry): DetailRow | null {
    const cLoad = pEntry.load ?? {};
    const cLabel = fTranslate('text_6fa4ae336375');
    if (cLoad.mtp === undefined) return null;
    if (!cLoad.mtp) return fStateRow(cLabel, false, fTranslate('text_4505af92e283'));

    const cLive = fMonitorModel(pEntry)?.mtp;
    if (cLive?.measuring) {
      return fStateRow(
        cLabel,
        true,
        fTranslate('text_92340695899b'),
        fTranslate('text_c67b7505e1e8', { p0: cLive.draft_n_max })
      );
    }
    if (cLive && cLive.draft_n === 0) {
      return fStateRow(
        cLabel,
        false,
        fTranslate('text_4505af92e283'),
        fTranslate('text_3e221c6c0f80')
      );
    }

    const cDraftN = cLive?.draft_n ?? cLoad.draft_n_max;
    return fStateRow(
      cLabel,
      true,
      fTranslate('text_92340695899b'),
      cDraftN ? fTranslate('text_c25d71c88199', { p0: cDraftN }) : ''
    );
  }

  // MoE expert cache: whether it was created, with its automatic or requested size. The first
  // request after loading the model creates it, only when experts stay in RAM on a dedicated GPU
  function fMoeCacheRow(pEntry: ApiModelDataEntry): DetailRow | null {
    const cMib = pEntry.load?.moe_cache_mib;
    const cLabel = fTranslate('text_72b38239f404');
    if (cMib === undefined) return null;
    if (cMib === 0) return fStateRow(cLabel, false, fTranslate('text_75081b593d15'));

    const cSize =
      cMib < 0
        ? fTranslate('text_27f865e28dff')
        : fTranslate('text_1286a31b376b', { p0: fFormatNumber(cMib) });
    const cLive = fMonitorModel(pEntry)?.expert_cache;
    if (!cLive) {
      return {
        label: cLabel,
        value: cMib < 0 ? fTranslate('text_d461a493a375') : `${fFormatNumber(cMib)} MiB`
      };
    }
    if (cLive.pending) return fStateRow(cLabel, false, fTranslate('text_401d1d5f1533'), cSize);
    if (cLive.active) return fStateRow(cLabel, true, fTranslate('text_52d26f44d33f'), cSize);

    return fStateRow(cLabel, false, fTranslate('text_14eaa544f2e2'), cSize);
  }

  function fModelRows(pEntry: ApiModelDataEntry): DetailRow[] {
    const cLoad = pEntry.load ?? {};
    const cMeta = (pEntry.meta ?? {}) as Record<string, unknown>;
    const cModelPath =
      pEntry.path ?? (vIsRouter ? '' : String(cServerStore.vProps?.model_path ?? ''));
    const aRows: DetailRow[] = [
      { label: fTranslate('text_5e2c614c23f0'), value: cLoad.description ?? '' },
      { label: fTranslate('text_cd74053c5481'), value: cLoad.architecture ?? '' },
      { label: fTranslate('text_a1039763f359'), value: cModelPath, mono: true },
      {
        label: pEntry.file_size ? fTranslate('text_5f1de958ff97') : fTranslate('text_bcae6c15f396'),
        value: pEntry.file_size
          ? fFormatFileSize(pEntry.file_size)
          : cMeta.size
            ? fFormatFileSize(cMeta.size)
            : ''
      },
      {
        label: fTranslate('text_e68b36b17cbd'),
        value: cMeta.n_params ? fFormatParameters(cMeta.n_params) : ''
      },
      {
        label: fTranslate('text_0bcd66e677f4'),
        value: cLoad.n_layer !== undefined ? fFormatNumber(cLoad.n_layer) : ''
      },
      {
        label: fTranslate('text_986e228db394'),
        value: cLoad.n_layer_mtp ? fFormatNumber(cLoad.n_layer_mtp) : ''
      },
      {
        label: fTranslate('text_a108e1f789d7'),
        value: cMeta.n_ctx_train ? fTokens(cMeta.n_ctx_train) : ''
      },
      {
        label: fTranslate('text_6149b8a47ac9'),
        value: cLoad.mmproj ?? pEntry.mmproj ?? '',
        mono: true
      }
    ];

    return aRows.filter((pRow) => pRow.value !== '');
  }

  function fLoadRows(pEntry: ApiModelDataEntry): DetailRow[] {
    const cLoad = pEntry.load;
    if (!cLoad) return [];

    const cDevices = (cLoad.devices ?? [])
      .map((pDevice) =>
        pDevice.description ? `${pDevice.name} (${pDevice.description})` : pDevice.name
      )
      .join(', ');
    const cSpeculative =
      cLoad.speculative && cLoad.speculative.length > 0
        ? fStateRow(
            fTranslate('text_1fc5d32840e0'),
            true,
            fTranslate('text_750af77aa42e'),
            fTranslate('text_258bddee7d46', {
              p0: cLoad.speculative.join(', '),
              p1: cLoad.draft_n_max ?? 0
            })
          )
        : fStateRow(fTranslate('text_1fc5d32840e0'), false, fTranslate('text_e5c557a8f88f'));
    const aRows: (DetailRow | null)[] = [
      {
        label: fTranslate('text_db8951096dc3'),
        value: cLoad.n_ctx !== undefined ? fTokens(cLoad.n_ctx) : ''
      },
      {
        label: fTranslate('text_14f89dfcfa6e'),
        value: cLoad.n_parallel !== undefined ? fFormatNumber(cLoad.n_parallel) : ''
      },
      {
        label: fTranslate('text_4d9114195710'),
        value:
          cLoad.n_parallel && cLoad.n_parallel > 1 && cLoad.n_ctx_slot !== undefined
            ? fTokens(cLoad.n_ctx_slot)
            : ''
      },
      {
        label: fTranslate('text_ab6fd62768ae'),
        value:
          cLoad.n_batch !== undefined
            ? `${fFormatNumber(cLoad.n_batch)} / ${fFormatNumber(cLoad.n_ubatch)}`
            : ''
      },
      {
        label: fTranslate('text_0b7ad81510b4'),
        value: cLoad.n_threads !== undefined ? `${cLoad.n_threads} / ${cLoad.n_threads_batch}` : ''
      },
      fFlashAttnRow(pEntry),
      {
        label: fTranslate('text_8a63e3185573'),
        value: cLoad.cache_type_k ? `${cLoad.cache_type_k} / ${cLoad.cache_type_v}` : ''
      },
      cLoad.fit === undefined
        ? null
        : fStateRow(
            fTranslate('text_4390f98dedc5'),
            cLoad.fit,
            cLoad.fit ? fTranslate('text_130011756125') : fTranslate('text_ca7981b46ecf')
          ),
      fMtpRow(pEntry),
      cSpeculative,
      fMoeCacheRow(pEntry),
      { label: fTranslate('text_4ba5121d4d1b'), value: cDevices || fTranslate('text_e5b50eb23c80') }
    ];

    return aRows.filter(
      (pRow): pRow is DetailRow => pRow !== null && (pRow.value !== '' || !!pRow.state)
    );
  }
</script>

{#snippet fDetailList(pRows: DetailRow[])}
  <dl class="divide-y divide-border/40 text-sm">
    {#each pRows as vRow (vRow.label)}
      <div class="flex flex-wrap items-baseline justify-between gap-x-4 gap-y-0.5 py-1.5">
        <dt class="min-w-0 text-muted-foreground">{vRow.label}</dt>
        <dd
          class={['min-w-0 text-right [overflow-wrap:anywhere]', vRow.mono && 'font-mono text-xs']}
        >
          {#if vRow.state}
            <span
              class={vRow.state.active ? 'font-medium text-green-600 dark:text-green-400' : undefined}
              >{vRow.state.text}</span
            >
            {vRow.value}
          {:else}
            {vRow.value}
          {/if}
        </dd>
      </div>
    {/each}
  </dl>
{/snippet}

<div in:fade={{ duration: 150 }} class="flex min-h-[calc(100dvh-4rem)] flex-col">
  <div class="fixed top-4.5 right-4 z-50 md:hidden">
    <ActionIcon icon={X} tooltip={fTranslate('text_7d9eb7acb13e')} onclick={fHandleClose} />
  </div>

  <div class="sticky top-0 z-10 mt-4 mb-2 flex items-center gap-2 px-4 md:p-4 md:px-8">
    <Box class="h-5 w-5 md:h-6 md:w-6" />

    <h1 class="text-lg font-semibold md:text-2xl">{fTranslate('text_5e2c614c23f0')}</h1>
  </div>

  <div class="mx-auto flex w-full min-w-0 max-w-6xl flex-col gap-4 {vClassName}">
    {#if !vReady}
      <div class="flex items-center gap-2 py-8 text-sm text-muted-foreground">
        <Loader2 class="h-4 w-4 animate-spin" />
        {fTranslate('text_68ad292b7736')}
      </div>
    {:else if vRunningEntries.length === 0}
      <div class="space-y-2 py-8 text-sm text-muted-foreground">
        <p>{fTranslate('text_33dded6e7696')}</p>

        <p>
          <a
            href={RouterService.mSettings(cSettingsSectionSlugs.MODELS)}
            class="underline underline-offset-2 hover:text-foreground"
            >{fTranslate('text_1ff5cec341ca')}</a
          >
        </p>
      </div>
    {:else if vEntry}
      {@const cLoadRows = fLoadRows(vEntry)}
      <h2 class="text-sm font-semibold [overflow-wrap:anywhere]">
        {fTranslate('text_49c8b4796622', { p0: vEntry.id })}
      </h2>

      <div class="grid min-w-0 items-start gap-4 lg:grid-cols-2">
        <!-- what the GGUF file holds (the global search highlights the boxes by their ids) -->
        <section
          id="model-file"
          class="min-w-0 space-y-2 rounded-xl border border-border/50 bg-card p-4"
        >
          <h3 class="text-xs font-semibold tracking-wide text-muted-foreground uppercase">
            {fTranslate('text_3264391adcc7')}
          </h3>

          {@render fDetailList(fModelRows(vEntry))}
        </section>

        <!-- how it was loaded: the configuration chosen by mgs or passed, and its command line -->
        <section
          id="model-load-parameters"
          class="min-w-0 space-y-4 rounded-xl border border-border/50 bg-card p-4"
        >
          <div class="space-y-2">
            <h3 class="text-xs font-semibold tracking-wide text-muted-foreground uppercase">
              {fTranslate('text_0832cad691aa')}
            </h3>

            {@render fDetailList(cLoadRows)}
          </div>

          {#if vEntry.status?.args && vEntry.status.args.length > 0}
            <div id="model-command-line" class="space-y-1 rounded-md">
              <h3 class="text-xs font-semibold tracking-wide text-muted-foreground uppercase">
                {fTranslate('text_c925a11891b8')}
              </h3>

              <pre
                class="rounded-md bg-muted p-3 font-mono text-xs whitespace-pre-wrap [overflow-wrap:anywhere]">{vEntry.status.args.join(
                  ' '
                )}</pre>
            </div>
          {/if}

          <p class="text-xs text-muted-foreground">
            <a href={cRoutes.MONITOR} class="underline underline-offset-2 hover:text-foreground"
              >{fTranslate('text_17428009a65c')}</a
            >
          </p>
        </section>
      </div>
    {/if}
  </div>
</div>
