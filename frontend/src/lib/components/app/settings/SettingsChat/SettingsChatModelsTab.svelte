<script lang="ts">
  import { fTranslate } from '../../../../i18n';

  import { onMount } from 'svelte';
  import { toast } from 'svelte-sonner';
  import { FolderOpen, Loader2 } from '@lucide/svelte';
  import { Button } from '$lib/components/ui/button';
  import { Input } from '$lib/components/ui/input';
  import { Badge, type BadgeVariant } from '$lib/components/ui/badge';
  import { DialogModelsFolderPicker } from '$lib/components/app/dialogs';
  import { ServerModelStatus } from '$lib/enums';
  import { cRoutes } from '$lib/constants';
  import { ModelsService } from '$lib/services/models.service';
  import { cModelsStore, fRouterModels } from '$lib/stores/models.svelte';
  import { cServerStore, fIsRouterMode } from '$lib/stores/server.svelte';
  import { cUiSettingsStore } from '$lib/stores/ui-settings.svelte';
  import { fFormatFileSize, fModelLoadFraction } from '$lib/utils';
  import type { ApiModelDataEntry, ApiModelsFolderInfo } from '$lib/types';

  let vReady = $state(false);
  let vIsRouter = $derived(fIsRouterMode());

  // router mode: folder of local models, as the server reports it
  let vFolderInfo = $state<ApiModelsFolderInfo | null>(null);
  let vFolderPickerOpen = $state(false);

  // router mode: the models of the folder and of the download cache
  let vEntries = $derived<ApiModelDataEntry[]>(fRouterModels());

  onMount(() => {
    void fInitialize();
  });

  async function fInitialize() {
    try {
      if (!cServerStore.vProps) {
        await cServerStore.mFetch();
      }

      if (fIsRouterMode()) {
        await Promise.all([fRefreshFolder(), cModelsStore.mFetch(true)]);
        cModelsStore.mSubscribeStatus();
      }
    } catch (vError) {
      console.error('Failed to read the models:', vError);
    } finally {
      vReady = true;
    }
  }

  async function fRefreshFolder() {
    vFolderInfo = await ModelsService.mGetFolder();
  }

  // the folder chosen in the folder browser is used at once: the server checks it, saves it in
  // ~/.mgs/config/ui-settings.json and rescans the models (also when the same folder is chosen
  // again). The browser shows the error and stays open when the server rejects it
  async function fApplyFolder(pFolder: string) {
    const cInfo = await ModelsService.mSetFolder(pFolder);
    vFolderInfo = cInfo;
    // the server has already saved it; the copy of the document must hold the same value, or the
    // next save of the settings would bring the previous folder back
    cUiSettingsStore.mSet('modelsFolder', cInfo.folder || undefined);
    toast.success(fTranslate('text_48b0527a69aa'));

    try {
      await cModelsStore.mFetch(true);
    } catch (vError) {
      console.error('Failed to read the models:', vError);
    }
  }

  async function fLoad(pModelId: string) {
    try {
      await cModelsStore.mLoadModel(pModelId);
      await cModelsStore.mFetchRouterModels();

      // the chat uses the model that was just loaded
      if (cModelsStore.mFindModelById(pModelId)) {
        await cModelsStore.mSelectModelById(pModelId);
      }
    } catch {
      // the models store already shows the error
    }
  }

  async function fUnload(pModelId: string) {
    try {
      await cModelsStore.mUnloadModel(pModelId);
      await cModelsStore.mFetchRouterModels();
    } catch {
      // the models store already shows the error
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

  function fIsBusy(pEntry: ApiModelDataEntry): boolean {
    return (
      pEntry.status?.value === ServerModelStatus.LOADING ||
      cModelsStore.mIsModelOperationInProgress(pEntry.id)
    );
  }

  function fStatusBadge(pEntry: ApiModelDataEntry): { label: string; variant: BadgeVariant } {
    const cStatus = pEntry.status;

    if (cStatus?.failed) {
      return { label: fTranslate('text_3d2301d732a7'), variant: 'destructive' };
    }

    switch (cStatus?.value) {
      case ServerModelStatus.LOADED:
        return { label: fTranslate('text_d01476dfee7e'), variant: 'default' };
      case ServerModelStatus.LOADING: {
        const cPercent = Math.round(
          fModelLoadFraction(cModelsStore.mGetLoadProgress(pEntry.id)) * 100
        );
        return { label: fTranslate('text_0c39f6f09a0e', { p0: cPercent }), variant: 'secondary' };
      }
      case ServerModelStatus.SLEEPING:
        return { label: fTranslate('text_12ff393b061d'), variant: 'tertiary' };
      default:
        return { label: fTranslate('text_9b41ba20b8a3'), variant: 'outline' };
    }
  }
</script>

{#if !vReady}
  <div class="flex items-center gap-2 py-8 text-sm text-muted-foreground">
    <Loader2 class="h-4 w-4 animate-spin" />
    {fTranslate('text_68ad292b7736')}
  </div>
{:else}
  <div class="space-y-8">
    {#if vIsRouter}
      <!-- Models folder (the global search highlights the sections by their ids) -->
      <section id="settings-models-folder" class="min-w-0 space-y-3 rounded-md">
        <h4 class="text-sm font-semibold">{fTranslate('text_9460a43cccd9')}</h4>

        <p class="text-xs text-muted-foreground">
          {fTranslate('text_d6ce3e7b675d')}
          {fTranslate('text_a55502cb93fb')}
        </p>

        <div class="flex flex-wrap items-center gap-2">
          <!-- the folder is only chosen in the folder browser, which gives its absolute path -->
          <Input
            class="min-w-0 flex-1 basis-56 font-mono text-xs"
            dir="ltr"
            value={vFolderInfo?.folder ?? ''}
            readonly
            disabled={vFolderInfo?.source === 'cli'}
            aria-label={fTranslate('text_9460a43cccd9')}
          />

          <Button
            type="button"
            size="sm"
            disabled={vFolderInfo?.source === 'cli'}
            onclick={() => (vFolderPickerOpen = true)}
          >
            <FolderOpen class="h-3.5 w-3.5" />
            {fTranslate('text_3db741003e46')}
          </Button>
        </div>

        <DialogModelsFolderPicker bind:open={vFolderPickerOpen} onChoose={fApplyFolder} />

        <div class="space-y-1 text-xs [overflow-wrap:anywhere]">
          {#if vFolderInfo?.source === 'cli'}
            <p class="text-muted-foreground">{fTranslate('text_208ed365d9f1')}</p>
          {/if}

          {#if !vFolderInfo || vFolderInfo.source === 'none'}
            <p class="text-muted-foreground">{fTranslate('text_de27cba4ce76')}</p>
          {:else if !vFolderInfo.exists}
            <p class="text-destructive">{fTranslate('text_84570201d82d')}</p>
          {:else}
            <p class="text-muted-foreground">
              {fTranslate('text_f9c513612078', { p0: vFolderInfo.model_count })}
            </p>
          {/if}

          {#if vFolderInfo?.source === 'settings' && vFolderInfo.settings_file}
            <p class="text-muted-foreground">
              {fTranslate('text_6d1cd7c6fd56', { p0: vFolderInfo.settings_file })}
            </p>
          {/if}

          {#if vFolderInfo?.cache_dir}
            <p class="text-muted-foreground">
              {fTranslate('text_15200a03c540', { p0: vFolderInfo.cache_dir })}
            </p>
          {/if}
        </div>
      </section>

      <!-- Available models -->
      <section id="settings-models-available" class="min-w-0 space-y-3 rounded-md">
        <h4 class="text-sm font-semibold">{fTranslate('text_9dbfd0478bd5')}</h4>

        {#if vEntries.length === 0}
          <p class="text-sm text-muted-foreground">{fTranslate('text_9e064877fb5f')}</p>
        {:else}
          <ul class="space-y-2">
            {#each vEntries as vEntry (vEntry.id)}
              {@const cBadge = fStatusBadge(vEntry)}
              {@const cBusy = fIsBusy(vEntry)}
              <li
                id="settings-model-{vEntry.id}"
                class="flex min-w-0 flex-wrap items-center gap-x-3 gap-y-2 rounded-lg border border-border/50 px-3 py-2"
              >
                <div class="min-w-0 flex-1 basis-48 space-y-0.5">
                  <div class="flex min-w-0 flex-wrap items-center gap-2">
                    <span class="min-w-0 text-sm font-medium [overflow-wrap:anywhere]"
                      >{vEntry.id}</span
                    >
                    <Badge variant={cBadge.variant}>{cBadge.label}</Badge>
                  </div>

                  {#if vEntry.path}
                    <p class="font-mono text-xs text-muted-foreground [overflow-wrap:anywhere]">
                      {vEntry.path}
                    </p>
                  {/if}

                  {#if vEntry.file_size}
                    <p class="text-xs text-muted-foreground">{fFormatFileSize(vEntry.file_size)}</p>
                  {/if}
                </div>

                <div class="flex shrink-0 items-center gap-2">
                  {#if cBusy}
                    <Loader2 class="h-4 w-4 animate-spin text-muted-foreground" />
                  {:else if fIsRunning(vEntry)}
                    <Button size="sm" variant="outline" onclick={() => fUnload(vEntry.id)}>
                      {fTranslate('text_c2ed103ced3e')}
                    </Button>
                  {:else}
                    <Button size="sm" onclick={() => fLoad(vEntry.id)}>
                      {vEntry.status?.failed
                        ? fTranslate('text_3890e4decfe9')
                        : fTranslate('text_889fad74ece5')}
                    </Button>
                  {/if}
                </div>
              </li>
            {/each}
          </ul>
        {/if}
      </section>
    {:else}
      <p class="text-sm text-muted-foreground">{fTranslate('text_8ba8bc8bb554')}</p>
    {/if}

    <!-- what the loaded models are and how they were loaded is on the Model page -->
    <p class="text-xs text-muted-foreground">
      <a href={cRoutes.MODEL} class="underline underline-offset-2 hover:text-foreground"
        >{fTranslate('text_b70205b219fc')}</a
      >
    </p>
  </div>
{/if}
