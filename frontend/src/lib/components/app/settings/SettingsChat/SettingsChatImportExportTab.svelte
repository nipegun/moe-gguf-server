<script lang="ts">
  import { fTranslate } from '../../../../i18n';

  import { Download, Upload, Trash2 } from '@lucide/svelte';
  import {
    DialogConversationSelection,
    DialogConfirmation,
    DialogExportSettings
  } from '$lib/components/app';
  import { createMessageCountMap } from '$lib/utils';
  import { cSettingsStore } from '$lib/stores/settings.svelte';
  import { cConversationsStore, fConversations } from '$lib/stores/conversations.svelte';
  import { toast } from 'svelte-sonner';
  import { fade } from 'svelte/transition';
  import { ConversationSelectionMode, HtmlInputType, FileExtensionText } from '$lib/enums';
  import SettingsChatImportExportSection from './SettingsChatImportExportSection.svelte';
  import SettingsGroup from '$lib/components/app/settings/SettingsGroup.svelte';

  let vExportedConversations = $state<DatabaseConversation[]>([]);
  let vImportedConversations = $state<DatabaseConversation[]>([]);
  let vShowExportSummary = $state(false);
  let vShowImportSummary = $state(false);

  let vShowExportDialog = $state(false);
  let vShowImportDialog = $state(false);
  let vAvailableConversations = $state<DatabaseConversation[]>([]);
  let vMessageCountMap = $state<Map<string, number>>(new Map());
  let vFullImportData = $state<Array<{ conv: DatabaseConversation; messages: DatabaseMessage[] }>>(
    []
  );

  // Delete functionality state
  let vShowDeleteDialog = $state(false);

  // Settings import/export state
  let vShowSettingsExportSummary = $state(false);
  let vShowSettingsImportSummary = $state(false);
  let vShowSettingsExportDialog = $state(false);
  let vIncludeSensitiveData = $state(false);

  function fHandleSettingsExport() {
    vShowSettingsExportDialog = true;
    vIncludeSensitiveData = false;
  }

  function fHandleSettingsExportConfirm() {
    vShowSettingsExportDialog = false;

    try {
      const cData = cSettingsStore.mExportSettings(vIncludeSensitiveData);
      const cBlob = new Blob([JSON.stringify(cData, null, 2)], { type: 'application/json' });
      const cUrl = URL.createObjectURL(cBlob);
      const cA = document.createElement('a');
      cA.href = cUrl;
      cA.download = `mgs_settings_${new Date().toISOString().split('T')[0]}.json`;
      document.body.appendChild(cA);
      cA.click();
      document.body.removeChild(cA);
      URL.revokeObjectURL(cUrl);

      vShowSettingsExportSummary = true;
      vShowSettingsImportSummary = false;
      toast.success(fTranslate('text_5bfca92e998f'));
    } catch (vErr) {
      console.error('Failed to export settings:', vErr);
      toast.error(fTranslate('text_3cf664c4bd2b'));
    }
  }

  function fHandleSettingsExportCancel() {
    vShowSettingsExportDialog = false;
  }

  function fHandleSettingsImport() {
    try {
      const cInput = document.createElement('input');
      cInput.type = HtmlInputType.FILE;
      cInput.accept = FileExtensionText.JSON;

      cInput.onchange = async (pE) => {
        const cFile = (pE.target as HTMLInputElement)?.files?.[0];
        if (!cFile) return;

        try {
          const cText = await cFile.text();
          const cData = JSON.parse(cText);

          if (!cData || typeof cData !== 'object' || !cData.config) {
            toast.error(fTranslate('text_e65771eee330'));
            return;
          }

          cSettingsStore.mImportSettings(cData);

          vShowSettingsImportSummary = true;
          vShowSettingsExportSummary = false;
          toast.success(fTranslate('text_f3addd823e40'));
        } catch (vErr) {
          console.error('Failed to import settings:', vErr);
          toast.error(fTranslate('text_6ffd4b51379f'));
        }
      };

      cInput.click();
    } catch (vErr) {
      console.error('Failed to open file picker:', vErr);
      toast.error(fTranslate('text_5843f72974fd'));
    }
  }

  async function fHandleExportClick() {
    try {
      const cAllConversations = fConversations();
      if (cAllConversations.length === 0) {
        toast.info(fTranslate('text_bf0779e06dd8'));
        return;
      }

      const cConversationsWithMessages = await cConversationsStore.mGetConversationsForExport(
        cAllConversations.map((pConversation) => pConversation.id)
      );

      vMessageCountMap = createMessageCountMap(cConversationsWithMessages);
      vAvailableConversations = cConversationsWithMessages.map((pEntry) => pEntry.conv);
      vShowExportDialog = true;
    } catch (vErr) {
      console.error('Failed to load conversations:', vErr);
      alert(fTranslate('text_ae55fc046c4c'));
    }
  }

  async function fHandleExportConfirm(pSelectedConversations: DatabaseConversation[]) {
    try {
      const cAllData = await cConversationsStore.mGetConversationsForExport(
        pSelectedConversations.map((pConversation) => pConversation.id)
      );

      if (cAllData.length === 0) {
        toast.info(fTranslate('text_bf0779e06dd8'));
        return;
      }

      if (cAllData.length === 1) {
        cConversationsStore.mDownloadConversationFile(cAllData[0]);
      } else {
        cConversationsStore.mDownloadConversationsArchive(cAllData);
      }

      vExportedConversations = cAllData.map((pEntry) => pEntry.conv);
      vShowExportSummary = true;
      vShowImportSummary = false;
      vShowExportDialog = false;
    } catch (vErr) {
      console.error('Export failed:', vErr);
      alert(fTranslate('text_96f775dbb3fc'));
    }
  }

  async function fHandleImportClick() {
    try {
      const cInput = document.createElement('input');

      // No `accept` filter: iOS resolves each entry to a UTI and has none for
      // `.jsonl`, which greys out exported conversations in the file picker.
      // `parseImportFile` detects the format from the file contents instead.
      cInput.type = HtmlInputType.FILE;

      cInput.onchange = async (pE) => {
        const cFile = (pE.target as HTMLInputElement)?.files?.[0];
        if (!cFile) return;

        try {
          const cImportedData = await cConversationsStore.mParseImportFile(cFile);

          if (cImportedData.length === 0) {
            throw new Error(fTranslate('text_5db93832d23b'));
          }

          vFullImportData = cImportedData;
          vAvailableConversations = cImportedData.map((pItem) => pItem.conv);
          vMessageCountMap = createMessageCountMap(cImportedData);
          vShowImportDialog = true;
        } catch (vErr: unknown) {
          const cMessage = vErr instanceof Error ? vErr.message : fTranslate('text_27c2ccd962c2');

          console.error('Failed to parse file:', vErr);
          alert(fTranslate('text_0d691e020bf8', { p0: cMessage }));
        }
      };

      cInput.click();
    } catch (vErr) {
      console.error('Import failed:', vErr);
      alert(fTranslate('text_ebbdaaef4f97'));
    }
  }

  async function fHandleImportConfirm(pSelectedConversations: DatabaseConversation[]) {
    try {
      const sSelectedIds = new Set(pSelectedConversations.map((pC) => pC.id));
      const cSelectedData = $state
        .snapshot(vFullImportData)
        .filter((pItem) => sSelectedIds.has(pItem.conv.id));

      const { imported: cImported, skipped: cSkipped } =
        await cConversationsStore.mImportConversationsData(cSelectedData);

      // A conversation already in the database is left untouched, so the summary
      // lists what was written and the toast accounts for the rest.
      if (cSkipped.length > 0) {
        toast.info(
          fTranslate('text_5a073b148789', {
            p0: cSkipped.length,
            p1: cSkipped.length === 1 ? '' : 's'
          })
        );
      }

      vImportedConversations = cImported;
      vShowImportSummary = true;
      vShowExportSummary = false;
      vShowImportDialog = false;
    } catch (vErr) {
      console.error('Import failed:', vErr);
      alert(fTranslate('text_1b082e402d1d'));
    }
  }

  async function fHandleDeleteAllClick() {
    try {
      const cAllConversations = fConversations();

      if (cAllConversations.length === 0) {
        toast.info(fTranslate('text_5c9a78a4a817'));
        return;
      }

      vShowDeleteDialog = true;
    } catch (vErr) {
      console.error('Failed to load conversations for deletion:', vErr);
      toast.error(fTranslate('text_ae55fc046c4c'));
    }
  }

  async function fHandleDeleteAllConfirm() {
    try {
      await cConversationsStore.mDeleteAll();

      vShowDeleteDialog = false;
    } catch (vErr) {
      console.error('Failed to delete conversations:', vErr);
    }
  }

  function fHandleDeleteAllCancel() {
    vShowDeleteDialog = false;
  }
</script>

<div class="space-y-12" in:fade={{ duration: 150 }}>
  <SettingsGroup title={fTranslate('text_1d432f58690c')}>
    <SettingsChatImportExportSection
      title={fTranslate('text_3664895579f0')}
      description={fTranslate('text_e1c29b8d7188')}
      IconComponent={Download}
      buttonText="Export conversations"
      onclick={fHandleExportClick}
      summary={{ show: vShowExportSummary, verb: 'Exported', items: vExportedConversations }}
    />

    <SettingsChatImportExportSection
      title={fTranslate('text_2cff9baabf56')}
      description={fTranslate('text_a659626215c6')}
      IconComponent={Upload}
      buttonText="Import conversations"
      onclick={fHandleImportClick}
      summary={{ show: vShowImportSummary, verb: 'Imported', items: vImportedConversations }}
    />

    <SettingsChatImportExportSection
      title={fTranslate('text_ef85afa1b92c')}
      description={fTranslate('text_21814f9e74f4')}
      IconComponent={Trash2}
      buttonText="Delete all conversations"
      onclick={fHandleDeleteAllClick}
      titleClass="text-destructive"
      buttonVariant="destructive"
      buttonClass="text-destructive-foreground justify-start justify-self-start bg-destructive hover:bg-destructive/80 md:w-auto"
    />
  </SettingsGroup>

  <SettingsGroup title={fTranslate('text_74a883a037bc')}>
    <SettingsChatImportExportSection
      title={fTranslate('text_3664895579f0')}
      description={fTranslate('text_72df65773b40')}
      IconComponent={Download}
      buttonText="Export settings"
      onclick={fHandleSettingsExport}
      summary={{ show: vShowSettingsExportSummary, verb: 'Exported', items: [] }}
    />

    <SettingsChatImportExportSection
      title={fTranslate('text_2cff9baabf56')}
      description={fTranslate('text_6235faca5e63')}
      IconComponent={Upload}
      buttonText="Import settings"
      onclick={fHandleSettingsImport}
      summary={{ show: vShowSettingsImportSummary, verb: 'Imported', items: [] }}
    />
  </SettingsGroup>
</div>

<DialogExportSettings
  bind:open={vShowSettingsExportDialog}
  bind:includeSensitiveData={vIncludeSensitiveData}
  onConfirm={fHandleSettingsExportConfirm}
  onCancel={fHandleSettingsExportCancel}
/>

<DialogConversationSelection
  conversations={vAvailableConversations}
  messageCountMap={vMessageCountMap}
  mode={ConversationSelectionMode.EXPORT}
  bind:open={vShowExportDialog}
  onCancel={() => (vShowExportDialog = false)}
  onConfirm={fHandleExportConfirm}
/>

<DialogConversationSelection
  conversations={vAvailableConversations}
  messageCountMap={vMessageCountMap}
  mode={ConversationSelectionMode.IMPORT}
  bind:open={vShowImportDialog}
  onCancel={() => (vShowImportDialog = false)}
  onConfirm={fHandleImportConfirm}
/>

<DialogConfirmation
  bind:open={vShowDeleteDialog}
  title={fTranslate('text_43a72ccf4424')}
  description={fTranslate('text_45145fe2a7ce')}
  confirmText={fTranslate('text_ef85afa1b92c')}
  cancelText={fTranslate('text_19766ed6ccb2')}
  variant="destructive"
  icon={Trash2}
  onConfirm={fHandleDeleteAllConfirm}
  onCancel={fHandleDeleteAllCancel}
/>
