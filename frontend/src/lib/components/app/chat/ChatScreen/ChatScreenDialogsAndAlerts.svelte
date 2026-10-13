<script lang="ts">
  import { fTranslate } from '../../../../i18n';

  import { Trash2 } from '@lucide/svelte';
  import { ErrorDialogType } from '$lib/enums';
  import {
    DialogChatError,
    DialogConfirmation,
    DialogEmptyFileAlert,
    DialogFileUploadError
  } from '$lib/components/app';

  let {
    showDeleteDialog: vShowDeleteDialog,
    handleDeleteConfirm: vHandleDeleteConfirm,
    showEmptyFileDialog: vShowEmptyFileDialog,
    emptyFileNames: vEmptyFileNames,
    activeErrorDialog: vActiveErrorDialog,
    handleErrorDialogOpenChange: vHandleErrorDialogOpenChange,
    fileUpload: vFileUpload
  } = $props();
</script>

<DialogFileUploadError
  bind:open={vFileUpload.showFileErrorDialog}
  fileErrorData={vFileUpload.fileErrorData}
/>

<DialogConfirmation
  bind:open={vShowDeleteDialog}
  title={fTranslate('text_9ccd1fb969eb')}
  description={fTranslate('text_7b20577dc1ec')}
  confirmText={fTranslate('text_e2d0a54968ea')}
  cancelText={fTranslate('text_19766ed6ccb2')}
  variant="destructive"
  icon={Trash2}
  onConfirm={vHandleDeleteConfirm}
  onCancel={() => (vShowDeleteDialog = false)}
/>

<DialogEmptyFileAlert
  bind:open={vShowEmptyFileDialog}
  emptyFiles={vEmptyFileNames}
  onOpenChange={(open) => {
    if (!open) {
      vEmptyFileNames = [];
    }
  }}
/>

<DialogChatError
  message={vActiveErrorDialog?.message ?? ''}
  contextInfo={vActiveErrorDialog?.contextInfo}
  onOpenChange={vHandleErrorDialogOpenChange}
  open={Boolean(vActiveErrorDialog)}
  type={vActiveErrorDialog?.type ?? ErrorDialogType.SERVER}
/>
