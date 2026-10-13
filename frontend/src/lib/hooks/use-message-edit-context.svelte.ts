import { setMessageEditContext } from '$lib/contexts';
import { MessageRole } from '$lib/enums';
import { fParseFilesToMessageExtras } from '$lib/utils/convert-files-to-extra';

interface UseMessageEditContextOptions {
  getContent: () => string;
  getExtras: () => DatabaseMessageExtra[];
  showSaveOnlyOption?: boolean;
  onSave: (pContent: string, pExtras?: DatabaseMessageExtra[]) => void;
}

export function fUseMessageEditContext(pOptions: UseMessageEditContextOptions) {
  let vIsEditing = $state(false);
  let vEditedContent = $state('');
  let vEditedExtras = $state<DatabaseMessageExtra[]>([]);
  let vEditedUploadedFiles = $state<ChatUploadedFile[]>([]);

  function fHandleEdit() {
    vEditedContent = pOptions.getContent();
    vEditedExtras = [...pOptions.getExtras()];
    vEditedUploadedFiles = [];
    vIsEditing = true;
  }

  async function fHandleSaveEdit() {
    const cTrimmed = vEditedContent.trim();
    if (!cTrimmed && vEditedExtras.length === 0 && vEditedUploadedFiles.length === 0) return;

    let aFinalExtras: DatabaseMessageExtra[] = $state.snapshot(vEditedExtras);
    if (vEditedUploadedFiles.length > 0) {
      const cPlainFiles = $state.snapshot(vEditedUploadedFiles);
      const cResult = await fParseFilesToMessageExtras(cPlainFiles);
      const cNewExtras = cResult?.extras || [];
      aFinalExtras = [...aFinalExtras, ...cNewExtras];
    }

    pOptions.onSave(cTrimmed, aFinalExtras.length > 0 ? aFinalExtras : undefined);
    vIsEditing = false;
  }

  function fHandleCancelEdit() {
    vIsEditing = false;
  }

  setMessageEditContext({
    get isEditing() {
      return vIsEditing;
    },
    get editedContent() {
      return vEditedContent;
    },
    get editedExtras() {
      return vEditedExtras;
    },
    get editedUploadedFiles() {
      return vEditedUploadedFiles;
    },
    get originalContent() {
      return pOptions.getContent();
    },
    get originalExtras() {
      return pOptions.getExtras();
    },
    get showSaveOnlyOption() {
      return pOptions.showSaveOnlyOption ?? false;
    },
    get showBranchAfterEditOption() {
      return false;
    },
    get shouldBranchAfterEdit() {
      return false;
    },
    get messageRole() {
      return MessageRole.USER;
    },
    setContent: (pC: string) => {
      vEditedContent = pC;
    },
    setExtras: (pE: DatabaseMessageExtra[]) => {
      vEditedExtras = pE;
    },
    setUploadedFiles: (pF: ChatUploadedFile[]) => {
      vEditedUploadedFiles = pF;
    },
    save: fHandleSaveEdit,
    saveOnly: fHandleSaveEdit,
    cancel: fHandleCancelEdit,
    startEdit: fHandleEdit
  });

  return {
    get isEditing() {
      return vIsEditing;
    },
    handleEdit: fHandleEdit,
    handleSaveEdit: fHandleSaveEdit,
    handleCancelEdit: fHandleCancelEdit
  };
}
