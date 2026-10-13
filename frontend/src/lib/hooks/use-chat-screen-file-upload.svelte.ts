/**
 * File upload lifecycle for the ChatScreen form.
 *
 * Owns the queue of processed `ChatUploadedFile`, the rejection-by-capability
 * dialog state, and the dual-layer validation pipeline (general format +
 * model modality). The caller provides the active model's capabilities and ID
 * as reactive getters so validation tracks the model in real time.
 */

import { processFilesToChatUploaded } from '$lib/utils/browser-only';
import { isFileTypeSupported, filterFilesByModalities } from '$lib/utils';

interface UseChatScreenFileUploadOptions {
  capabilities: () => { hasVision: boolean; hasAudio: boolean; hasVideo: boolean };
  activeModelId: () => string | null | undefined;
}

export interface FileErrorData {
  generallyUnsupported: File[];
  modalityUnsupported: File[];
  modalityReasons: Record<string, string>;
  supportedTypes: string[];
}

export function fUseChatScreenFileUpload(pOptions: UseChatScreenFileUploadOptions) {
  let vUploadedFiles = $state<ChatUploadedFile[]>([]);
  let vShowFileErrorDialog = $state(false);
  let vFileErrorData = $state<FileErrorData>({
    generallyUnsupported: [],
    modalityUnsupported: [],
    modalityReasons: {},
    supportedTypes: []
  });

  async function fProcessFiles(pFiles: File[]) {
    const aGenerallySupported: File[] = [];
    const aGenerallyUnsupported: File[] = [];

    for (const cFile of pFiles) {
      if (isFileTypeSupported(cFile.name, cFile.type)) {
        aGenerallySupported.push(cFile);
      } else {
        aGenerallyUnsupported.push(cFile);
      }
    }

    const {
      supportedFiles: cSupportedFiles,
      unsupportedFiles: cUnsupportedFiles,
      modalityReasons: cModalityReasons
    } = filterFilesByModalities(aGenerallySupported, pOptions.capabilities());

    const aAllUnsupportedFiles = [...aGenerallyUnsupported, ...cUnsupportedFiles];

    if (aAllUnsupportedFiles.length > 0) {
      const aSupportedTypes: string[] = ['text files', 'PDFs'];
      const cCaps = pOptions.capabilities();
      if (cCaps.hasVision) aSupportedTypes.push('images');
      if (cCaps.hasAudio) aSupportedTypes.push('audio files');
      if (cCaps.hasVideo) aSupportedTypes.push('video files');

      vFileErrorData = {
        generallyUnsupported: aGenerallyUnsupported,
        modalityUnsupported: cUnsupportedFiles,
        modalityReasons: cModalityReasons,
        supportedTypes: aSupportedTypes
      };
      vShowFileErrorDialog = true;
    }

    if (cSupportedFiles.length > 0) {
      const cProcessed = await processFilesToChatUploaded(
        cSupportedFiles,
        pOptions.activeModelId() ?? undefined
      );
      vUploadedFiles = [...vUploadedFiles, ...cProcessed];
    }
  }

  function fHandleFileUpload(pFiles: File[]) {
    return fProcessFiles(pFiles);
  }

  function fHandleFileRemove(pFileId: string) {
    vUploadedFiles = vUploadedFiles.filter((pF) => pF.id !== pFileId);
  }

  return {
    get uploadedFiles() {
      return vUploadedFiles;
    },
    set uploadedFiles(pValue) {
      vUploadedFiles = pValue;
    },
    get showFileErrorDialog() {
      return vShowFileErrorDialog;
    },
    set showFileErrorDialog(pValue) {
      vShowFileErrorDialog = pValue;
    },
    get fileErrorData() {
      return vFileErrorData;
    },
    handleFileUpload: fHandleFileUpload,
    handleFileRemove: fHandleFileRemove
  };
}
