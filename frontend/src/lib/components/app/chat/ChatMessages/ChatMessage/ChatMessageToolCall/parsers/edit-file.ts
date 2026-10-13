// Meta parser for `edit_file` tool calls. Reads the file path and the
// array of edits from the streamed args (partial JSON for incremental
// rendering), plus the result blob for `result` / `edits_applied` /
// `error` fields.

import { BuiltInTool } from '$lib/enums';
import { cFilePathSeparatorRegex } from '$lib/constants';
import { tryParseToolResultObject, type AgenticSection } from '$lib/utils';
import { fParseToolArgs } from './_shared';

export type EditFileEdit = {
  oldText: string;
  newText: string;
};

export type EditFileMeta = {
  fileName: string;
  filePath: string;
  edits: EditFileEdit[];
  resultMessage?: string;
  editsApplied?: number;
  errorMessage?: string;
};

export function fParseEditFileMeta(pSection: AgenticSection): EditFileMeta | null {
  const cArgs = fParseToolArgs(BuiltInTool.EDIT_FILE, pSection, { partial: true });
  if (!cArgs) return null;

  const cRawPath = cArgs.path ?? cArgs.file_path ?? cArgs.filePath;
  if (typeof cRawPath !== 'string' || !cRawPath) return null;

  const cFileName = cRawPath.split(cFilePathSeparatorRegex).pop() || cRawPath;

  // Filter the streamed edits array strictly: each entry must be an
  // object with a non-empty `old_text`. Edits without an old_text
  // would diff against empty and render as a full re-write.
  const cRawEdits = Array.isArray(cArgs.edits) ? cArgs.edits : [];
  const aEdits: EditFileEdit[] = [];
  for (const cE of cRawEdits) {
    if (!cE || typeof cE !== 'object' || Array.isArray(cE)) continue;
    const cObj = cE as Record<string, unknown>;
    const cOldText = typeof cObj.old_text === 'string' ? cObj.old_text : '';
    if (!cOldText) continue;
    const cNewText = typeof cObj.new_text === 'string' ? cObj.new_text : '';
    aEdits.push({ oldText: cOldText, newText: cNewText });
  }

  const cResultObj = tryParseToolResultObject(pSection.toolResult);
  let vResultMessage: string | undefined;
  let vEditsApplied: number | undefined;
  let vErrorMessage: string | undefined;
  if (typeof cResultObj?.error === 'string') {
    vErrorMessage = cResultObj.error;
  } else if (cResultObj) {
    if (typeof cResultObj.result === 'string') {
      vResultMessage = cResultObj.result;
    }
    if (Number.isFinite(Number(cResultObj.edits_applied))) {
      vEditsApplied = Number(cResultObj.edits_applied);
    }
  }

  return {
    fileName: cFileName,
    filePath: cRawPath,
    edits: aEdits,
    resultMessage: vResultMessage,
    editsApplied: vEditsApplied,
    errorMessage: vErrorMessage
  };
}
