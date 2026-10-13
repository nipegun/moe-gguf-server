// Meta parser for `write_file` tool calls. Reads the path/content from
// the streamed args (partial JSON so we can render before the call
// finishes) and surfaces `bytes`, `result`, and `error` from the
// result blob.

import { BuiltInTool } from '$lib/enums';
import {
  cDefaultLanguage,
  cFilePathSeparatorRegex,
  cTextLanguagePrefixRegex
} from '$lib/constants';
import { getFileTypeByExtension, tryParseToolResultObject, type AgenticSection } from '$lib/utils';
import { fParseToolArgs } from './_shared';

export type WriteFileMeta = {
  fileName: string;
  filePath: string;
  language: string;
  content: string;
  bytesWritten?: number;
  resultMessage?: string;
  errorMessage?: string;
};

export function fParseWriteFileMeta(pSection: AgenticSection): WriteFileMeta | null {
  const cArgs = fParseToolArgs(BuiltInTool.WRITE_FILE, pSection, { partial: true });
  if (!cArgs) return null;

  // Tool contracts drifted over time: some models emit `path`,
  // others `file_path` / `filePath`. Accept all three.
  const cRawPath = cArgs.path ?? cArgs.file_path ?? cArgs.filePath;
  if (typeof cRawPath !== 'string' || !cRawPath) return null;

  const cFileName = cRawPath.split(cFilePathSeparatorRegex).pop() || cRawPath;
  const cContent = typeof cArgs.content === 'string' ? cArgs.content : '';
  const cLanguage =
    getFileTypeByExtension(cRawPath)?.replace(cTextLanguagePrefixRegex, '') ?? cDefaultLanguage;

  const cResultObj = tryParseToolResultObject(pSection.toolResult);
  const cBytesWritten =
    cResultObj && Number.isFinite(Number(cResultObj.bytes)) ? Number(cResultObj.bytes) : undefined;
  const cResultMessage = typeof cResultObj?.result === 'string' ? cResultObj.result : undefined;
  const cErrorMessage = typeof cResultObj?.error === 'string' ? cResultObj.error : undefined;

  return {
    fileName: cFileName,
    filePath: cRawPath,
    language: cLanguage,
    content: cContent,
    bytesWritten: cBytesWritten,
    resultMessage: cResultMessage,
    errorMessage: cErrorMessage
  };
}
