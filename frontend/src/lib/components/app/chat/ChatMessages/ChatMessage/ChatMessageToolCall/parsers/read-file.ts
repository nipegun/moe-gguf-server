// Meta parser for `read_file` tool calls. Reads the file path and an
// optional line range (either `start_line`+`end_line` or
// `start_line`+`line_count`). Args are parsed partially so a header
// can render incrementally as the file path streams in.

import { BuiltInTool } from '$lib/enums';
import {
  cDefaultLanguage,
  cFilePathSeparatorRegex,
  cTextLanguagePrefixRegex
} from '$lib/constants';
import { getFileTypeByExtension, type AgenticSection } from '$lib/utils';
import { fParseToolArgs } from './_shared';

export type ReadFileMeta = {
  fileName: string;
  lineRange: { start: number; end: number } | null;
  language: string;
};

export function fParseReadFileMeta(pSection: AgenticSection): ReadFileMeta | null {
  const cArgs = fParseToolArgs(BuiltInTool.READ_FILE, pSection, { partial: true });
  if (!cArgs) return null;

  const cRawPath = cArgs.path ?? cArgs.file_path ?? cArgs.filePath;
  if (typeof cRawPath !== 'string' || !cRawPath) return null;

  const cFileName = cRawPath.split(cFilePathSeparatorRegex).pop() || cRawPath;

  // Models emit range arguments under several aliases. Accept all to
  // stay forgiving across prompt variations.
  const cStartRaw = cArgs.start_line ?? cArgs.line_start ?? cArgs.startLine ?? cArgs.from_line;
  const cEndRaw = cArgs.end_line ?? cArgs.line_end ?? cArgs.endLine ?? cArgs.to_line;
  const cCountRaw = cArgs.line_count ?? cArgs.count ?? cArgs.num_lines;

  let vLineRange: { start: number; end: number } | null = null;
  const sNum = Number(cStartRaw);
  const cENum = Number(cEndRaw);
  if (cStartRaw != null && cEndRaw != null && Number.isFinite(sNum) && Number.isFinite(cENum)) {
    vLineRange = { start: sNum, end: cENum };
  } else if (cStartRaw != null && cCountRaw != null) {
    const cNum = Number(cCountRaw);
    if (Number.isFinite(sNum) && Number.isFinite(cNum)) {
      vLineRange = { start: sNum, end: sNum + cNum - 1 };
    }
  }

  const cFileType = getFileTypeByExtension(cFileName);
  const cLanguage = cFileType ? cFileType.replace(cTextLanguagePrefixRegex, '') : cDefaultLanguage;

  return { fileName: cFileName, lineRange: vLineRange, language: cLanguage };
}
