// Meta parser for `file_glob_search` tool calls. Reads the path,
// include pattern, and optional exclude from the args (strict parsing)
// and the matches from the result blob. Like grep_search, the result
// parser keeps the original raw-text fallback for MCP servers that
// emit unparseable output.

import { BuiltInTool } from '$lib/enums';
import { splitSearchSummaryList, type AgenticSection } from '$lib/utils';
import { fParseToolArgs } from './_shared';

export type FileGlobSearchMeta = {
  path: string;
  include: string;
  exclude?: string;
  matches: string[];
  totalMatches?: number;
  errorMessage?: string;
};

export function fParseFileGlobSearchMeta(pSection: AgenticSection): FileGlobSearchMeta | null {
  const cArgs = fParseToolArgs(BuiltInTool.FILE_GLOB_SEARCH, pSection);
  if (!cArgs) return null;

  const cPath = typeof cArgs.path === 'string' ? cArgs.path : '';
  const cInclude = typeof cArgs.include === 'string' && cArgs.include ? cArgs.include : '**';
  const cExclude = typeof cArgs.exclude === 'string' && cArgs.exclude ? cArgs.exclude : undefined;
  if (!cPath) return null;

  let aMatches: string[] = [];
  let vTotalMatches: number | undefined;
  let vErrorMessage: string | undefined;

  const cToolResultString = pSection.toolResult;
  if (cToolResultString) {
    try {
      const cParsed: unknown = JSON.parse(cToolResultString);
      if (cParsed && typeof cParsed === 'object' && !Array.isArray(cParsed)) {
        const cObj = cParsed as Record<string, unknown>;
        if (typeof cObj.error === 'string') {
          vErrorMessage = cObj.error;
        } else if (typeof cObj.plain_text_response === 'string') {
          const cSplit = splitSearchSummaryList(cObj.plain_text_response, (pTotal) => {
            vTotalMatches = pTotal;
          });
          aMatches = cSplit.lines;
        }
      }
    } catch {
      // See grep-search.ts: same fallback used there.
      const cSplit = splitSearchSummaryList(cToolResultString, (pTotal) => {
        vTotalMatches = pTotal;
      });
      aMatches = cSplit.lines;
    }
  }

  return {
    path: cPath,
    include: cInclude,
    exclude: cExclude,
    matches: aMatches,
    totalMatches: vTotalMatches,
    errorMessage: vErrorMessage
  };
}
