// Meta parser for `grep_search` tool calls. Reads the path/pattern
// triplet from args (strict parsing - we wait for the args to
// complete) and the matches from the result blob. The result parser
// keeps the original "scan result as raw text on JSON.parse failure"
// fallback so MCP servers that return unparseable output still get
// surfaced.

import { BuiltInTool } from '$lib/enums';
import { splitSearchSummaryList, type AgenticSection } from '$lib/utils';
import { fParseToolArgs } from './_shared';

export type GrepSearchMatch = {
  file: string;
  line?: number;
  content: string;
};

export type GrepSearchMeta = {
  path: string;
  pattern: string;
  include: string;
  exclude?: string;
  showLineNumbers: boolean;
  matches: GrepSearchMatch[];
  totalMatches?: number;
  errorMessage?: string;
};

export function fParseGrepSearchMeta(pSection: AgenticSection): GrepSearchMeta | null {
  const cArgs = fParseToolArgs(BuiltInTool.GREP_SEARCH, pSection);
  if (!cArgs) return null;

  const cPath = typeof cArgs.path === 'string' ? cArgs.path : '';
  const cPattern = typeof cArgs.pattern === 'string' ? cArgs.pattern : '';
  if (!cPath || !cPattern) return null;

  const cInclude = typeof cArgs.include === 'string' && cArgs.include ? cArgs.include : '**';
  const cExclude = typeof cArgs.exclude === 'string' && cArgs.exclude ? cArgs.exclude : undefined;
  const cShowLineNumbers = cArgs.return_line_numbers === true;

  let aMatches: GrepSearchMatch[] = [];
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
          aMatches = cSplit.lines.map((pLine) => fParseGrepLine(pLine, cShowLineNumbers));
        }
      }
    } catch {
      // Result wasn't JSON: keep behaviour for MCP servers that
      // emit raw text and treat each line as a `<file>:<content>`
      // (or `<file>:<line>:<content>`) match.
      const cSplit = splitSearchSummaryList(cToolResultString, (pTotal) => {
        vTotalMatches = pTotal;
      });
      aMatches = cSplit.lines.map((pLine) => fParseGrepLine(pLine, cShowLineNumbers));
    }
  }

  return {
    path: cPath,
    pattern: cPattern,
    include: cInclude,
    exclude: cExclude,
    showLineNumbers: cShowLineNumbers,
    matches: aMatches,
    totalMatches: vTotalMatches,
    errorMessage: vErrorMessage
  };
}

function fParseGrepLine(pLine: string, pShowLineNumbers: boolean): GrepSearchMatch {
  // Server output:
  //   <file>:<content>          when return_line_numbers=false
  //   <file>:<lineno>:<content> when return_line_numbers=true
  const cFirstColon = pLine.indexOf(':');
  if (cFirstColon === -1) {
    return { file: pLine, content: '' };
  }
  const cFile = pLine.slice(0, cFirstColon);
  const cTail = pLine.slice(cFirstColon + 1);

  if (!pShowLineNumbers) {
    return { file: cFile, content: cTail };
  }

  const cSecondColon = cTail.indexOf(':');
  if (cSecondColon === -1) {
    return { file: cFile, content: cTail };
  }
  const cLineNum = parseInt(cTail.slice(0, cSecondColon), 10);
  return {
    file: cFile,
    line: Number.isFinite(cLineNum) ? cLineNum : undefined,
    content: cTail.slice(cSecondColon + 1)
  };
}
