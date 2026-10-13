/**
 * Line-level unified diff for tool result rendering.
 *
 * Pure functions: no DOM, no Svelte, no highlight.js dependency. The
 * returned `DiffLine[]` carries enough information both to render a
 * custom diff block (per-entry kind/text) and to fold back into a
 * unified-diff-format string for off-the-shelf highlighter languages
 * (`renderUnifiedDiff`).
 *
 * Algorithm: LCS dynamic programming with a soft "remove before add"
 * tiebreak so the resulting diff reads `(old -> new)` left to right.
 * O(m*n) time/space which is fine for the handful of lines an
 * `edit_file` snippet typically carries.
 */

import { DiffLineKind } from '$lib/enums';

export interface DiffLine {
  kind: DiffLineKind;
  text: string;
  /** 1-indexed line number in the OLD content. Undefined for `add` lines. */
  oldLine?: number;
  /** 1-indexed line number in the NEW content. Undefined for `remove` lines. */
  newLine?: number;
}

export function fComputeLineDiff(pOldText: string, pNewText: string): DiffLine[] {
  const cOldLines = fSplitLines(pOldText);
  const cNewLines = fSplitLines(pNewText);

  const cM = cOldLines.length;
  const cN = cNewLines.length;

  if (cM === 0 && cN === 0) return [];
  if (cM === 0)
    return cNewLines.map((pT, pK) => ({ kind: DiffLineKind.ADD, text: pT, newLine: pK + 1 }));
  if (cN === 0)
    return cOldLines.map((pT, pK) => ({ kind: DiffLineKind.REMOVE, text: pT, oldLine: pK + 1 }));

  const aLcs: number[][] = Array.from({ length: cM + 1 }, () => new Array(cN + 1).fill(0));
  for (let vI = 1; vI <= cM; vI++) {
    for (let vJ = 1; vJ <= cN; vJ++) {
      if (cOldLines[vI - 1] === cNewLines[vJ - 1]) {
        aLcs[vI][vJ] = aLcs[vI - 1][vJ - 1] + 1;
      } else {
        aLcs[vI][vJ] = Math.max(aLcs[vI - 1][vJ], aLcs[vI][vJ - 1]);
      }
    }
  }

  const aResult: DiffLine[] = [];
  let vI = cM;
  let vJ = cN;
  while (vI > 0 && vJ > 0) {
    if (cOldLines[vI - 1] === cNewLines[vJ - 1]) {
      aResult.push({
        kind: DiffLineKind.CONTEXT,
        text: cOldLines[vI - 1],
        oldLine: vI,
        newLine: vJ
      });
      vI--;
      vJ--;
    } else if (aLcs[vI - 1][vJ] >= aLcs[vI][vJ - 1]) {
      aResult.push({ kind: DiffLineKind.REMOVE, text: cOldLines[vI - 1], oldLine: vI });
      vI--;
    } else {
      aResult.push({ kind: DiffLineKind.ADD, text: cNewLines[vJ - 1], newLine: vJ });
      vJ--;
    }
  }
  while (vI > 0) {
    aResult.push({ kind: DiffLineKind.REMOVE, text: cOldLines[vI - 1], oldLine: vI });
    vI--;
  }
  while (vJ > 0) {
    aResult.push({ kind: DiffLineKind.ADD, text: cNewLines[vJ - 1], newLine: vJ });
    vJ--;
  }

  aResult.reverse();
  return aResult;
}

/** Folds `DiffLine[]` into a unified-diff-format text (`` ` ``/`+`/`-` prefixes).
 *  Pass to a diff-aware highlighter (e.g., SyntaxHighlightedCode with
 *  `language="diff"`) for colorization.
 */
export function fRenderUnifiedDiff(pLines: DiffLine[]): string {
  if (pLines.length === 0) return '';
  return pLines.map((pL) => fPrefixFor(pL.kind) + pL.text).join('\n');
}

/** Column-1 marker for a `DiffLine`: ` `, `+`, or `-`. */
export function fPrefixFor(pKind: DiffLineKind): string {
  if (pKind === DiffLineKind.ADD) return '+';
  if (pKind === DiffLineKind.REMOVE) return '-';
  return ' ';
}

function fSplitLines(pText: string): string[] {
  if (pText === '') return [];
  const cParts = pText.split('\n');
  if (cParts[cParts.length - 1] === '') cParts.pop();
  return cParts.map((pL) => (pL.endsWith('\r') ? pL.slice(0, -1) : pL));
}
