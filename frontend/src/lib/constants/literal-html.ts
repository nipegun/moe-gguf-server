export const cLineBreak = /\r?\n/;

export const sPhraseParents = new Set([
  'paragraph',
  'heading',
  'emphasis',
  'strong',
  'delete',
  'link',
  'linkReference',
  'tableCell'
]);

export const cNbsp = '\u00a0';
export const cTabAsSpaces = cNbsp.repeat(4);
