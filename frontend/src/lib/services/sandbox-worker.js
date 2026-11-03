const aLogs = [];
const fFmt = (pValue) => {
  if (typeof pValue === 'string') return pValue;
  try {
    return JSON.stringify(pValue);
  } catch {
    return String(pValue);
  }
};
const fCapture =
  (pLevel, pPrefix) =>
  (...pArgs) => {
    aLogs.push(pPrefix + pArgs.map(fFmt).join(' '));
  };
console.log = fCapture('log', '');
console.info = fCapture('info', '');
console.debug = fCapture('debug', '');
console.warn = fCapture('warn', 'warn: ');
console.error = fCapture('error', 'error: ');
self.onmessage = async (pEvent) => {
  const dReply = { logs: aLogs, result: null, error: null };
  try {
    const cAsyncFunction = Object.getPrototypeOf(async function () {}).constructor;
    // The prelude bundled ahead of this shim defines self.nerdamer,
    // passed into the execution scope as the `nerdamer` parameter.
    const cValue = await new cAsyncFunction('nerdamer', pEvent.data.code)(self.nerdamer);
    if (cValue !== undefined) dReply.result = fFmt(cValue);
  } catch (vErr) {
    dReply.error = vErr instanceof Error ? vErr.stack || vErr.message : String(vErr);
  }
  self.postMessage(dReply);
};
