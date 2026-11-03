/** The server publishes this before the application modules load. */
declare global {
  var cApiPrefix: string | undefined;
}

const cPrefix = typeof globalThis.cApiPrefix === 'string' ? globalThis.cApiPrefix : '/api';

export function fApiUrl(pPath: string): string {
  if (/^https?:\/\//.test(pPath)) return pPath;
  const vPath = pPath.replace(/^\.\//, '/');
  if (vPath === cPrefix || vPath.startsWith(`${cPrefix}/`)) return vPath;
  return `${cPrefix}/${vPath.replace(/^\/+/, '')}`;
}
