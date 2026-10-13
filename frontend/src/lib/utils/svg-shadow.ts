/**
 * Mounts svg markup inside an open shadow root on the host element.
 * The shadow boundary scopes the svg <style> and its animations to the host,
 * so model authored css can not reach the surrounding page. The caller passes
 * markup that is already sanitized, this only isolates and sizes it.
 */
export function fMountSvgShadow(pHost: HTMLElement, pMarkup: string, pStyle: string): void {
  const cRoot = pHost.shadowRoot ?? pHost.attachShadow({ mode: 'open' });
  cRoot.innerHTML = pMarkup ? `<style>${pStyle}</style>${pMarkup}` : '';
}
