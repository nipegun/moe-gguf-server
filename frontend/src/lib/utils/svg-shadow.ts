/** Marks the element, created inside the host, that carries the shadow root of the svg */
const cSvgShadowRootAttr = 'data-svg-shadow-root';

/**
 * Mounts svg markup inside an open shadow root, under the host element.
 * The shadow boundary scopes the svg <style> and its animations, so model
 * authored css can not reach the surrounding page. That css can still style
 * its own shadow host (:host, even with !important) and position content as
 * fixed, so the shadow root goes on an inner element and the host, which the
 * shadow css never reaches, gets paint containment: whatever the svg draws
 * stays clipped to the host box and can never cover the rest of the page,
 * such as the tool approval buttons. The caller passes markup that is
 * already sanitized, this only isolates and sizes it.
 */
export function fMountSvgShadow(pHost: HTMLElement, pMarkup: string, pStyle: string): void {
  pHost.style.contain = 'paint';

  let vShadowHost = pHost.querySelector<HTMLElement>(`:scope > [${cSvgShadowRootAttr}]`);
  if (!vShadowHost) {
    vShadowHost = document.createElement('div');
    vShadowHost.setAttribute(cSvgShadowRootAttr, '');
    pHost.replaceChildren(vShadowHost);
  }

  const cRoot = vShadowHost.shadowRoot ?? vShadowHost.attachShadow({ mode: 'open' });
  cRoot.innerHTML = pMarkup ? `<style>${pStyle}</style>${pMarkup}` : '';
}
