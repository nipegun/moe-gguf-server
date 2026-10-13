export function fPortalToBody(pNode: HTMLElement) {
  if (typeof document === 'undefined') {
    return;
  }

  const cTarget = document.body;
  if (!cTarget) {
    return;
  }

  cTarget.appendChild(pNode);

  return {
    destroy() {
      if (pNode.parentNode === cTarget) {
        cTarget.removeChild(pNode);
      }
    }
  };
}
