import { fTranslate } from '../i18n';
import { fIsHttpUrl } from './safe-url';

/**
 * Builds the fallback shown in place of an image that failed to load.
 * Displays a centered message and, only for http(s) sources, a link to open
 * the image in a new tab. Built with DOM properties so the source never
 * becomes markup and a non-http(s) scheme never becomes a link.
 */
export function fCreateImageErrorFallback(pSrc: string): HTMLDivElement {
  const cContent = document.createElement('div');
  cContent.className = 'image-error-content';

  const cMessage = document.createElement('span');
  cMessage.textContent = fTranslate('image.unavailable');
  cContent.appendChild(cMessage);

  if (fIsHttpUrl(pSrc)) {
    const cLink = document.createElement('a');
    cLink.href = pSrc;
    cLink.target = '_blank';
    cLink.rel = 'noopener noreferrer';
    cLink.textContent = fTranslate('image.openLink');
    cContent.appendChild(cLink);
  }

  return cContent;
}
