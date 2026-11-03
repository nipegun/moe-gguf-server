import { fCreatePreTransform } from './pre-transform';
import { cSvgBlockClass, cSvgLanguage, cXmlLanguage, cSvgTagPrefix } from '$lib/constants';

/**
 * Converts svg code blocks to <pre class="svg-block"> for client-side rendering.
 * Also claims xml blocks whose content starts with <svg, since models often emit
 * svg inside an xml fence.
 */
export const cRehypeSvgPre = fCreatePreTransform(
  [cSvgLanguage, cXmlLanguage],
  cSvgBlockClass,
  (pText) => pText.startsWith(cSvgTagPrefix)
);
