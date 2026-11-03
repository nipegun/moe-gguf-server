import { fCreatePreTransform } from './pre-transform';
import { cMermaidBlockClass, cMermaidLanguage } from '$lib/constants';

/**
 * Converts mermaid code blocks to <pre class="mermaid"> for client-side rendering.
 */
export const cRehypeMermaidPre = fCreatePreTransform(cMermaidLanguage, cMermaidBlockClass);
