import {
  cTemplateExpressionRegex,
  cUriSchemeSeparator,
  cUriTemplateOperators,
  cUriTemplateSeparators,
  cVariableExplodeModifierRegex,
  cVariablePrefixModifierRegex,
  cLeadingSlashesRegex
} from '../constants';

/**
 * Normalize a resource URI for comparison.
 *
 * URI template expansion (especially with path operators like {/var})
 * can produce URIs that differ from listed resource URIs in slash placement.
 * For example, the template `svelte://{/slug*}.md` with slug="svelte/$effect"
 * expands to `svelte:///svelte/$effect.md`, while the listed resource URI is
 * `svelte://svelte/$effect.md`.
 *
 * This function strips extra leading slashes after the scheme to normalize
 * both forms to the same string for comparison purposes.
 *
 * @param pUri - The URI to normalize
 * @returns Normalized URI string
 */
export function fNormalizeResourceUri(pUri: string): string {
  const cSchemeEnd = pUri.indexOf(cUriSchemeSeparator);
  if (cSchemeEnd === -1) return pUri;

  const cScheme = pUri.substring(0, cSchemeEnd);
  const cRest = pUri
    .substring(cSchemeEnd + cUriSchemeSeparator.length)
    .replace(cLeadingSlashesRegex, '');

  return `${cScheme}${cUriSchemeSeparator}${cRest}`;
}

/**
 * A parsed variable from a URI template expression.
 */
export interface UriTemplateVariable {
  /** Variable name */
  name: string;
  /** Operator prefix (+, #, /, etc.) or empty string */
  operator: string;
}

/**
 * Extract all variable names from a URI template string.
 *
 * @param pTemplate - URI template string (RFC 6570)
 * @returns Array of unique variable descriptors
 *
 * @example
 * ```ts
 * extractTemplateVariables("file:///{path}")
 * // => [{ name: "path", operator: "" }]
 *
 * extractTemplateVariables("db://{schema}/{table}")
 * // => [{ name: "schema", operator: "" }, { name: "table", operator: "" }]
 * ```
 */
export function fExtractTemplateVariables(pTemplate: string): UriTemplateVariable[] {
  const aVariables: UriTemplateVariable[] = [];
  const sSeen = new Set<string>();

  let vMatch;
  cTemplateExpressionRegex.lastIndex = 0;

  while ((vMatch = cTemplateExpressionRegex.exec(pTemplate)) !== null) {
    const cOperator = vMatch[1] || '';
    const cVarList = vMatch[2];

    // RFC 6570 allows comma-separated variable lists: {x,y,z}
    for (const cVarSpec of cVarList.split(',')) {
      // Strip explode modifier (*) and prefix modifier (:N)
      const cName = cVarSpec
        .replace(cVariableExplodeModifierRegex, '')
        .replace(cVariablePrefixModifierRegex, '')
        .trim();

      if (cName && !sSeen.has(cName)) {
        sSeen.add(cName);
        aVariables.push({ name: cName, operator: cOperator });
      }
    }
  }

  return aVariables;
}

/**
 * Expand a URI template with the given variable values.
 * Implements a simplified RFC 6570 Level 2 expansion.
 *
 * @param pTemplate - URI template string
 * @param pValues - Map of variable name to value
 * @returns Expanded URI string
 *
 * @example
 * ```ts
 * expandTemplate("file:///{path}", { path: "src/main.rs" })
 * // => "file:///src/main.rs"
 * ```
 */
export function fExpandTemplate(pTemplate: string, pValues: Record<string, string>): string {
  cTemplateExpressionRegex.lastIndex = 0;

  return pTemplate.replace(
    cTemplateExpressionRegex,
    (pMatch, pOperator: string, pVarList: string) => {
      const cVarNames = pVarList
        .split(',')
        .map((pV: string) =>
          pV
            .replace(cVariableExplodeModifierRegex, '')
            .replace(cVariablePrefixModifierRegex, '')
            .trim()
        );

      const cExpandedParts = cVarNames
        .map((pName: string) => pValues[pName] ?? '')
        .filter((pV: string) => pV !== '');

      if (cExpandedParts.length === 0) return '';

      switch (pOperator) {
        case cUriTemplateOperators.RESERVED:
          // Reserved expansion: no encoding
          return cExpandedParts.join(cUriTemplateSeparators.COMMA);
        case cUriTemplateOperators.FRAGMENT:
          // Fragment expansion
          return cUriTemplateOperators.FRAGMENT + cExpandedParts.join(cUriTemplateSeparators.COMMA);
        case cUriTemplateOperators.PATH_SEGMENT:
          // Path segments
          return cUriTemplateSeparators.SLASH + cExpandedParts.join(cUriTemplateSeparators.SLASH);
        case cUriTemplateOperators.LABEL:
          // Label expansion
          return cUriTemplateSeparators.PERIOD + cExpandedParts.join(cUriTemplateSeparators.PERIOD);
        case cUriTemplateOperators.PATH_PARAM:
          // Path-style parameters
          return cVarNames
            .filter((pIgnored: string, pI: number) => cExpandedParts[pI])
            .map(
              (pName: string, pI: number) =>
                `${cUriTemplateSeparators.SEMICOLON}${pName}=${cExpandedParts[pI]}`
            )
            .join('');
        case cUriTemplateOperators.FORM_QUERY:
          // Form-style query
          return (
            cUriTemplateSeparators.QUERY_PREFIX +
            cVarNames
              .filter((pIgnored: string, pI: number) => cExpandedParts[pI])
              .map(
                (pName: string, pI: number) =>
                  `${encodeURIComponent(pName)}=${encodeURIComponent(cExpandedParts[pI])}`
              )
              .join(cUriTemplateSeparators.QUERY_CONTINUATION)
          );
        case cUriTemplateOperators.FORM_CONTINUATION:
          // Form-style query continuation
          return (
            cUriTemplateSeparators.QUERY_CONTINUATION +
            cVarNames
              .filter((pIgnored: string, pI: number) => cExpandedParts[pI])
              .map(
                (pName: string, pI: number) =>
                  `${encodeURIComponent(pName)}=${encodeURIComponent(cExpandedParts[pI])}`
              )
              .join(cUriTemplateSeparators.COMMA)
          );
        default:
          // Simple string expansion (default operator)
          return cExpandedParts
            .map((pV: string) => encodeURIComponent(pV))
            .join(cUriTemplateSeparators.COMMA);
      }
    }
  );
}

/**
 * Check whether all required variables in a template have been provided.
 *
 * @param pTemplate - URI template string
 * @param pValues - Map of variable name to value
 * @returns true if all variables have non-empty values
 */
export function fIsTemplateComplete(pTemplate: string, pValues: Record<string, string>): boolean {
  const cVariables = fExtractTemplateVariables(pTemplate);

  return cVariables.every((pV) => (pValues[pV.name] ?? '').trim() !== '');
}
