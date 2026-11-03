/**
 * URI Template constants for RFC 6570 template processing.
 */

/** URI scheme separator */
export const cUriSchemeSeparator = '://';

/** Regex to match template expressions like {var}, {+var}, {#var}, {/var} */
export const cTemplateExpressionRegex = /\{([+#./;?&]?)([^}]+)\}/g;

/** RFC 6570 URI template operators */
export const cUriTemplateOperators = {
  /** Simple string expansion (default) */
  SIMPLE: '',
  /** Reserved expansion */
  RESERVED: '+',
  /** Fragment expansion */
  FRAGMENT: '#',
  /** Path segment expansion */
  PATH_SEGMENT: '/',
  /** Label expansion */
  LABEL: '.',
  /** Path-style parameters */
  PATH_PARAM: ';',
  /** Form-style query */
  FORM_QUERY: '?',
  /** Form-style query continuation */
  FORM_CONTINUATION: '&'
} as const;

/** URI template separators used in expansion */
export const cUriTemplateSeparators = {
  /** Comma separator for list expansion */
  COMMA: ',',
  /** Slash separator for path segments */
  SLASH: '/',
  /** Period separator for label expansion */
  PERIOD: '.',
  /** Semicolon separator for path parameters */
  SEMICOLON: ';',
  /** Question mark prefix for query string */
  QUERY_PREFIX: '?',
  /** Ampersand prefix for query continuation */
  QUERY_CONTINUATION: '&'
} as const;

/** Maximum number of leading slashes to strip during URI normalization */
export const cMaxLeadingSlashesToStrip = 3;

/** Regex to strip explode modifier (*) from variable names */
export const cVariableExplodeModifierRegex = /[*]$/;

/** Regex to strip prefix modifier (:N) from variable names */
export const cVariablePrefixModifierRegex = /:[\d]+$/;

/** Regex to strip one or more leading slashes */
export const cLeadingSlashesRegex = /^\/+/;

/** Regex to match base64-encoded image URIs (format: "data:image/[media type];base64,[data]")*/
export const cBase64ImageUriRegex = /^data:(image\/[a-z0-9.\-+]+);base64/;
