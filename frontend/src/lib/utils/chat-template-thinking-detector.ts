import { fTranslate } from '../i18n';
/**
 * Detects whether a model's chat template supports thinking/reasoning control.
 *
 * The server "/props" endpoint does NOT expose a supports_thinking flag.
 * It is computed internally by common_chat_templates_support_enable_thinking
 * in common/chat.cpp. A proper server flag would make this unnecessary.
 *
 * Detection order (most reliable first):
 *  1. Thinking-control Jinja2 variables  === pass-through via chat_template_kwargs
 *  2. Thinking-control Jinja2 conditionals  === template-native on/off logic
 *  3. Paired thinking-content tag pairs    === models that output special tags
 */

const aThinkingKwargVars = ['enable_thinking', 'reasoning_effort', 'thinking_budget'];

/**
 * Paired thinking-content tag patterns.
 *
 * Inspected: deepseek-r1/v3, nim-nemotron-{3,4}-nano, qwen-qwq-32b,
 * qwen-3-32b, kimikimi-k2-thinking, apertus-8b-instruct,
 * mistralai-Mistral-Small-3.2-24B, ByteDance-Seed-OSS.
 *
 * The self-closing entry is the Kimi-K2 fixed-length pair,
 * where both tags always appear adjacent with no content between.
 */
const aThinkingTagPatterns: Array<[string, string | null]> = [
  ['<think>', '</think>'],
  ['<|think|>', '</|think|>'],
  ['<seed:think|>', '</seed:think|>'],
  ['<think></think>', null]
];

const aJinjaThinkingConditionals: RegExp[] = [
  // Matches: {% if enable thinking %}, {% if enable_thinking %}, {% if (enable_thinking is defined) %}
  // Handles: underscore-separated (enable_thinking), space-separated (enable thinking),
  // and optional parens/brackets before enable (if (enable_thinking )
  /\{%-?\s*if\s+\(?\s*\w*enable[\s_]+\w*(thinking|think|reasoning)/i,
  /\{%-?\s*if\s+\w*(thinking|reasoning)\s*(is not|==|!=)/i,
  /\{%-?\s*if\s+not\s+\w*enable/i,
  /\{%-?\s*if\s+ns\.enable_thinking/i
];

/** Guards against false positives:
 *  - Generic thought keyword (tool descriptions say "chain of thought")
 *  - Qwen vertical-bar token (used for ALL tool calls, not thinking)
 */
export function fDetectThinkingSupport(pT: string): boolean {
  if (!pT) return false;
  for (const cKwarg of aThinkingKwargVars) {
    const cRegex = new RegExp(
      `(\\{\\{[^{}]*\\b${cKwarg}\\b[^{}]*\\}\\}|\\{%[^{}]*\\b${cKwarg}\\b[^{}]*%\\})`,
      'i'
    );
    if (cRegex.test(pT)) return true;
  }
  for (const cP of aJinjaThinkingConditionals) {
    if (cP.test(pT)) return true;
  }
  for (const [cS, cE] of aThinkingTagPatterns) {
    if (pT.includes(cS) && (!cE || pT.includes(cE))) return true;
  }
  return false;
}

export function fDetectThinkingSupportWithReason(pT: string): {
  supported: boolean;
  reason: string;
} {
  if (!pT) return { supported: false, reason: fTranslate('text_8af1005e8f02') };
  for (const cKwarg of aThinkingKwargVars) {
    const cRegex = new RegExp(
      `(\\{\\{[^{}]*\\b${cKwarg}\\b[^{}]*\\}\\}|\\{%[^{}]*\\b${cKwarg}\\b[^{}]*%\\})`,
      'i'
    );
    if (cRegex.test(pT)) {
      return { supported: true, reason: 'Found: ' + cKwarg };
    }
  }
  for (const cP of aJinjaThinkingConditionals) {
    if (cP.test(pT)) return { supported: true, reason: fTranslate('text_2c45e400e9e9') };
  }
  for (const [cS, cE] of aThinkingTagPatterns) {
    if (pT.includes(cS) && (!cE || pT.includes(cE))) {
      return { supported: true, reason: 'Found: ' + cS + (cE ? ' .. ' + cE : ' (self)') };
    }
  }
  return { supported: false, reason: fTranslate('text_44b181b8b0e5') };
}
