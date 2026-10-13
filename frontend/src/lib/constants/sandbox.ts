import { fTranslate } from '../i18n';
import { BuiltInTool, JsonSchemaType, ToolCallType } from '$lib/enums';
import type { OpenAIToolDefinition } from '$lib/types';

export const cSandboxToolName = BuiltInTool.RUN_JAVASCRIPT;

export const cSandboxTimeoutMsDefault = 10000;

export const cSandboxTimeoutMsMax = 30000;

export const cSandboxOutputMaxChars = 8192;

export const cSandboxEmptyOutput = '(no output)';

export const cSandboxTruncationNotice = '[output truncated]';

const cNerdamerDescription = `
Symbolic/numeric math via \`nerdamer\`
nerdamer(expr,subs?,opts?)/nerdamer.func(...)→Expression Format via .text(fmt?) (fmt: 'decimals'|'fractions'|'scientific') eval via .evaluate(subs?)
nerdamer(expr,{x:2}) substitutes numeric via opts 'numer' or .evaluate()
simplify/expand/factor(expr) div/gcd/lcm(...) coeffs/partfrac(expr,var)
diff/integrate(expr,var) defint(expr,lo,hi,var?) sum/product(expr,var,lo,hi) limit(expr,var,pt)
solve(expr,var) solveEquations([eq1,eq2],[var1,var2])
polarform/rectform/arg/realpart/imagpart(z)
set/get Var/Constant(name,val?) setFunction(name,[params],body)
IMPORTANT:Identifier 'nerdamer' has already been declared, use it directly`;

/**
 * Build the sandbox tool definition. When `includeSymbolicMath` is true,
 * the description includes nerdamer API documentation; otherwise it
 * describes a plain JavaScript sandbox.
 */
export function fBuildSandboxToolDefinition(pIncludeSymbolicMath: boolean): OpenAIToolDefinition {
  return {
    type: ToolCallType.FUNCTION,
    function: {
      name: cSandboxToolName,
      description: pIncludeSymbolicMath
        ? fTranslate('text_999011ebffd3', { p0: cNerdamerDescription })
        : fTranslate('text_871bcbcae6f4'),
      parameters: {
        type: JsonSchemaType.OBJECT,
        properties: {
          code: {
            type: JsonSchemaType.STRING,
            description: fTranslate('text_8b54c4c622a2')
          },
          timeout_ms: {
            type: JsonSchemaType.NUMBER,
            description: fTranslate('text_d0cd0e2ee0fc', {
              p0: cSandboxTimeoutMsDefault,
              p1: cSandboxTimeoutMsMax
            })
          }
        },
        required: ['code']
      }
    }
  };
}
