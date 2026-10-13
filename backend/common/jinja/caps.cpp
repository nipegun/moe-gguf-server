#include "value.h"
#include "runtime.h"
#include "caps.h"

// note: the json dependency is only for defining input in a convenient way
// we can remove it in the future when we figure out a better way to define inputs using jinja::value
#include <nlohmann/json.hpp>

#include <functional>
#include <sstream>

#define FILENAME "jinja-caps"

using json = nlohmann::ordered_json;

namespace jinja {

using caps_json_fn = std::function<json()>;
using caps_ctx_fn = std::function<void(context &)>;
using caps_analyze_fn = std::function<void(bool, value &, value &, const std::string &)>;

void fCapsApplyPreserveReasoning(jinja::context &pCtx, bool pEnabled) {
  pCtx.mSetVal("preserve_thinking", fMkVal<value_bool>(pEnabled));
  pCtx.mSetVal("clear_thinking", fMkVal<value_bool>(!pEnabled));
  pCtx.mSetVal("truncate_history_thinking", fMkVal<value_bool>(!pEnabled));
  pCtx.mSetVal("drop_thinking", fMkVal<value_bool>(!pEnabled));
}

static void fCapsTryExecute(jinja::program &pProg, const caps_json_fn &pMessagesFn, const caps_ctx_fn &pCtxFn,
                            const caps_json_fn &pToolsFn, const caps_analyze_fn &pAnalyzeFn) {
  context vCtx;
  vCtx.vIsGetStats = true;
  jinja::fGlobalFromJson(vCtx,
                         json{{"messages", pMessagesFn()},
                              {"tools", pToolsFn ? pToolsFn() : json::array()},
                              {"bos_token", ""},
                              {"eos_token", ""},
                              {"add_generation_prompt", true}},
                         true);

  if (pCtxFn) {
    pCtxFn(vCtx);
  }

  auto vMessages = vCtx.mGetVal("messages");
  auto vTools = vCtx.mGetVal("tools");

  bool vSuccess = false;
  std::string vResult;
  try {
    jinja::runtime vRuntime(vCtx);
    auto vResults = vRuntime.mExecute(pProg);
    auto vParts = jinja::runtime::mGatherStringParts(vResults);
    vResult = vParts->mAsString().mStr();
    vSuccess = true;
  } catch (const std::exception &vE) {
    JJ_DEBUG("Exception during execution: %s", vE.what());
    vResult = "";
    // ignore exceptions during capability analysis
  }

  pAnalyzeFn(vSuccess, vMessages, vTools, vResult);
}

// for debugging only
static void fCapsPrintStats(value &pV, const std::string &pPath) {
  std::string vOps;
  for (const auto &vName : pV->vStats.sOps) {
    vOps += vName + " ";
  }
  JJ_DEBUG("Value %s, type: %s %s, ops: %s", pPath.c_str(), pV->mType().c_str(), pV->vStats.vUsed ? "(used)" : "",
           vOps.c_str());
}

std::map<std::string, bool> caps::mToMap() const {
  return {
      {"supports_string_content", vSupportsStringContent},
      {"supports_typed_content", vSupportsTypedContent},
      {"supports_tools", vSupportsTools},
      {"supports_tool_calls", vSupportsToolCalls},
      {"supports_parallel_tool_calls", vSupportsParallelToolCalls},
      {"supports_system_role", vSupportsSystemRole},
      {"supports_preserve_reasoning", vSupportsPreserveReasoning},
      {"supports_object_arguments", vSupportsObjectArguments},
  };
}

std::string caps::mToString() const {
  std::ostringstream vSs;
  vSs << "Caps(\n";
  for (const auto &[vKey, vValue] : mToMap()) {
    vSs << "  " << vKey << "=" << (vValue ? "true" : "false") << "\n";
  }
  vSs << ")";
  return vSs.str();
}

caps fCapsGet(jinja::program &pProg) {
  caps vResult;

  static const auto cHasOp = [](value &pV, const std::string &pOpName) {
    return pV->vStats.sOps.find(pOpName) != pV->vStats.sOps.end();
  };

  JJ_DEBUG("%s\n", ">>> Running capability check: typed content");

  // case: typed content support
  fCapsTryExecute(
      pProg,
      [&]() {
        // messages
        return json::array({{{"role", "user"}, {"content", "content"}}});
      },
      nullptr, // ctx_fn
      nullptr, // tools_fn
      [&](bool pSuccess, value &pMessages, value &, const std::string &) {
        auto &vContent = pMessages->mAt(0)->mAt("content");
        fCapsPrintStats(vContent, "messages[0].content");
        if (cHasOp(vContent, "selectattr") || cHasOp(vContent, "array_access")) {
          // accessed as an array
          vResult.vSupportsTypedContent = true;
        }
        if (!pSuccess) {
          // failed to execute with content as string
          vResult.vSupportsStringContent = false;
        }
      });

  JJ_DEBUG("%s\n", ">>> Running capability check: system prompt");

  // case: system prompt support
  fCapsTryExecute(
      pProg,
      [&]() {
        // messages
        return json::array({
            {{"role", "system"}, {"content", "System message"}},
            {{"role", "user"}, {"content", "User message"}},
        });
      },
      nullptr, // ctx_fn
      nullptr, // tools_fn
      [&](bool, value &pMessages, value &, const std::string &) {
        auto &vContent = pMessages->mAt(0)->mAt("content");
        fCapsPrintStats(vContent, "messages[0].content");
        if (!vContent->vStats.vUsed) {
          vResult.vSupportsSystemRole = false;
        }
      });

  JJ_DEBUG("%s\n", ">>> Running capability check: single tool with object arguments support");

  // case: tools support: single call with object arguments
  fCapsTryExecute(
      pProg,
      [&]() {
        // messages
        return json::array({
            {
                {"role", "user"},
                {"content", "User message"},
            },
            {{"role", "assistant"},
             {"content", ""}, // Some templates expect content to be empty with tool calls
             {"tool_calls", json::array({{{"id", "call00001"},
                                          {"type", "function"},
                                          {"function", {{"name", "tool1"}, {"arguments", {{"arg", "value"}}}}}}})}},
            {{"role", "tool"}, {"content", "Tool response"}, {"tool_call_id", "call00001"}},
            {{"role", "assistant"}, {"content", "The tool response was 'tool response'"}},
            {
                {"role", "user"},
                {"content", "User message"},
            },
        });
      },
      nullptr, // ctx_fn
      [&]() {
        // tools
        return json::array({
            {
                {"name", "tool"},
                {"type", "function"},
                {"function",
                 {
                     {"name", "tool1"},
                     {"description", "Tool description"},
                     {"parameters",
                      {
                          {"type", "object"},
                          {"properties",
                           {
                               {"arg",
                                {
                                    {"type", "string"},
                                    {"description", "Arg description"},
                                }},
                           }},
                          {"required", json::array({"arg"})},
                      }},
                 }},
            },
        });
      },
      [&](bool pSuccess, value &pMessages, value &pTools, const std::string &) {
        if (!pSuccess) {
          return; // Nothing can be inferred
        }

        auto &vToolName = pTools->mAt(0)->mAt("function")->mAt("name");
        fCapsPrintStats(vToolName, "tools[0].function.name");
        fCapsPrintStats(pTools, "tools");
        if (!vToolName->vStats.vUsed) {
          vResult.vSupportsTools = false;
        }

        auto &vToolCalls = pMessages->mAt(1)->mAt("tool_calls");
        ;
        fCapsPrintStats(vToolCalls, "messages[1].tool_calls");
        if (!vToolCalls->vStats.vUsed) {
          vResult.vSupportsToolCalls = false;
          return;
        }

        auto &vToolArg = vToolCalls->mAt(0)->mAt("function")->mAt("arguments")->mAt("arg");
        fCapsPrintStats(vToolArg, "messages[1].tool_calls[0].function.arguments.arg");
        if (vToolArg->vStats.vUsed) {
          vResult.vSupportsObjectArguments = true;
        }
      });

  if (!vResult.vSupportsObjectArguments) {
    JJ_DEBUG("%s\n", ">>> Running capability check: single tool with string arguments support");

    // case: tools support: single call with string arguments
    fCapsTryExecute(
        pProg,
        [&]() {
          // messages
          return json::array({
              {
                  {"role", "user"},
                  {"content", "User message"},
              },
              {{"role", "assistant"},
               {"content", ""}, // Some templates expect content to be empty with tool calls
               {"tool_calls",
                json::array({{{"id", "call00001"},
                              {"type", "function"},
                              {"function", {{"name", "tool1"}, {"arguments", R"({"arg": "value"})"}}}}})}},
              {{"role", "tool"}, {"content", "Tool response"}, {"tool_call_id", "call00001"}},
              {{"role", "assistant"}, {"content", "The tool response was 'tool response'"}},
              {
                  {"role", "user"},
                  {"content", "User message"},
              },
          });
        },
        nullptr, // ctx_fn
        [&]() {
          // tools
          return json::array({
              {
                  {"name", "tool"},
                  {"type", "function"},
                  {"function",
                   {
                       {"name", "tool1"},
                       {"description", "Tool description"},
                       {"parameters",
                        {
                            {"type", "object"},
                            {"properties",
                             {
                                 {"arg",
                                  {
                                      {"type", "string"},
                                      {"description", "Arg description"},
                                  }},
                             }},
                            {"required", json::array({"arg"})},
                        }},
                   }},
              },
          });
        },
        [&](bool pSuccess, value &pMessages, value &pTools, const std::string &) {
          if (!pSuccess) {
            vResult.vSupportsToolCalls = false;
            vResult.vSupportsTools = false;
            return;
          }

          auto &vToolName = pTools->mAt(0)->mAt("function")->mAt("name");
          fCapsPrintStats(vToolName, "tools[0].function.name");
          fCapsPrintStats(pTools, "tools");
          if (!vToolName->vStats.vUsed) {
            vResult.vSupportsTools = false;
          }

          auto &vToolCalls = pMessages->mAt(1)->mAt("tool_calls");
          fCapsPrintStats(vToolCalls, "messages[1].tool_calls");
          if (!vToolCalls->vStats.vUsed) {
            vResult.vSupportsToolCalls = false;
            return;
          }
        });
  }

  JJ_DEBUG("%s\n", ">>> Running capability check: parallel tool support");

  // case: tools support: parallel calls
  fCapsTryExecute(
      pProg,
      [&]() {
        json vArgs = json(R"({"arg": "value"})");
        if (vResult.vSupportsObjectArguments) {
          vArgs = json{{"arg", "value"}};
        }

        // messages
        return json::array({
            {
                {"role", "user"},
                {"content", "User message"},
            },
            {{"role", "assistant"},
             {"content", ""}, // Some templates expect content to be empty with tool calls
             {"tool_calls",
              json::array(
                  {{{"id", "call00001"}, {"type", "function"}, {"function", {{"name", "tool1"}, {"arguments", vArgs}}}},
                   {{"id", "call00002"},
                    {"type", "function"},
                    {"function", {{"name", "tool1"}, {"arguments", vArgs}}}}})}},
            {{"role", "tool"}, {"content", "Tool response"}, {"tool_call_id", "call00001"}},
            {{"role", "assistant"}, {"content", "The tool response was 'tool response'"}},
            {
                {"role", "user"},
                {"content", "User message"},
            },
        });
      },
      nullptr, // ctx_fn
      [&]() {
        // tools
        return json::array({
            {
                {"name", "tool"},
                {"type", "function"},
                {"function",
                 {
                     {"name", "tool1"},
                     {"description", "Tool description"},
                     {"parameters",
                      {
                          {"type", "object"},
                          {"properties",
                           {
                               {"arg",
                                {
                                    {"type", "string"},
                                    {"description", "Arg description"},
                                }},
                           }},
                          {"required", json::array({"arg"})},
                      }},
                 }},
            },
        });
      },
      [&](bool pSuccess, value &pMessages, value &, const std::string &) {
        if (!pSuccess) {
          vResult.vSupportsParallelToolCalls = false;
          return;
        }

        auto &vToolCalls = pMessages->mAt(1)->mAt("tool_calls");
        fCapsPrintStats(vToolCalls, "messages[1].tool_calls");

        // check for second tool call usage
        auto &vToolCall1 = vToolCalls->mAt(1)->mAt("function");
        fCapsPrintStats(vToolCall1, "messages[1].tool_calls[1].function");
        if (!vToolCall1->vStats.vUsed) {
          vResult.vSupportsParallelToolCalls = false;
        }
      });

  JJ_DEBUG("%s\n", ">>> Running capability check: preserve reasoning");

  // case: preserve reasoning content in chat history
  const std::string cReasoningPlaceholder = "<REASONING_CONTENT_PLACEHOLDER>";
  fCapsTryExecute(
      pProg,
      [&]() {
        // messages
        return json::array({
            {{"role", "user"}, {"content", "User message"}},
            {{"role", "assistant"},
             {"content", "Assistant message"},
             // check of reasoning_content deeper in the history, not just the last assistant message
             {"reasoning_content", cReasoningPlaceholder}},
            {{"role", "user"}, {"content", "User message"}},
            {{"role", "assistant"}, {"content", "Assistant message"}, {"reasoning_content", "Reasoning content"}},
            {{"role", "user"}, {"content", "User message"}},
        });
      },
      [&](context &pCtx) {
        pCtx.mSetVal("enable_thinking", fMkVal<value_bool>(true));
        fCapsApplyPreserveReasoning(pCtx, true);
      },
      nullptr, // tools_fn
      [&](bool, value &, value &, const std::string &pOutput) {
        // note: we cannot use stats here because the reasoning_content may be used for "if" condition test, but not
        // actually outputted in the final result
        if (pOutput.find(cReasoningPlaceholder) != std::string::npos) {
          vResult.vSupportsPreserveReasoning = true;
        }
      });

  JJ_DEBUG("%s\n", vResult.mToString().c_str());

  return vResult;
}

} // namespace jinja
