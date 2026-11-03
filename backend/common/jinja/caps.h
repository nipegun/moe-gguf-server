#pragma once

#include "runtime.h"

#include <string>
#include <map>

namespace jinja {

struct caps {
  bool vSupportsTools = true;
  bool vSupportsToolCalls = true;
  bool vSupportsSystemRole = true;
  bool vSupportsParallelToolCalls = true;

  // supports preserve reasoning trace in the full history, not just the last assistant message
  bool vSupportsPreserveReasoning = false;

  // one of the 2 content capabilities must be true
  bool vSupportsStringContent = true;
  bool vSupportsTypedContent = false;

  bool vSupportsObjectArguments = false;

  // for reporting on server
  std::map<std::string, bool> mToMap() const;

  // for debugging
  std::string mToString() const;
};

caps fCapsGet(jinja::program &pProg);

void fCapsApplyPreserveReasoning(jinja::context &pCtx, bool pEnabled);

} // namespace jinja
