#pragma once

#include <nlohmann/json_fwd.hpp>

#include <functional>
#include <memory>
#include <string>

std::string fJsonSchemaToGrammar(const nlohmann::ordered_json &pSchema, bool pForceGbnf = false);

class common_schema_converter;

// Probes a JSON schema to extract information about its structure and type constraints.
class common_schema_info {
  std::unique_ptr<common_schema_converter> vImpl;

public:
  common_schema_info();
  ~common_schema_info();

  common_schema_info(const common_schema_info &) = delete;
  common_schema_info &operator=(const common_schema_info &) = delete;
  common_schema_info(common_schema_info &&) noexcept;
  common_schema_info &operator=(common_schema_info &&) noexcept;

  void mResolveRefs(nlohmann::ordered_json &pSchema);
  bool mResolvesToString(const nlohmann::ordered_json &pSchema);
};

struct common_grammar_builder {
  std::function<std::string(const std::string &, const std::string &)> vAddRule;
  std::function<std::string(const std::string &, const nlohmann::ordered_json &)> vAddSchema;
  std::function<void(nlohmann::ordered_json &)> vResolveRefs;
};

struct common_grammar_options {
  bool vDotall = false;
};

std::string fGbnfFormatLiteral(const std::string &pLiteral);

std::string fBuildGrammar(const std::function<void(const common_grammar_builder &)> &pCb,
                          const common_grammar_options &pOptions = {});
