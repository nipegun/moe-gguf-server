#pragma once

#include "chat-auto-parser.h"

#include <functional>
#include <optional>
#include <string>

std::string fTrimWhitespace(const std::string &pStr);
std::string fTrimLeadingWhitespace(const std::string &pStr);
std::string fTrimTrailingWhitespace(const std::string &pStr);
std::string fTrimTrailingNewlines(const std::string &pStr);

// calculate a diff split (longest common prefix, longest common suffix excluding prefix,
// mismatched part on the left, mismatched part on the right) between two strings
// account for markers - align prefix and suffix endings so that they end on markers
// * eg.:
// fCalculateDiffSplit("<html><body><div></div></body></html>", "<html><body><p>Something</p></body><html>") ->
//  { "prefix": "<html><body>" (not: "<html><body><"), "suffix": "</body></html>", "left": "<div></div>", "right":
//  "<p>Something</p>" }
// fCalculateDiffSplit("<html><body>Something</body></html>", "<html><body></body><html>") ->
//  { "prefix": "<html><body>", "suffix": "</body></html>", "left": "Something", "right": "" }
diff_split fCalculateDiffSplit(const std::string &pLeft, const std::string &pRight);

// Returns the prefix of `pFull` up until the first occurrence of the common prefix of `pLeft` and `pRight`
// Returns empty string if there's no common prefix
// * eg.:
// fUntilCommonPrefix("really want a FUNCTION call", "FUNCTION alpha", "FUNCTION beta") -> "really want a "
// fUntilCommonPrefix("<tool_call>", "<something>", "<something_else>") -> ""
// fUntilCommonPrefix("some text", "1234", "abcd") -> ""
// fUntilCommonPrefix("one arg two args three args four", "argument alpha", "argument beta") -> "one ""
std::string fUntilCommonPrefix(const std::string &pFull, const std::string &pLeft, const std::string &pRight);

// Returns the suffix of `pFull` after the last occurrence of the common suffix of `pLeft` and `pRight`
// Returns empty string if there's no common suffix
// Mirror function of `fUntilCommonPrefix`
// * eg.:
// fAfterCommonSuffix("really want a FUNCTION call", "first FUNCTION", "second FUNCTION") -> " call"
// fAfterCommonSuffix("one arg two-args three args four", "alpha-args", "beta-args") -> " three args four"
std::string fAfterCommonSuffix(const std::string &pFull, const std::string &pLeft, const std::string &pRight);

// Segmentize text into markers and non-marker fragments
// * eg.:
// fSegmentizeMarkers("<html><head><title>The site title</title><body><div>Here's some
// <b>content</b></div></body></html>" ->
//  [ (MARKER, "<html>"), (MARKER, "<head>"), (MARKER, "<title>"), (TEXT, "The site title"), (MARKER, "</title>"),
//    (MARKER, "<body>"), (MARKER, "<div>"), (TEXT, "Here's some "), (MARKER, "<b>"), (TEXT, "content"), (MARKER,
//    "</b>"), (MARKER, "</div>"), (MARKER, "</body>"), (MARKER, "</html>")
//  ]
// fSegmentizeMarkers("<|tool_call|>[args]{ are here }[/args]<|tool_call_end|>") ->
//  [ (MARKER, "<|tool_call|>"), (MARKER, "[args]"), (TEXT, "{ are here }"), (MARKER, "[/args]"), (MARKER,
//  "<|tool_call_end|>") ]
std::vector<segment> fSegmentizeMarkers(const std::string &pText);

// Prune whitespace-only segments from a vector of segments
// * eg.:
// fSegmentizeMarkers("<tool_call>\n<function=foo>\n<arg=bar>\n   \n</arg>\n</function>\n</tool_call>") ->
//  X = [ (MARKER, "<tool_call>"), (TEXT, "\n"), (MARKER, "<function=foo>"), (TEXT, "\n"), (MARKER, "<arg=bar>"), (TEXT,
//  "\n   \n"),
//        (MARKER, "</arg>"), (TEXT, "\n"), (MARKER, "</function>"), (TEXT, "\n"), (MARKER, "</tool_call>") ]
// fPruneWhitespaceSegments(X) -> [ (MARKER, "<tool_call>"), (MARKER, "<function=foo>"), (MARKER, "<arg=bar>"),
// (MARKER, "</arg>"),
//                                   (MARKER, "</function>"), (MARKER, "</tool_call>") ]
std::vector<segment> fPruneWhitespaceSegments(const std::vector<segment> &pSegments);

namespace autoparser {

// Apply a template with the given parameters, returning the rendered string (empty on failure)
std::string fApplyTemplate(const common_chat_template &pTmpl, const template_params &pParams);

// Factorized differential comparison function
// Takes base params and a single modifier lambda to create variant B
// Returns compare_variants_result containing diff and both outputs, or std::nullopt on failure
std::optional<compare_variants_result> fCompareVariants(const common_chat_template &pTmpl,
                                                        const template_params &pParamsA,
                                                        const std::function<void(template_params &)> &pParamsModifier);

} // namespace autoparser
