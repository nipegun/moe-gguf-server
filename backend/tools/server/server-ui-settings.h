#pragma once

#include "server-common.h"
#include "server-http.h"

#include <string>

//
// Settings of the Web UI, kept by the server as one JSON document in ~/.mgs/config/ui-settings.json of the user that
// runs mgs, so every browser that opens the interface shares them. GET <api prefix>/ui/settings returns
// {"settings": DOCUMENT}, or {"settings": null} before the first save, and POST {"settings": DOCUMENT} replaces the
// whole file. The API key that a browser uses to reach the server is not part of the document. /api/config.js
// publishes the saved interface language and theme, so the interface starts with them before its modules load.
//
// The section modelsFolder holds the folder of local models of Settings > Models. Router mode (mgs without --model)
// scans it when mgs starts without --models-dir, and POST <api prefix>/models/folder writes it (the interface then
// keeps the same value in its copy of the document, so its next save does not revert it).
//

// Home folder of the user that runs mgs (HOME, or the account database), or "" when it is unknown.
std::string fConfigHomeDirectory();

// Saved interface language, or "" when there is none.
std::string fUiSettingsLanguage();

// Saved interface theme ("light", "dark" or "system"), or "" when there is none.
std::string fUiSettingsTheme();

// Path of the settings file (~/.mgs/config/ui-settings.json), or "" when the home folder is unknown.
std::string fUiSettingsFile();

// Saved models folder (section modelsFolder), or "" when there is none.
std::string fUiSettingsModelsFolder();

// Saves the models folder in the section modelsFolder ("" removes it) and keeps the rest of the document as it is.
// Returns false with the reason in pError.
bool fUiSettingsSaveModelsFolder(const std::string &pFolder, std::string &pError);

// Whether the web UI turned the local shell MCP server on (section localShellMcpServer).
bool fUiSettingsLocalShellMcpServer();

// Saves whether the web UI turned the local shell MCP server on (false removes the section) and keeps the rest of the
// document as it is. Returns false with the reason in pError.
bool fUiSettingsSaveLocalShellMcpServer(bool pEnabled, std::string &pError);

server_http_context::handler_t fUiSettingsMakeGetHandler();
server_http_context::handler_t fUiSettingsMakePostHandler();
