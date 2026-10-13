#pragma once

#include "server-common.h"
#include "server-http.h"

#include <string>

//
// Settings of the Web UI, kept by the server as one JSON document in ~/.mgs/config/ui-settings.json of the user that
// runs mgs, so every browser that opens the interface shares them. GET <api prefix>/ui/settings returns
// {"settings": DOCUMENT}, or {"settings": null} before the first save, and POST {"settings": DOCUMENT} replaces the
// whole file. The API key that a browser uses to reach the server is not part of the document. /api/config.js
// publishes the saved interface language, so the interface starts in it before its modules load.
//

// Saved interface language, or "" when there is none.
std::string fUiSettingsLanguage();

server_http_context::handler_t fUiSettingsMakeGetHandler();
server_http_context::handler_t fUiSettingsMakePostHandler();
