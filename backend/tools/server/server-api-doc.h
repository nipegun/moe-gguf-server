#pragma once

namespace httplib {
class Server;
}
struct server_http_context;

void fRegisterApiDocumentation(httplib::Server &pServer, server_http_context &pContext);
