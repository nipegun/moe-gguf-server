#pragma once

#include <memory>
#include <string>

namespace httplib {
class Server;
}

// Use supplied PEM files, or the self-signed identity of ~/.mgs/certificates (created there the first time).
std::unique_ptr<httplib::Server> fCreateHttpsServer(const std::string &pHost, const std::string &pCertificate,
                                                    const std::string &pKey);
