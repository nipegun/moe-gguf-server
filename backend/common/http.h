#pragma once

#include <cpp-httplib/httplib.h>

#ifdef _WIN32
#include <winsock2.h>
#include <windows.h>
#else
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <unistd.h>
#endif

struct common_http_url {
  std::string vScheme;
  std::string vUser;
  std::string vPassword;
  std::string vHost;
  int vPort;
  std::string vPath;
};

// bracket an IPv6 literal host for a URL authority (RFC 3986)
static std::string fCommonHttpFormatHost(const std::string &pHost) {
  return pHost.find(':') != std::string::npos ? "[" + pHost + "]" : pHost;
}

static common_http_url fCommonHttpParseUrl(const std::string &pUrl) {
  common_http_url vParts;
  auto vSchemeEnd = pUrl.find("://");

  if (vSchemeEnd == std::string::npos) {
    throw std::runtime_error("invalid URL: no scheme");
  }
  vParts.vScheme = pUrl.substr(0, vSchemeEnd);

  if (vParts.vScheme != "http" && vParts.vScheme != "https") {
    throw std::runtime_error("unsupported URL scheme: " + vParts.vScheme);
  }

  auto vRest = pUrl.substr(vSchemeEnd + 3);
  auto vAtPos = vRest.find('@');

  if (vAtPos != std::string::npos) {
    auto vAuth = vRest.substr(0, vAtPos);
    auto vColonPos = vAuth.find(':');
    if (vColonPos != std::string::npos) {
      vParts.vUser = vAuth.substr(0, vColonPos);
      vParts.vPassword = vAuth.substr(vColonPos + 1);
    } else {
      vParts.vUser = vAuth;
    }
    vRest = vRest.substr(vAtPos + 1);
  }

  auto vSlashPos = vRest.find('/');

  if (vSlashPos != std::string::npos) {
    vParts.vHost = vRest.substr(0, vSlashPos);
    vParts.vPath = vRest.substr(vSlashPos);
  } else {
    vParts.vHost = vRest;
    vParts.vPath = "/";
  }

  // split the authority into host and optional port, a bracketed IPv6 literal keeps its inner colons (RFC 3986)
  std::string vPortStr;
  if (!vParts.vHost.empty() && vParts.vHost.front() == '[') {
    auto vClose = vParts.vHost.find(']');
    if (vClose == std::string::npos) {
      throw std::runtime_error("invalid IPv6 URL authority: " + vParts.vHost);
    }
    auto vAfter = vParts.vHost.substr(vClose + 1);
    if (!vAfter.empty() && vAfter.front() == ':') {
      vPortStr = vAfter.substr(1);
    }
    vParts.vHost = vParts.vHost.substr(1, vClose - 1);
  } else {
    auto vColonPos = vParts.vHost.find(':');
    if (vColonPos != std::string::npos) {
      vPortStr = vParts.vHost.substr(vColonPos + 1);
      vParts.vHost = vParts.vHost.substr(0, vColonPos);
    }
  }

  if (!vPortStr.empty()) {
    vParts.vPort = std::stoi(vPortStr);
  } else if (vParts.vScheme == "http") {
    vParts.vPort = 80;
  } else if (vParts.vScheme == "https") {
    vParts.vPort = 443;
  } else {
    throw std::runtime_error("unsupported URL scheme: " + vParts.vScheme);
  }

  return vParts;
}

static std::pair<httplib::Client, common_http_url> fCommonHttpClient(const std::string &pUrl) {
  common_http_url vParts = fCommonHttpParseUrl(pUrl);

  if (vParts.vHost.empty()) {
    throw std::runtime_error("error: invalid URL format");
  }

#ifndef CPPHTTPLIB_OPENSSL_SUPPORT
  if (vParts.vScheme == "https") {
    throw std::runtime_error("HTTPS is not supported. Please rebuild with one of:\n"
                             "  -DMGS_BUILD_BORINGSSL=ON\n"
                             "  -DMGS_BUILD_LIBRESSL=ON\n"
                             "  -DMGS_OPENSSL=ON (default, requires OpenSSL dev files installed)");
  }
#endif

  httplib::Client vCli(vParts.vScheme + "://" + fCommonHttpFormatHost(vParts.vHost) + ":" +
                       std::to_string(vParts.vPort));

  if (!vParts.vUser.empty()) {
    vCli.set_basic_auth(vParts.vUser, vParts.vPassword);
  }

  vCli.set_follow_location(true);

  return {std::move(vCli), std::move(vParts)};
}

static std::string fCommonHttpShowMaskedUrl(const common_http_url &pParts) {
  return pParts.vScheme + "://" + (pParts.vUser.empty() ? "" : "****:****@") + fCommonHttpFormatHost(pParts.vHost) +
         pParts.vPath;
}

static int fCommonHttpGetFreePort() {
#ifdef _WIN32
  WSADATA vWsaData;
  if (WSAStartup(MAKEWORD(2, 2), &vWsaData) != 0) {
    return -1;
  }
  typedef SOCKET native_socket_t;
#define INVALID_SOCKET_VAL INVALID_SOCKET
#define CLOSE_SOCKET(s) closesocket(s)
#else
  typedef int native_socket_t;
#define INVALID_SOCKET_VAL -1
#define CLOSE_SOCKET(s) close(s)
#endif

  native_socket_t vSock = socket(AF_INET, SOCK_STREAM, 0);
  if (vSock == INVALID_SOCKET_VAL) {
#ifdef _WIN32
    WSACleanup();
#endif
    return -1;
  }

  struct sockaddr_in vServAddr;
  std::memset(&vServAddr, 0, sizeof(vServAddr));
  vServAddr.sin_family = AF_INET;
  vServAddr.sin_addr.s_addr = htonl(INADDR_ANY);
  vServAddr.sin_port = htons(0);

  if (bind(vSock, (struct sockaddr *)&vServAddr, sizeof(vServAddr)) != 0) {
    CLOSE_SOCKET(vSock);
#ifdef _WIN32
    WSACleanup();
#endif
    return -1;
  }

#ifdef _WIN32
  int vNamelen = sizeof(vServAddr);
#else
  socklen_t vNamelen = sizeof(vServAddr);
#endif
  if (getsockname(vSock, (struct sockaddr *)&vServAddr, &vNamelen) != 0) {
    CLOSE_SOCKET(vSock);
#ifdef _WIN32
    WSACleanup();
#endif
    return -1;
  }

  int vPort = ntohs(vServAddr.sin_port);

  CLOSE_SOCKET(vSock);
#ifdef _WIN32
  WSACleanup();
#endif

  return vPort;
}
