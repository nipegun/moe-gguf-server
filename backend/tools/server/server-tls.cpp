#include "server-tls.h"
#include "server-common.h"

#include <cpp-httplib/httplib.h>

#ifdef CPPHTTPLIB_OPENSSL_SUPPORT
#include <openssl/ec.h>
#include <openssl/pem.h>
#include <openssl/rand.h>
#include <openssl/x509v3.h>

#include <cerrno>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <functional>
#include <set>
#include <sstream>

#if !defined(_WIN32)
#include <arpa/inet.h>
#include <fcntl.h>
#include <ifaddrs.h>
#include <net/if.h>
#include <netinet/in.h>
#include <pwd.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

namespace fs = std::filesystem;

// The self-signed identity is created once and kept here, under the home of the user that runs mgs, so every run
// (and every new build) serves the same certificate and the browser keeps its trust exception.
static const char *cCertificateFolder = ".mgs/certificates";
static const char *cCertificateFile = "fullchain.pem";
static const char *cKeyFile = "privkey.pem";
static constexpr long cCertificateDays = 365;

using server_tls_key = std::unique_ptr<EVP_PKEY, decltype(&EVP_PKEY_free)>;
using server_tls_certificate = std::unique_ptr<X509, decltype(&X509_free)>;

static bool fAddCertificateExtension(X509 *pCertificate, int pId, const std::string &pValue) {
  std::unique_ptr<X509_EXTENSION, decltype(&X509_EXTENSION_free)> vExtension(
      X509V3_EXT_conf_nid(nullptr, nullptr, pId, pValue.c_str()), X509_EXTENSION_free);
  return vExtension && X509_add_ext(pCertificate, vExtension.get(), -1) == 1;
}

static std::string fHomeDirectory() {
  const char *vHome = std::getenv("HOME");
  if (vHome != nullptr && vHome[0] != '\0') {
    return vHome;
  }
#if defined(_WIN32)
  const char *vProfile = std::getenv("USERPROFILE");
  return vProfile != nullptr ? vProfile : "";
#else
  const passwd *vAccount = getpwuid(geteuid());
  return vAccount != nullptr && vAccount->pw_dir != nullptr ? vAccount->pw_dir : "";
#endif
}

// Names the certificate is valid for: localhost and the loopback addresses, the host it listens on when that is a
// single address or name, and, because the same certificate is reused whatever --host says, the machine name and
// every address of its network interfaces at the time it is created.
static std::string fSubjectAltNames(const std::string &pHost) {
  std::set<std::string> sNames = {"DNS:localhost", "IP:127.0.0.1", "IP:::1"};
  const auto fAddName = [&sNames](const std::string &pName) {
    if (pName.empty() || pName == "0.0.0.0" || pName == "::" ||
        pName.find_first_not_of("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789.-:") !=
            std::string::npos) {
      return;
    }
    std::unique_ptr<ASN1_OCTET_STRING, decltype(&ASN1_OCTET_STRING_free)> vIp(a2i_IPADDRESS(pName.c_str()),
                                                                              ASN1_OCTET_STRING_free);
    if (vIp) {
      sNames.insert("IP:" + pName);
    } else if (pName.find(':') == std::string::npos) {
      sNames.insert("DNS:" + pName);
    }
  };
  fAddName(pHost);
#if !defined(_WIN32)
  char aHostName[256] = {};
  if (gethostname(aHostName, sizeof(aHostName) - 1) == 0) {
    fAddName(aHostName);
  }
  ifaddrs *vAddresses = nullptr;
  if (getifaddrs(&vAddresses) == 0) {
    for (const ifaddrs *vEntry = vAddresses; vEntry != nullptr; vEntry = vEntry->ifa_next) {
      if (vEntry->ifa_addr == nullptr || (vEntry->ifa_flags & IFF_UP) == 0) {
        continue;
      }
      char aAddress[INET6_ADDRSTRLEN] = {};
      if (vEntry->ifa_addr->sa_family == AF_INET) {
        inet_ntop(AF_INET, &reinterpret_cast<const sockaddr_in *>(vEntry->ifa_addr)->sin_addr, aAddress,
                  sizeof(aAddress));
      } else if (vEntry->ifa_addr->sa_family == AF_INET6) {
        const in6_addr &vAddress = reinterpret_cast<const sockaddr_in6 *>(vEntry->ifa_addr)->sin6_addr;
        if (IN6_IS_ADDR_LINKLOCAL(&vAddress)) {
          continue; // only valid with a zone index, which certificates cannot hold
        }
        inet_ntop(AF_INET6, &vAddress, aAddress, sizeof(aAddress));
      }
      fAddName(aAddress);
    }
    freeifaddrs(vAddresses);
  }
#endif
  std::string vNames;
  for (const std::string &vName : sNames) {
    vNames += (vNames.empty() ? "" : ",") + vName;
  }
  return vNames;
}

static bool fGenerateServerIdentity(const std::string &pHost, server_tls_key &pKey,
                                    server_tls_certificate &pCertificate) {
  std::unique_ptr<EVP_PKEY_CTX, decltype(&EVP_PKEY_CTX_free)> vGenerator(EVP_PKEY_CTX_new_id(EVP_PKEY_EC, nullptr),
                                                                         EVP_PKEY_CTX_free);
  if (!vGenerator || EVP_PKEY_keygen_init(vGenerator.get()) <= 0 ||
      EVP_PKEY_CTX_set_ec_paramgen_curve_nid(vGenerator.get(), NID_X9_62_prime256v1) <= 0) {
    return false;
  }
  EVP_PKEY *vRawKey = nullptr;
  const int vKeyStatus = EVP_PKEY_keygen(vGenerator.get(), &vRawKey);
  server_tls_key vKey(vRawKey, EVP_PKEY_free);
  server_tls_certificate vCertificate(X509_new(), X509_free);
  if (vKeyStatus <= 0 || !vKey || !vCertificate) {
    return false;
  }

  unsigned char aSerial[16];
  if (RAND_bytes(aSerial, sizeof(aSerial)) != 1) {
    return false;
  }
  std::unique_ptr<BIGNUM, decltype(&BN_free)> vSerial(BN_bin2bn(aSerial, sizeof(aSerial), nullptr), BN_free);
  if (!vSerial || !BN_to_ASN1_INTEGER(vSerial.get(), X509_get_serialNumber(vCertificate.get())) ||
      X509_set_version(vCertificate.get(), 2) != 1 || !X509_gmtime_adj(X509_getm_notBefore(vCertificate.get()), -300) ||
      !X509_gmtime_adj(X509_getm_notAfter(vCertificate.get()), cCertificateDays * 24 * 60 * 60) ||
      X509_set_pubkey(vCertificate.get(), vKey.get()) != 1) {
    return false;
  }
  X509_NAME *vName = X509_get_subject_name(vCertificate.get());
  if (!vName ||
      X509_NAME_add_entry_by_txt(vName, "CN", MBSTRING_ASC, reinterpret_cast<const unsigned char *>("moe-gguf-server"),
                                 -1, -1, 0) != 1 ||
      X509_set_issuer_name(vCertificate.get(), vName) != 1) {
    return false;
  }
  if (!fAddCertificateExtension(vCertificate.get(), NID_basic_constraints, "critical,CA:FALSE") ||
      !fAddCertificateExtension(vCertificate.get(), NID_key_usage, "critical,digitalSignature") ||
      !fAddCertificateExtension(vCertificate.get(), NID_ext_key_usage, "serverAuth") ||
      !fAddCertificateExtension(vCertificate.get(), NID_subject_alt_name, fSubjectAltNames(pHost)) ||
      X509_sign(vCertificate.get(), vKey.get(), EVP_sha256()) <= 0) {
    return false;
  }
  pKey = std::move(vKey);
  pCertificate = std::move(vCertificate);
  return true;
}

// The saved identity, when both files exist, match and the certificate has not expired.
static bool fLoadServerIdentity(const fs::path &pFolder, server_tls_key &pKey, server_tls_certificate &pCertificate) {
  std::unique_ptr<BIO, decltype(&BIO_free)> vCertificateFile(
      BIO_new_file((pFolder / cCertificateFile).string().c_str(), "r"), BIO_free);
  std::unique_ptr<BIO, decltype(&BIO_free)> vKeyFile(BIO_new_file((pFolder / cKeyFile).string().c_str(), "r"),
                                                     BIO_free);
  if (!vCertificateFile || !vKeyFile) {
    ERR_clear_error();
    return false;
  }
  server_tls_certificate vCertificate(PEM_read_bio_X509(vCertificateFile.get(), nullptr, nullptr, nullptr), X509_free);
  server_tls_key vKey(PEM_read_bio_PrivateKey(vKeyFile.get(), nullptr, nullptr, nullptr), EVP_PKEY_free);
  if (!vCertificate || !vKey || X509_check_private_key(vCertificate.get(), vKey.get()) != 1 ||
      X509_cmp_current_time(X509_get0_notAfter(vCertificate.get())) <= 0) {
    ERR_clear_error();
    return false;
  }
  pKey = std::move(vKey);
  pCertificate = std::move(vCertificate);
  return true;
}

// Writes pText to pPath through a temporary file and a rename, so a reader never sees half a file. The private key
// is only readable by its owner.
static bool fWriteFileAtomically(const fs::path &pPath, const std::string &pText, bool pPrivate) {
  unsigned char aRandom[8];
  if (RAND_bytes(aRandom, sizeof(aRandom)) != 1) {
    return false;
  }
  std::string vSuffix;
  for (const unsigned char vByte : aRandom) {
    static const char *cHex = "0123456789abcdef";
    vSuffix += cHex[vByte >> 4];
    vSuffix += cHex[vByte & 0x0F];
  }
  const fs::path vTemporary = pPath.string() + ".tmp-" + vSuffix;
#if defined(_WIN32)
  (void)pPrivate;
  {
    std::ofstream vFile(vTemporary, std::ios::binary | std::ios::trunc);
    if (!(vFile << pText)) {
      return false;
    }
  }
#else
  const int vFd = open(vTemporary.c_str(), O_WRONLY | O_CREAT | O_EXCL | O_CLOEXEC, pPrivate ? 0600 : 0644);
  if (vFd < 0) {
    return false;
  }
  size_t vWritten = 0;
  while (vWritten < pText.size()) {
    const ssize_t vCount = write(vFd, pText.data() + vWritten, pText.size() - vWritten);
    if (vCount < 0 && errno == EINTR) {
      continue;
    }
    if (vCount <= 0) {
      close(vFd);
      unlink(vTemporary.c_str());
      return false;
    }
    vWritten += (size_t)vCount;
  }
  if (close(vFd) != 0) {
    unlink(vTemporary.c_str());
    return false;
  }
#endif
  std::error_code vError;
  fs::rename(vTemporary, pPath, vError);
  if (vError) {
    fs::remove(vTemporary, vError);
    return false;
  }
  return true;
}

static std::string fPemText(const std::function<int(BIO *)> &pWrite) {
  std::unique_ptr<BIO, decltype(&BIO_free)> vMemory(BIO_new(BIO_s_mem()), BIO_free);
  if (!vMemory || pWrite(vMemory.get()) != 1) {
    return std::string();
  }
  char *vData = nullptr;
  const long vSize = BIO_get_mem_data(vMemory.get(), &vData);
  return vSize > 0 && vData != nullptr ? std::string(vData, (size_t)vSize) : std::string();
}

static bool fSaveServerIdentity(const fs::path &pFolder, EVP_PKEY *pKey, X509 *pCertificate) {
  std::error_code vError;
  if (!fs::is_directory(pFolder, vError)) {
    fs::create_directories(pFolder, vError);
    if (vError) {
      return false;
    }
    // the folder of a private key: only its owner may list it
    fs::permissions(pFolder, fs::perms::owner_all, fs::perm_options::replace, vError);
  }
  const std::string vKeyText = fPemText(
      [pKey](BIO *pBio) { return PEM_write_bio_PrivateKey(pBio, pKey, nullptr, nullptr, 0, nullptr, nullptr); });
  const std::string vCertificateText =
      fPemText([pCertificate](BIO *pBio) { return PEM_write_bio_X509(pBio, pCertificate); });
  // the key first: a certificate file is only ever next to its own key
  return !vKeyText.empty() && !vCertificateText.empty() && fWriteFileAtomically(pFolder / cKeyFile, vKeyText, true) &&
         fWriteFileAtomically(pFolder / cCertificateFile, vCertificateText, false);
}

static bool fUseServerIdentity(SSL_CTX *pContext, EVP_PKEY *pKey, X509 *pCertificate) {
  return SSL_CTX_set_min_proto_version(pContext, TLS1_2_VERSION) == 1 &&
         SSL_CTX_use_certificate(pContext, pCertificate) == 1 && SSL_CTX_use_PrivateKey(pContext, pKey) == 1 &&
         SSL_CTX_check_private_key(pContext) == 1;
}

// The self-signed identity of ~/.mgs/certificates: reused while it is valid, created and saved otherwise. When it
// cannot be saved (no home, read-only folder), this run uses it from memory and the next one creates another.
static bool fPersistentServerIdentity(SSL_CTX *pContext, const std::string &pHost) {
  const std::string vHome = fHomeDirectory();
  const fs::path vFolder = vHome.empty() ? fs::path() : fs::path(vHome) / cCertificateFolder;
  server_tls_key vKey(nullptr, EVP_PKEY_free);
  server_tls_certificate vCertificate(nullptr, X509_free);
  if (!vFolder.empty() && fLoadServerIdentity(vFolder, vKey, vCertificate)) {
    SRV_INF("using the self-signed HTTPS certificate %s\n", (vFolder / cCertificateFile).string().c_str());
    return fUseServerIdentity(pContext, vKey.get(), vCertificate.get());
  }
  if (!fGenerateServerIdentity(pHost, vKey, vCertificate)) {
    return false;
  }
  if (!vFolder.empty() && fSaveServerIdentity(vFolder, vKey.get(), vCertificate.get())) {
    SRV_WRN("created the self-signed HTTPS certificate %s; it is reused by every run until it expires in %ld days. "
            "Browsers require a trust exception once, or import the certificate as trusted.\n",
            (vFolder / cCertificateFile).string().c_str(), cCertificateDays);
  } else {
    SRV_WRN("cannot save the self-signed HTTPS certificate in %s: %s; this run uses one generated in memory and the "
            "next one will generate another. Use --ssl-cert-file and --ssl-key-file for a persistent certificate.\n",
            vFolder.empty() ? "the home folder" : vFolder.string().c_str(), std::strerror(errno));
  }
  return fUseServerIdentity(pContext, vKey.get(), vCertificate.get());
}
#endif

std::unique_ptr<httplib::Server> fCreateHttpsServer(const std::string &pHost, const std::string &pCertificate,
                                                    const std::string &pKey) {
#ifdef CPPHTTPLIB_OPENSSL_SUPPORT
  if (!pCertificate.empty()) {
    return std::make_unique<httplib::SSLServer>(pCertificate.c_str(), pKey.c_str());
  }
  return std::make_unique<httplib::SSLServer>([&pHost](httplib::tls::ctx_t pContext) {
    return fPersistentServerIdentity(static_cast<SSL_CTX *>(pContext), pHost);
  });
#else
  (void)pHost;
  (void)pCertificate;
  (void)pKey;
  SRV_ERR("%s", "HTTPS requires an OpenSSL-enabled build; rebuild with MGS_OPENSSL=ON or explicitly use --http.\n");
  return nullptr;
#endif
}
