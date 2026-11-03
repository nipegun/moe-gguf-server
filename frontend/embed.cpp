// mgs-ui-embed: generate ui.cpp / ui.h that embed UI assets as C arrays.
//
// Usage:
//   mgs-ui-embed <out_cpp> <out_h> [<asset_dir>]
//
// Recursively embeds every regular file under <asset_dir>.
// Asset names are relative paths from <asset_dir> (e.g. "_app/immutable/bundle.HASH.js").
// Without <asset_dir>, emits an empty asset table.

#include <inttypes.h>
#include <stdarg.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <functional>
#include <string>
#include <vector>

static const char *fMimeFromExt(const std::string &pName) {
  auto vExt = pName.rfind('.');
  if (vExt == std::string::npos)
    return "application/octet-stream";
  std::string vE = pName.substr(vExt + 1);
  if (vE == "html")
    return "text/html; charset=utf-8";
  if (vE == "css")
    return "text/css";
  if (vE == "js")
    return "application/javascript";
  if (vE == "json")
    return "application/json";
  if (vE == "webmanifest")
    return "application/manifest+json";
  if (vE == "svg")
    return "image/svg+xml";
  if (vE == "png")
    return "image/png";
  if (vE == "jpg" || vE == "jpeg")
    return "image/jpeg";
  if (vE == "ico")
    return "image/x-icon";
  if (vE == "woff")
    return "font/woff";
  if (vE == "woff2")
    return "font/woff2";
  return "application/octet-stream";
}

// Computes FNV-1a hash of the data
static uint64_t fFnvHash(const uint8_t *pData, size_t pLen) {
  const uint64_t cFnvPrime = 0x100000001b3ULL;
  uint64_t vHash = 0xcbf29ce484222325ULL;

  for (size_t vI = 0; vI < pLen; ++vI) {
    vHash ^= pData[vI];
    vHash *= cFnvPrime;
  }
  return vHash;
}

static bool fReadFile(const std::filesystem::path &pPath, std::vector<unsigned char> &pOut) {
  std::ifstream vF(pPath, std::ios::binary | std::ios::ate);
  if (!vF) {
    fprintf(stderr, "embed: cannot open %s\n", pPath.string().c_str());
    return false;
  }
  const auto cSz = vF.tellg();
  if (cSz < 0) {
    return false;
  }
  vF.seekg(0);
  pOut.resize(static_cast<size_t>(cSz));
  if (cSz > 0 && !vF.read(reinterpret_cast<char *>(pOut.data()), cSz)) {
    return false;
  }
  return true;
}

static void fAppendBytesHex(std::string &pOut, const std::vector<unsigned char> &pBytes) {
  static const char cHex[] = "0123456789abcdef";
  pOut.reserve(pOut.size() + pBytes.size() * 5);
  for (unsigned char vB : pBytes) {
    pOut += '0';
    pOut += 'x';
    pOut += cHex[vB >> 4];
    pOut += cHex[vB & 0xf];
    pOut += ',';
  }
}

static bool fWriteIfDifferent(const std::string &pPath, const std::string &pContent) {
  std::ifstream vF(pPath, std::ios::binary | std::ios::ate);
  if (vF) {
    const auto cSz = vF.tellg();
    if (cSz >= 0 && static_cast<size_t>(cSz) == pContent.size()) {
      std::string vExisting(static_cast<size_t>(cSz), '\0');
      vF.seekg(0);
      if (cSz == 0 || vF.read(vExisting.data(), cSz)) {
        if (vExisting == pContent) {
          return true;
        }
      }
    }
  }

  std::ofstream vOut(pPath, std::ios::binary | std::ios::trunc);
  if (!vOut) {
    fprintf(stderr, "embed: cannot write %s\n", pPath.c_str());
    return false;
  }
  if (!pContent.empty()) {
    vOut.write(pContent.data(), static_cast<std::streamsize>(pContent.size()));
  }
  bool vOk = vOut.good();
  if (vOk) {
    printf("embed: write output file %s\n", pPath.c_str());
  }
  return vOk;
}

static std::string fPathBasename(const std::string &pName) {
  const size_t cP = pName.rfind('/');
  return cP == std::string::npos ? pName : pName.substr(cP + 1);
}
static bool fStrStartsWith(const std::string &pS, const char *pPrefix) {
  const size_t cN = strlen(pPrefix);
  return pS.size() >= cN && pS.compare(0, cN, pPrefix) == 0;
}
static bool fStrEndsWith(const std::string &pS, const char *pSuffix) {
  const size_t cN = strlen(pSuffix);
  return pS.size() >= cN && pS.compare(pS.size() - cN, cN, pSuffix) == 0;
}

static std::string fFmt(const char *pPattern, ...) {
  char vTmp[512];
  va_list vAp;
  va_start(vAp, pPattern);
  const int cN = vsnprintf(vTmp, sizeof(vTmp), pPattern, vAp);
  va_end(vAp);
  return (cN > 0) ? std::string(vTmp, static_cast<size_t>(cN)) : std::string();
}

struct asset_entry {
  std::string vName;
  std::filesystem::path vPath;
};

int main(int argc, char **argv) {
  if (argc < 3 || argc > 4) {
    fprintf(stderr, "usage: %s <out_cpp> <out_h> [<asset_dir>]\n", argv[0]);
    return 1;
  }

  const std::string cOutCpp = argv[1];
  const std::string cOutH = argv[2];
  const std::string cAssetDir = (argc >= 4) ? argv[3] : std::string();

  const bool cUseGzip = !cAssetDir.empty() && std::filesystem::exists(cAssetDir + "/_gzip");
  const std::string cInDir = cUseGzip ? (cAssetDir + "/_gzip") : cAssetDir;

  std::vector<asset_entry> vAssets;
  if (!cInDir.empty()) {
    const std::filesystem::path cDir = cInDir;

    std::error_code vEc;
    std::filesystem::recursive_directory_iterator vIt(cDir, vEc);
    if (vEc) {
      fprintf(stderr, "embed: cannot iterate %s: %s\n", argv[3], vEc.message().c_str());
      return 1;
    }
    for (const auto &vEntry : vIt) {
      if (!vEntry.is_regular_file()) {
        continue;
      }
      // name is the relative path from dir, with forward slashes
      const std::string cName = vEntry.path().lexically_relative(cDir).generic_string();
      vAssets.push_back({cName, vEntry.path()});
    }

    // directory iteration order is unspecified; sort for reproducible output
    std::sort(vAssets.begin(), vAssets.end(),
              [](const asset_entry &pA, const asset_entry &pB) { return pA.vName < pB.vName; });
  }

  const int cNAssets = static_cast<int>(vAssets.size());

  if (cNAssets > 0) {
    using match_fn = std::function<bool(const std::string &)>;
    auto vExact = [](const char *pName) -> match_fn {
      return [pName](const std::string &pBase) { return pBase == pName; };
    };

    struct required_check {
      const char *vLabel;
      match_fn vMatch;
      bool vFound;
    };
    required_check vChecks[] = {
        {"index.html", vExact("index.html"), false},
        {"manifest.webmanifest", vExact("manifest.webmanifest"), false},
        {"sw.js", vExact("sw.js"), false},
        {"build.json", vExact("build.json"), false},
        {"version.json", vExact("version.json"), false},
        {"bundle[hash].js",
         [](const std::string &pB) { return fStrStartsWith(pB, "bundle") && fStrEndsWith(pB, ".js"); }, false},
        {"bundle[hash].css",
         [](const std::string &pB) { return fStrStartsWith(pB, "bundle") && fStrEndsWith(pB, ".css"); }, false},
        {"workbox[hash].js",
         [](const std::string &pB) { return fStrStartsWith(pB, "workbox") && fStrEndsWith(pB, ".js"); }, false},
    };

    for (const auto &vA : vAssets) {
      const std::string cBase = fPathBasename(vA.vName);
      for (auto &vC : vChecks) {
        if (!vC.vFound) {
          vC.vFound = vC.vMatch(cBase);
        }
      }
    }

    std::vector<const char *> vMissing;
    for (const auto &vC : vChecks) {
      if (!vC.vFound) {
        vMissing.push_back(vC.vLabel);
      }
    }
    if (!vMissing.empty()) {
      fprintf(stderr, "\ncurrent asset files:\n");
      for (const auto &vA : vAssets) {
        fprintf(stderr, "    %s\n", vA.vName.c_str());
      }
      fprintf(stderr, "missing required asset(s):\n");
      for (const char *vM : vMissing) {
        fprintf(stderr, "    %s\n", vM);
      }
      fprintf(stderr, "hint: try cleaning your build directory: %s\n", cInDir.c_str());
      return 1;
    }
  }

  std::string vH;
  vH += "#pragma once\n\n#include <array>\n#include <string>\n\n";
  if (cNAssets > 0) {
    vH += "#define MGS_UI_HAS_ASSETS 1\n\n";
  }
  vH += "struct mgs_ui_asset {\n"
        "    std::string           name;\n"
        "    const unsigned char * data;\n"
        "    std::size_t           size;\n"
        "    std::string           etag;\n"
        "    std::string           type;\n"
        "};\n\n"
        "const mgs_ui_asset * mgs_ui_find_asset(const std::string & name);\n"
        "bool mgs_ui_use_gzip();\n";
  vH += fFmt("const std::array<mgs_ui_asset, %d> & mgs_ui_get_assets();\n", cNAssets);

  std::string vCpp;
  vCpp += "#include \"ui.h\"\n\n";

  if (cNAssets > 0) {
    for (int vI = 0; vI < cNAssets; vI++) {
      std::vector<unsigned char> vBytes;
      if (!fReadFile(vAssets[vI].vPath, vBytes)) {
        return 1;
      }
      if (vBytes.empty()) {
        fprintf(stderr, "embed: empty file: %s\n", vAssets[vI].vPath.generic_string().c_str());
        return 1;
      }
      vCpp += fFmt("static const unsigned char asset_%d_data[] = {", vI);
      fAppendBytesHex(vCpp, vBytes);
      const auto cHash = fFnvHash(vBytes.data(), vBytes.size());

      vCpp += fFmt("};\nstatic const std::size_t   asset_%d_size = %zu;\n", vI, vBytes.size());
      vCpp += fFmt("static const char          asset_%d_etag[] = \"\\\"0x%016" PRIx64 "\\\"\";\n\n", vI, cHash);
    }

    vCpp += fFmt("static const std::array<mgs_ui_asset, %d> g_assets = {{\n", cNAssets);
    for (int vI = 0; vI < cNAssets; vI++) {
      const std::string &vName = vAssets[vI].vName;
      vCpp += fFmt("    { \"%s\", asset_%d_data, asset_%d_size, asset_%d_etag, \"%s\" },\n", vName.c_str(), vI, vI, vI,
                   fMimeFromExt(vName));
    }
    vCpp += "}};\n\n";

    vCpp += "const mgs_ui_asset * mgs_ui_find_asset(const std::string & name) {\n"
            "    for (const auto & a : g_assets) {\n"
            "        if (a.name == name) {\n"
            "            return &a;\n"
            "        }\n"
            "    }\n"
            "    return nullptr;\n"
            "}\n";
    vCpp += fFmt("const std::array<mgs_ui_asset, %d> & mgs_ui_get_assets() {\n", cNAssets);
    vCpp += "    return g_assets;\n"
            "}\n";
  } else {
    vCpp += "const mgs_ui_asset * mgs_ui_find_asset(const std::string &) {\n"
            "    return nullptr;\n"
            "}\n"
            "const std::array<mgs_ui_asset, 0> & mgs_ui_get_assets() {\n"
            "    static const std::array<mgs_ui_asset, 0> empty{};\n"
            "    return empty;\n"
            "}\n";
  }
  vCpp += fFmt("bool mgs_ui_use_gzip() { return %s; }\n", cUseGzip ? "true" : "false");

  bool vOk = true;
  vOk = fWriteIfDifferent(cOutH, vH) && vOk;
  vOk = fWriteIfDifferent(cOutCpp, vCpp) && vOk;
  return vOk ? 0 : 1;
}
