#include "ggml-backend-dl.h"

#ifdef _WIN32

dl_handle *dl_load_library(const fs::path &path) {
  // suppress error dialogs for missing DLLs
  DWORD old_mode = SetErrorMode(SEM_FAILCRITICALERRORS);
  SetErrorMode(old_mode | SEM_FAILCRITICALERRORS);

  HMODULE handle = LoadLibraryW(path.wstring().c_str());

  SetErrorMode(old_mode);

  return handle;
}

void *dl_get_sym(dl_handle *handle, const char *name) {
  DWORD old_mode = SetErrorMode(SEM_FAILCRITICALERRORS);
  SetErrorMode(old_mode | SEM_FAILCRITICALERRORS);

  void *p = (void *)GetProcAddress(handle, name);

  SetErrorMode(old_mode);

  return p;
}

const char *dl_error() { return ""; }

#else

dl_handle *dl_load_library(const fs::path &pPath) {
  dl_handle *vHandle = dlopen(pPath.string().c_str(), RTLD_NOW | RTLD_LOCAL);
  return vHandle;
}

void *dl_get_sym(dl_handle *pHandle, const char *pName) { return dlsym(pHandle, pName); }

const char *dl_error() {
  const char *vRslt = dlerror();
  return vRslt != nullptr ? vRslt : "";
}

#endif
