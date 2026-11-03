#pragma once

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <winevt.h>
#else
#include <dlfcn.h>
#include <unistd.h>
#endif
#include <filesystem>

namespace fs = std::filesystem;

#ifdef _WIN32

using dl_handle = std::remove_pointer_t<HMODULE>;

struct dl_handle_deleter {
  void operator()(HMODULE handle) { FreeLibrary(handle); }
};

#else

using dl_handle = void;

struct dl_handle_deleter {
  void operator()(void *pHandle) { dlclose(pHandle); }
};

#endif

using dl_handle_ptr = std::unique_ptr<dl_handle, dl_handle_deleter>;

dl_handle *dl_load_library(const fs::path &pPath);
void *dl_get_sym(dl_handle *pHandle, const char *pName);
const char *dl_error();
