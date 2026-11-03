#ifndef SPINE_TCM_PUBLIC_H_
#define SPINE_TCM_PUBLIC_H_

/*
 * spine_tcm public API
 *
 * Usage:
 *   1. Direct link mode
 *      Define SPINE_TCM_DIRECT_LINK and link against libspine_tcm.so.
 *
 *      if (spine_tcm_is_available()) {
 *          void *buffer = spine_tcm_mem_get(0);
 *          spine_tcm_mem_free(0);
 *      }
 *
 *   2. Header-only loader mode
 *      Include this header without linking libspine_tcm.so. The loader first
 *      tries to reuse a process-global spine_tcm instance and falls back to
 *      dlopen("libspine_tcm.so") when needed.
 *
 *      spine_tcm_open_handle(NULL);  // optional pre-bind
 *      if (spine_tcm_is_available()) {
 *          void *buffer = spine_tcm_mem_get(0);
 *          spine_tcm_mem_free(0);
 *      }
 */

#include <stddef.h>
#include <stdint.h>
#include <string.h>

#if !defined(SPINE_TCM_BUILD_SHARED) && !defined(SPINE_TCM_DIRECT_LINK)
#include <dlfcn.h>
#endif

#ifdef __cplusplus
extern "C" {
#endif

#if defined(_WIN32)
#if defined(SPINE_TCM_BUILD_SHARED)
#define SPINE_TCM_API __declspec(dllexport)
#else
#define SPINE_TCM_API __declspec(dllimport)
#endif
#else
#define SPINE_TCM_API __attribute__((visibility("default")))
#endif

typedef struct spine_tcm_mem_info {
  size_t blk_size;
  size_t blk_num;
  int is_fake_tcm;
} spine_tcm_mem_info_t;

typedef struct spine_tcm_block_info {
  int id;
  void *va;
  size_t size;
  uint64_t phys_addr;
  uint64_t cpu_affinity_mask;
  int owner_tid;
  int is_acquired;
} spine_tcm_block_info_t;

/* Shared-library runtime ABI exported by libspine_tcm.so. */
SPINE_TCM_API const char *spine_tcm_runtime_version(void);
SPINE_TCM_API int spine_tcm_runtime_is_available(void);
SPINE_TCM_API int spine_tcm_runtime_layout_info(spine_tcm_mem_info_t *pInfo);
SPINE_TCM_API int spine_tcm_runtime_mem_info(int pId, spine_tcm_block_info_t *pInfo);
SPINE_TCM_API void *spine_tcm_runtime_mem_get(int pId);
SPINE_TCM_API int spine_tcm_runtime_mem_free(int pId);
SPINE_TCM_API void *spine_tcm_runtime_mem_try_wait(int pId, size_t pTimeoutUs);
SPINE_TCM_API int spine_tcm_runtime_mem_release(int pId);
SPINE_TCM_API int spine_tcm_runtime_mem_force_release(int pId);
SPINE_TCM_API int spine_tcm_runtime_mem_query(int pId);

#if defined(SPINE_TCM_DIRECT_LINK)
/* Optional no-op in direct-link mode. */
static inline int spine_tcm_open_handle(const char *so_path) {
  (void)so_path;
  return 0;
}

static inline const char *spine_tcm_version(void) { return spine_tcm_runtime_version(); }

/* Returns 1 when the runtime driver is available, otherwise 0. */
static inline int spine_tcm_is_available(void) { return spine_tcm_runtime_is_available(); }

/* Returns runtime memory geometry and whether the current backend is fake TCM. */
static inline int spine_tcm_mem_info(spine_tcm_mem_info_t *info) { return spine_tcm_runtime_layout_info(info); }

/* Returns per-block runtime metadata for the given TCM id. */
static inline int spine_tcm_block_info(int id, spine_tcm_block_info_t *info) {
  return spine_tcm_runtime_mem_info(id, info);
}

/* Returns a cached buffer for the given TCM id, or NULL on failure. */
static inline void *spine_tcm_mem_get(int id) { return spine_tcm_runtime_mem_get(id); }

/* Releases one reference acquired by spine_tcm_mem_get(id). */
static inline int spine_tcm_mem_free(int id) { return spine_tcm_runtime_mem_free(id); }

/* Waits for a TCM block handoff and returns the driver-owned buffer when available. */
static inline void *spine_tcm_mem_try_wait(int id, size_t pOverTime) {
  return spine_tcm_runtime_mem_try_wait(id, pOverTime);
}

/* Releases a buffer acquired by spine_tcm_mem_try_wait(id, over_time). */
static inline int spine_tcm_mem_release(int id) { return spine_tcm_runtime_mem_release(id); }

/* Forces a release for the given TCM id when the backend supports it. */
static inline int spine_tcm_mem_force_release(int id) { return spine_tcm_runtime_mem_force_release(id); }

/* Returns whether the given TCM id is currently acquired. */
static inline int spine_tcm_mem_query(int id) { return spine_tcm_runtime_mem_query(id); }
#elif !defined(SPINE_TCM_BUILD_SHARED)
typedef struct spine_tcm_handle {
  void *module_handle;
  int use_global_scope;
  int owns_module_handle;
  const char *(*runtime_version)(void);
  int (*runtime_is_available)(void);
  int (*runtime_layout_info)(spine_tcm_mem_info_t *info);
  int (*runtime_mem_info)(int id, spine_tcm_block_info_t *info);
  void *(*runtime_mem_get)(int id);
  int (*runtime_mem_free)(int id);
  void *(*runtime_mem_try_wait)(int id, size_t pOverTime);
  int (*runtime_mem_release)(int id);
  int (*runtime_mem_force_release)(int id);
  int (*runtime_mem_query)(int id);
} spine_tcm_handle_t;

static inline spine_tcm_handle_t *spine_tcm_default_handle(void) {
  static spine_tcm_handle_t vHandle = {0};
  return &vHandle;
}

static inline void spine_tcm_handle_reset(spine_tcm_handle_t *pHandle) {
  if (pHandle != NULL) {
    memset(pHandle, 0, sizeof(*pHandle));
  }
}

static inline int spine_tcm_handle_bind(spine_tcm_handle_t *pHandle) {
  void *vSymbolScope = pHandle->use_global_scope ? RTLD_DEFAULT : pHandle->module_handle;

  pHandle->runtime_version = (const char *(*)(void))dlsym(vSymbolScope, "spine_tcm_runtime_version");
  pHandle->runtime_is_available = (int (*)(void))dlsym(vSymbolScope, "spine_tcm_runtime_is_available");
  pHandle->runtime_layout_info = (int (*)(spine_tcm_mem_info_t *))dlsym(vSymbolScope, "spine_tcm_runtime_layout_info");
  pHandle->runtime_mem_info = (int (*)(int, spine_tcm_block_info_t *))dlsym(vSymbolScope, "spine_tcm_runtime_mem_info");
  pHandle->runtime_mem_get = (void *(*)(int))dlsym(vSymbolScope, "spine_tcm_runtime_mem_get");
  pHandle->runtime_mem_free = (int (*)(int))dlsym(vSymbolScope, "spine_tcm_runtime_mem_free");
  pHandle->runtime_mem_try_wait = (void *(*)(int, size_t))dlsym(vSymbolScope, "spine_tcm_runtime_mem_try_wait");
  pHandle->runtime_mem_release = (int (*)(int))dlsym(vSymbolScope, "spine_tcm_runtime_mem_release");
  pHandle->runtime_mem_force_release = (int (*)(int))dlsym(vSymbolScope, "spine_tcm_runtime_mem_force_release");
  pHandle->runtime_mem_query = (int (*)(int))dlsym(vSymbolScope, "spine_tcm_runtime_mem_query");

  return pHandle->runtime_version != NULL && pHandle->runtime_is_available != NULL &&
                 pHandle->runtime_layout_info != NULL && pHandle->runtime_mem_info != NULL &&
                 pHandle->runtime_mem_get != NULL && pHandle->runtime_mem_free != NULL &&
                 pHandle->runtime_mem_try_wait != NULL && pHandle->runtime_mem_release != NULL &&
                 pHandle->runtime_mem_force_release != NULL && pHandle->runtime_mem_query != NULL
             ? 0
             : -1;
}

/*
 * Try to bind against an already-loaded process-global spine_tcm instance.
 * The shared library exports spine_tcm_runtime_marker only for this probe.
 */
static inline int spine_tcm_try_bind_global(spine_tcm_handle_t *pHandle) {
  if (dlsym(RTLD_DEFAULT, "spine_tcm_runtime_marker") == NULL) {
    return -1;
  }

  pHandle->use_global_scope = 1;
  return spine_tcm_handle_bind(pHandle);
}

/*
 * Optional pre-bind entry point.
 *
 * Behavior:
 *   - Reuses an already-loaded global spine_tcm instance when available.
 *   - Otherwise loads the shared library from so_path or the default soname.
 *   - Repeated calls are safe and return 0 after the first successful bind.
 */
static inline int spine_tcm_open_handle(const char *pSoPath) {
  spine_tcm_handle_t *vResolved = spine_tcm_default_handle();
  const char *vLibrary = (pSoPath != NULL && pSoPath[0] != '\0') ? pSoPath : "libspine_tcm.so";

  if (vResolved->module_handle != NULL || vResolved->use_global_scope) {
    return 0;
  }

  if (spine_tcm_try_bind_global(vResolved) == 0) {
    return 0;
  }

  spine_tcm_handle_reset(vResolved);

  vResolved->module_handle = dlopen(vLibrary, RTLD_LAZY | RTLD_GLOBAL);
  vResolved->owns_module_handle = vResolved->module_handle != NULL ? 1 : 0;

  if (vResolved->module_handle == NULL) {
    spine_tcm_handle_reset(vResolved);
    return -1;
  }

  if (spine_tcm_handle_bind(vResolved) != 0) {
    if (vResolved->owns_module_handle) {
      dlclose(vResolved->module_handle);
    }
    spine_tcm_handle_reset(vResolved);
    return -1;
  }

  return 0;
}

/* Returns 1 when the runtime driver is available, otherwise 0. */
static inline int spine_tcm_is_available(void) {
  spine_tcm_handle_t *vResolved = spine_tcm_default_handle();

  if (vResolved->module_handle == NULL && !vResolved->use_global_scope) {
    (void)spine_tcm_open_handle(NULL);
  }

  if ((vResolved->module_handle == NULL && !vResolved->use_global_scope) || vResolved->runtime_is_available == NULL) {
    return 0;
  }

  return vResolved->runtime_is_available();
}

/* Returns runtime memory geometry and whether the current backend is fake TCM. */
static inline int spine_tcm_mem_info(spine_tcm_mem_info_t *pInfo) {
  spine_tcm_handle_t *vResolved = spine_tcm_default_handle();

  if (vResolved->module_handle == NULL && !vResolved->use_global_scope) {
    (void)spine_tcm_open_handle(NULL);
  }

  if ((vResolved->module_handle == NULL && !vResolved->use_global_scope) || vResolved->runtime_layout_info == NULL) {
    return -1;
  }

  return vResolved->runtime_layout_info(pInfo);
}

static inline const char *spine_tcm_version(void) {
  spine_tcm_handle_t *vResolved = spine_tcm_default_handle();

  if (vResolved->module_handle == NULL && !vResolved->use_global_scope) {
    (void)spine_tcm_open_handle(NULL);
  }

  if ((vResolved->module_handle == NULL && !vResolved->use_global_scope) || vResolved->runtime_version == NULL) {
    return "unknown";
  }

  return vResolved->runtime_version();
}

/* Returns per-block runtime metadata for the given TCM id. */
static inline int spine_tcm_block_info(int pId, spine_tcm_block_info_t *pInfo) {
  spine_tcm_handle_t *vResolved = spine_tcm_default_handle();

  if (vResolved->module_handle == NULL && !vResolved->use_global_scope) {
    (void)spine_tcm_open_handle(NULL);
  }

  if ((vResolved->module_handle == NULL && !vResolved->use_global_scope) || vResolved->runtime_mem_info == NULL) {
    return -1;
  }

  return vResolved->runtime_mem_info(pId, pInfo);
}

/* Returns a cached buffer for the given TCM id, or NULL on failure. */
static inline void *spine_tcm_mem_get(int pId) {
  spine_tcm_handle_t *vResolved = spine_tcm_default_handle();

  if (vResolved->module_handle == NULL && !vResolved->use_global_scope) {
    (void)spine_tcm_open_handle(NULL);
  }

  if (vResolved->module_handle == NULL && !vResolved->use_global_scope) {
    return NULL;
  }

  if (vResolved->runtime_mem_get == NULL) {
    return NULL;
  }

  return vResolved->runtime_mem_get(pId);
}

/* Releases one reference acquired by spine_tcm_mem_get(id). */
static inline int spine_tcm_mem_free(int pId) {
  spine_tcm_handle_t *vResolved = spine_tcm_default_handle();

  if (vResolved->module_handle == NULL && !vResolved->use_global_scope) {
    (void)spine_tcm_open_handle(NULL);
  }

  if ((vResolved->module_handle == NULL && !vResolved->use_global_scope) || vResolved->runtime_mem_free == NULL) {
    return -1;
  }

  return vResolved->runtime_mem_free(pId);
}

/* Waits for a TCM block handoff and returns the driver-owned buffer when available. */
static inline void *spine_tcm_mem_try_wait(int pId, size_t pOverTime) {
  spine_tcm_handle_t *vResolved = spine_tcm_default_handle();

  if (vResolved->module_handle == NULL && !vResolved->use_global_scope) {
    (void)spine_tcm_open_handle(NULL);
  }

  if (vResolved->module_handle == NULL && !vResolved->use_global_scope) {
    return NULL;
  }

  if (vResolved->runtime_mem_try_wait == NULL) {
    return NULL;
  }

  return vResolved->runtime_mem_try_wait(pId, pOverTime);
}

/* Releases a buffer acquired by spine_tcm_mem_try_wait(id, over_time). */
static inline int spine_tcm_mem_release(int pId) {
  spine_tcm_handle_t *vResolved = spine_tcm_default_handle();

  if (vResolved->module_handle == NULL && !vResolved->use_global_scope) {
    (void)spine_tcm_open_handle(NULL);
  }

  if ((vResolved->module_handle == NULL && !vResolved->use_global_scope) || vResolved->runtime_mem_release == NULL) {
    return -1;
  }

  return vResolved->runtime_mem_release(pId);
}

/* Forces a release for the given TCM id when the backend supports it. */
static inline int spine_tcm_mem_force_release(int pId) {
  spine_tcm_handle_t *vResolved = spine_tcm_default_handle();

  if (vResolved->module_handle == NULL && !vResolved->use_global_scope) {
    (void)spine_tcm_open_handle(NULL);
  }

  if ((vResolved->module_handle == NULL && !vResolved->use_global_scope) ||
      vResolved->runtime_mem_force_release == NULL) {
    return -1;
  }

  return vResolved->runtime_mem_force_release(pId);
}

/* Returns whether the given TCM id is currently acquired. */
static inline int spine_tcm_mem_query(int pId) {
  spine_tcm_handle_t *vResolved = spine_tcm_default_handle();

  if (vResolved->module_handle == NULL && !vResolved->use_global_scope) {
    (void)spine_tcm_open_handle(NULL);
  }

  if ((vResolved->module_handle == NULL && !vResolved->use_global_scope) || vResolved->runtime_mem_query == NULL) {
    return -1;
  }

  return vResolved->runtime_mem_query(pId);
}
#else
static inline const char *spine_tcm_version(void) { return spine_tcm_runtime_version(); }
#endif

#define SPINE_TCM_VERSION (spine_tcm_version())

#ifdef __cplusplus
}
#endif

#endif
