// Relocatable entry point of the self-contained local build, installed as PREFIX/bin/mgs.
// It starts PREFIX/lib/mgs/mgs through the bundled glibc dynamic loader in
// PREFIX/lib/mgs/, both found relative to this launcher, so the installation keeps
// working wherever the prefix is moved. The launcher is linked statically and loads no library.

#define _POSIX_C_SOURCE 200809L

#include <errno.h>
#include <limits.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#ifndef cLoaderName
#error "Define cLoaderName with the file name of the bundled dynamic loader."
#endif

static const char *const cServerName = "mgs";
static const char *const cExecutableVariable = "MGS_EXECUTABLE";

static int fBuildPath(char *pBuffer, size_t pSize, const char *pPrefix, const char *pName) {
  int vLength = snprintf(pBuffer, pSize, "%s/lib/%s/%s", pPrefix, cServerName, pName);
  return vLength > 0 && (size_t)vLength < pSize;
}

int main(int pArgumentCount, char **pArguments) {
  char vSelfPath[PATH_MAX];
  char vLoaderPath[PATH_MAX];
  char vServerPath[PATH_MAX];
  char **aArguments;
  char *vSlash;
  ssize_t vLength;
  int vCount;
  int vIndex;

  vLength = readlink("/proc/self/exe", vSelfPath, sizeof(vSelfPath) - 1);
  if (vLength < 0) {
    fprintf(stderr, "mgs: cannot resolve the launcher location: %s\n", strerror(errno));
    return 127;
  }
  if (vLength == 0 || (size_t)vLength >= sizeof(vSelfPath) - 1) {
    fprintf(stderr, "mgs: the launcher path is too long\n");
    return 127;
  }
  vSelfPath[vLength] = '\0';

  // The server process image is the loader, so the router re-executes this launcher for child servers.
  if (setenv(cExecutableVariable, vSelfPath, 1) != 0) {
    fprintf(stderr, "mgs: cannot set %s: %s\n", cExecutableVariable, strerror(errno));
    return 127;
  }

  // PREFIX/bin/mgs -> PREFIX
  for (vIndex = 0; vIndex < 2; vIndex++) {
    vSlash = strrchr(vSelfPath, '/');
    if (vSlash == NULL) {
      fprintf(stderr, "mgs: unexpected launcher path: %s\n", vSelfPath);
      return 127;
    }
    *vSlash = '\0';
  }
  if (!fBuildPath(vLoaderPath, sizeof(vLoaderPath), vSelfPath, cLoaderName) ||
      !fBuildPath(vServerPath, sizeof(vServerPath), vSelfPath, cServerName)) {
    fprintf(stderr, "mgs: the installation path is too long\n");
    return 127;
  }

  // loader server [arguments...]
  vCount = pArgumentCount > 0 ? pArgumentCount : 1;
  aArguments = calloc((size_t)vCount + 2, sizeof(char *));
  if (aArguments == NULL) {
    fprintf(stderr, "mgs: out of memory\n");
    return 127;
  }
  aArguments[0] = vLoaderPath;
  aArguments[1] = vServerPath;
  for (vIndex = 1; vIndex < pArgumentCount; vIndex++) {
    aArguments[vIndex + 1] = pArguments[vIndex];
  }

  execv(vLoaderPath, aArguments);
  fprintf(stderr, "mgs: cannot start %s: %s\n", vLoaderPath, strerror(errno));
  free(aArguments);
  return 127;
}
