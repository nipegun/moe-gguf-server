#!/bin/bash

set -euo pipefail

readonly cBinaryName=mgs
readonly cMarkerName=.moe-gguf-server-bundle
readonly cArchiveUrl=https://github.com/nipegun/moe-gguf-server/archive/refs/heads/main.tar.gz
readonly cEntryScript="${BASH_SOURCE[0]:-}"
vLogEnabled=false
vLogProcess=''
vBuildRoot=''
vNewLibraryDirectory=''
vNewLicenseDirectory=''
vNewLauncher=''

fCleanup() {
  local vExitStatus=$?
  local vPath
  # Remove partially assembled entries; completed ones were already renamed into place.
  for vPath in "$vNewLibraryDirectory" "$vNewLicenseDirectory" "$vNewLauncher"; do
    if [[ -n "$vPath" ]] && [[ -e "$vPath" || -L "$vPath" ]]; then
      rm -rf -- "$vPath" || true
    fi
  done
  if [[ "$vLogEnabled" == true ]]; then
    exec 1>&3 2>&4
    wait "$vLogProcess" || true
    exec 3>&- 4>&-
  fi
  if [[ -n "$vBuildRoot" ]]; then
    if ! rm -rf -- "$vBuildRoot"; then
      printf 'Cannot remove temporary build directory: %s\n' "$vBuildRoot" >&2
      [[ "$vExitStatus" -ne 0 ]] || vExitStatus=1
    fi
  fi
  if [[ "$vLogEnabled" == true ]]; then
    printf 'Local build finished with status %s.\n' "$vExitStatus"
  fi
  exit "$vExitStatus"
}

trap fCleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
trap 'exit 129' HUP
# A broken log pipe must still remove the private workspace.
trap 'exit 141' PIPE

fFail() {
  printf 'Error: %s\n' "$*" >&2
  exit 1
}

fRequireCommand() {
  local pCommand="$1"
  local pPackage="$2"
  command -v -- "$pCommand" >/dev/null 2>&1 || fFail "Required command not found: ${pCommand}. Install the Debian package ${pPackage} as root."
}

fShowHelp() {
  printf '%s\n' \
    'Build moe-gguf-server locally and install it for the current user, self-contained.' \
    'No services are installed and nothing is left running; launch mgs manually.' \
    '' \
    'Usage: bash build/build-local.sh [options]' \
    '   or: curl -fsSL https://raw.githubusercontent.com/nipegun/moe-gguf-server/refs/heads/main/build/build-local.sh | bash -s -- [options]' \
    'Run from a complete local copy, a standalone script, or curl | bash.' \
    'Without a local source tree beside the script, sources are downloaded from main.' \
    '' \
    'Options:' \
    '  --prefix PATH              Installation prefix (default: ~/.local).' \
    '  --jobs N                   Number of parallel build jobs.' \
    '  --cuda                     Use an existing CUDA Toolkit (default).' \
    '  --vulkan                   Build the Vulkan backend (AMD, Intel or NVIDIA GPUs) instead of CUDA.' \
    '  --cpu                      Build without GPU support.' \
    '  --cuda-architectures LIST  CMAKE_CUDA_ARCHITECTURES value (default: GPUs present at build time).' \
    '  --build-ui                 Build the local, translated web interface (default).' \
    '  --no-ui                    Build without the web interface.' \
    '  --no-openssl               Build without native HTTPS support.' \
    '  --no-nccl                  Disable NCCL explicitly.' \
    '  -h, --help                 Show this help without downloading sources.' \
    '' \
    'Environment: CUDACXX selects the CUDA compiler.' \
    'Sources, build files, caches and logs use a private /tmp/moe-gguf-server-build.*' \
    'folder, removed on exit, including failures and handled interruptions.' \
    'The prefix receives bin/mgs (a static launcher), lib/mgs/ (the server binary and' \
    'every shared library, including the glibc dynamic loader) and share/doc/mgs/licenses/.' \
    'The launcher runs lib/mgs/ through paths relative to its own location, so the prefix' \
    'can be moved as a whole. Only the NVIDIA driver library (libcuda) comes from the system,' \
    'because it must match the running kernel module. With --vulkan, the Vulkan driver of the' \
    'system (its ICD, such as Mesa RADV, and the libraries it loads) is the other exception,' \
    'for the same reason; build on the Debian release that will run the binary.' \
    'System packages are never installed; root is not required.'
}

fHasProjectSources() {
  local pSourceDirectory="$1"
  local vEntry
  for vEntry in build/CMakeLists.txt build/mgs-launcher.c \
    backend/CMakeLists.txt backend/ggml/CMakeLists.txt backend/tools/server/CMakeLists.txt \
    frontend/CMakeLists.txt frontend/package.json frontend/package-lock.json; do
    [[ -f "${pSourceDirectory}/${vEntry}" ]] || return 1
  done
  [[ -d "${pSourceDirectory}/build/licenses" ]]
}

fDownloadSources() {
  local pSourceDirectory="$1"
  local vArchive="${vBuildRoot}/source.tar.gz"
  local vMember
  fRequireCommand curl curl
  fRequireCommand gzip gzip
  printf 'Downloading source snapshot: %s\n' "$cArchiveUrl"
  curl --fail --location --show-error --silent --proto '=https' --proto-redir '=https' \
    --connect-timeout 30 --retry 3 --output "$vArchive" "$cArchiveUrl" || fFail 'Cannot download the main source archive; check the connection and CA certificates (ca-certificates).'
  tar -tzf "$vArchive" > "${vBuildRoot}/members.txt" || fFail 'The downloaded source archive is invalid or incomplete.'
  while IFS= read -r vMember; do
    [[ "$vMember" == moe-gguf-server-main/* && "/${vMember}/" != */../* && "/${vMember}/" != */./* ]] || fFail 'The source archive contains an unexpected path.'
  done < "${vBuildRoot}/members.txt"
  tar -xzf "$vArchive" --strip-components=1 --no-same-owner --no-same-permissions \
    -C "$pSourceDirectory" || fFail 'Cannot extract the source archive.'
  fHasProjectSources "$pSourceDirectory" || fFail 'The source archive is missing required project files.'
}

fDetectJobs() {
  if command -v nproc >/dev/null 2>&1; then
    nproc 2>/dev/null || printf '1\n'
  elif command -v getconf >/dev/null 2>&1; then
    getconf _NPROCESSORS_ONLN 2>/dev/null || printf '1\n'
  else
    printf '1\n'
  fi
}

fResolveExecutable() {
  local pCandidate="$1"
  if [[ "$pCandidate" == */* ]]; then
    if [[ -x "$pCandidate" ]]; then
      printf '%s\n' "$pCandidate"
    fi
  else
    command -v -- "$pCandidate" 2>/dev/null || true
  fi
  return 0
}

fIsDriverLibrary() {
  local pName="$1"
  # The driver user-space libraries must match the loaded kernel module, so they are never bundled.
  case "$pName" in
    libcuda.so|libcuda.so.*|libnvidia-*.so|libnvidia-*.so.*|libcudadebugger.so|libcudadebugger.so.*) return 0 ;;
  esac
  return 1
}

fListLibraries() {
  local pObject="$1"
  local vListing
  vListing="$(env -u LD_LIBRARY_PATH -u LD_PRELOAD ldd "$pObject")" || fFail "Cannot list the shared libraries of ${pObject}."
  if [[ "$vListing" == *'=> not found'* ]]; then
    printf '%s\n' "$vListing" >&2
    fFail "Shared libraries required by ${pObject} are missing."
  fi
  printf '%s\n' "$vListing" | sed -nE 's/^[[:space:]]*([^[:space:]]+) => (\/[^[:space:]]+) \(0x[0-9a-fA-F]+\)$/\1\t\2/p'
}

fBundleDependencies() {
  local pObject="$1"
  local pLibraryDirectory="$2"
  local vEntries vName vPath
  vEntries="$(fListLibraries "$pObject")" || fFail "Cannot resolve the dependencies of ${pObject}."
  while IFS=$'\t' read -r vName vPath; do
    [[ -n "$vName" ]] || continue
    if fIsDriverLibrary "$vName"; then
      printf 'Kept outside the bundle (NVIDIA driver): %s\n' "$vPath"
      continue
    fi
    if [[ ! -e "${pLibraryDirectory}/${vName}" ]]; then
      install -m 0755 -- "$vPath" "${pLibraryDirectory}/${vName}" || fFail "Cannot copy ${vPath}."
    fi
  done < <(printf '%s\n' "$vEntries")
}

fBundleNameServiceModules() {
  local pSystemDirectory="$1"
  local pLibraryDirectory="$2"
  local vModule
  # glibc loads these with dlopen for host and user lookups; the binary's RPATH also covers them.
  for vModule in "${pSystemDirectory}"/libnss_*.so.2; do
    [[ -f "$vModule" ]] || continue
    install -m 0755 -- "$vModule" "${pLibraryDirectory}/${vModule##*/}" || fFail "Cannot copy ${vModule}."
    fBundleDependencies "$vModule" "$pLibraryDirectory"
  done
}

fPinLibrarySearchPaths() {
  local pLibraryDirectory="$1"
  local pInterpreterName="$2"
  local vLibrary vSearchPath
  # DT_RPATH takes precedence over LD_LIBRARY_PATH, so bundled libraries always find their siblings first.
  for vLibrary in "${pLibraryDirectory}"/*; do
    [[ -f "$vLibrary" ]] || continue
    [[ "${vLibrary##*/}" != "$pInterpreterName" ]] || continue
    vSearchPath="$(patchelf --print-rpath "$vLibrary")" || fFail "Cannot read the search path of ${vLibrary}."
    if [[ -n "$vSearchPath" ]]; then
      patchelf --force-rpath --set-rpath '$ORIGIN' "$vLibrary" || fFail "Cannot set the search path of ${vLibrary}."
    fi
  done
}

fReplaceDirectory() {
  local pNewDirectory="$1"
  local pFinalDirectory="$2"
  local vPreviousDirectory="${pFinalDirectory%/*}/.${pFinalDirectory##*/}.previous"
  rm -rf -- "$vPreviousDirectory" || fFail "Cannot remove ${vPreviousDirectory}."
  if [[ -e "$pFinalDirectory" || -L "$pFinalDirectory" ]]; then
    mv -T -- "$pFinalDirectory" "$vPreviousDirectory" || fFail "Cannot move aside ${pFinalDirectory}."
  fi
  if ! mv -T -- "$pNewDirectory" "$pFinalDirectory"; then
    if [[ -e "$vPreviousDirectory" ]]; then
      mv -T -- "$vPreviousDirectory" "$pFinalDirectory" || true
    fi
    fFail "Cannot install ${pFinalDirectory}."
  fi
  rm -rf -- "$vPreviousDirectory" || fFail "Cannot remove ${vPreviousDirectory}."
}

fVerifyBundle() {
  local pLibraryDirectory="$1"
  local pInterpreterName="$2"
  local vListing vLine vName vPath vProblems=''
  # Resolve exactly as the launcher does: the bundled loader runs the bundled server explicitly.
  vListing="$("${pLibraryDirectory}/${pInterpreterName}" --list "${pLibraryDirectory}/${cBinaryName}")" || fFail 'The bundled loader cannot resolve the server binary.'
  while IFS= read -r vLine; do
    [[ -n "$vLine" ]] || continue
    case "$vLine" in
      *'=> not found'*) vProblems+="${vLine}"$'\n'; continue ;;
      linux-vdso.so.*) continue ;;
    esac
    if [[ "$vLine" == *' => '* ]]; then
      vName="${vLine%% => *}"
      vPath="${vLine#* => }"
    else
      vPath="$vLine"
      vName="${vLine##*/}"
    fi
    vPath="${vPath% (0x*}"
    vName="${vName% (0x*}"
    # The loader line repeats the binary's original PT_INTERP, but the running loader is the bundled one.
    if [[ "$vPath" == "${pLibraryDirectory}/"* || "$vName" == "$pInterpreterName" ]] || fIsDriverLibrary "$vName"; then
      continue
    fi
    vProblems+="${vLine}"$'\n'
  done < <(printf '%s\n' "$vListing" | sed -E 's/^[[:space:]]+//')
  if [[ -n "$vProblems" ]]; then
    printf '%s' "$vProblems" >&2
    fFail 'Some libraries would still be loaded from outside the bundle.'
  fi
}

fMain() {
  local vScriptDirectory vProjectDirectory='' vStagedSourceDirectory vCmakeDirectory
  local vBundleDirectory vBundleLibraryDirectory vLibraryDirectory vLicenseDirectory vLauncherPath
  local vPrefix vBuildJobs='' vBuildBackend=cuda vBuildUi=ON vBuildOpenSsl=ON vUseNccl=ON
  local vCudaArchitectures='' vCudaEnabled=OFF vNcclEnabled=OFF vCudaCompiler='' vCudaVersion=''
  local vVulkanEnabled=OFF vSystemLibraries
  local vNodeMajor vNodeMinor vNcclIncludeDirectory vNcclLibrary vServerBinary vCandidate
  local vInterpreter vInterpreterName vLibcPath vEntry
  local -a aGeneratorArguments=() aConfigureArguments=()
  vPrefix=''

  while [[ $# -gt 0 ]]; do
    case "$1" in
      --prefix|--jobs|--cuda-architectures)
        [[ $# -ge 2 ]] || fFail "Missing value after $1."
        case "$1" in
          --prefix) vPrefix="$2" ;;
          --jobs) vBuildJobs="$2" ;;
          --cuda-architectures) vCudaArchitectures="$2" ;;
        esac
        shift 2
        ;;
      --cuda) vBuildBackend=cuda; shift ;;
      --vulkan) vBuildBackend=vulkan; shift ;;
      --cpu) vBuildBackend=cpu; shift ;;
      --build-ui) vBuildUi=ON; shift ;;
      --no-ui) vBuildUi=OFF; shift ;;
      --no-openssl) vBuildOpenSsl=OFF; shift ;;
      --no-nccl) vUseNccl=OFF; shift ;;
      -h|--help) fShowHelp; return 0 ;;
      *) fFail "Unknown option: $1" ;;
    esac
  done

  if [[ -n "$cEntryScript" && -f "$cEntryScript" ]]; then
    vScriptDirectory="$(CDPATH= cd -- "$(dirname -- "$cEntryScript")" && pwd -P)" || fFail 'Cannot locate the build script.'
    if fHasProjectSources "${vScriptDirectory}/.."; then
      vProjectDirectory="$(CDPATH= cd -- "${vScriptDirectory}/.." && pwd -P)" || fFail 'Cannot locate the project.'
    fi
  fi
  if [[ -z "$vPrefix" ]]; then
    [[ -n "${HOME:-}" ]] || fFail 'HOME is not set; use --prefix.'
    vPrefix="${HOME}/.local"
  fi
  fRequireCommand realpath coreutils
  fRequireCommand flock util-linux
  fRequireCommand tee coreutils
  fRequireCommand mktemp coreutils
  vPrefix="$(realpath -m -- "$vPrefix")" || fFail 'Cannot resolve the installation prefix.'
  [[ "$vPrefix" != / ]] || fFail '--prefix cannot be the root directory.'
  if [[ -n "$vProjectDirectory" ]]; then
    [[ "$vPrefix" != "$vProjectDirectory" && "$vPrefix" != "$vProjectDirectory/"* ]] || fFail '--prefix must be outside the project.'
  fi
  vLauncherPath="${vPrefix}/bin/${cBinaryName}"
  vLibraryDirectory="${vPrefix}/lib/${cBinaryName}"
  vLicenseDirectory="${vPrefix}/share/doc/${cBinaryName}/licenses"
  [[ -n "$vBuildJobs" ]] || vBuildJobs="$(fDetectJobs)"
  [[ "$vBuildJobs" =~ ^[1-9][0-9]*$ ]] || fFail '--jobs must be a positive integer.'
  vBuildRoot="$(mktemp -d /tmp/moe-gguf-server-build.XXXXXXXX)" || fFail 'Cannot create the temporary build directory in /tmp.'
  vStagedSourceDirectory="${vBuildRoot}/source"
  vCmakeDirectory="${vBuildRoot}/cmake"
  vBundleDirectory="${vBuildRoot}/bundle"
  vBundleLibraryDirectory="${vBundleDirectory}/lib/${cBinaryName}"

  # Keep tool scratch files and caches inside this disposable workspace.
  export TMPDIR="${vBuildRoot}/tmp"
  export TMP="$TMPDIR"
  export TEMP="$TMPDIR"
  export XDG_CACHE_HOME="${vBuildRoot}/cache"
  export npm_config_cache="${vBuildRoot}/npm-cache"
  export npm_config_logs_dir="${vBuildRoot}/npm-logs"
  export NODE_COMPILE_CACHE="${vBuildRoot}/node-cache"
  export CUDA_CACHE_PATH="${vBuildRoot}/cuda-cache"
  export CCACHE_DIR="${vBuildRoot}/ccache"
  export CCACHE_TEMPDIR="$TMPDIR"
  export CCACHE_DISABLE=1
  mkdir -p "$TMPDIR" "$XDG_CACHE_HOME" "$npm_config_cache" "$npm_config_logs_dir" \
    "$NODE_COMPILE_CACHE" "$CUDA_CACHE_PATH" "$CCACHE_DIR" || fFail 'Cannot create temporary tool directories.'
  exec 3>&1 4>&2
  exec > >(tee -a "${vBuildRoot}/build.log") 2>&1
  vLogProcess=$!
  vLogEnabled=true
  printf '\nmoe-gguf-server local build: %s\n' "$(date -u '+%Y-%m-%dT%H:%M:%SZ')"
  printf 'Installation prefix: %s\n' "$vPrefix"
  printf 'Temporary build folder (removed on exit): %s\n' "$vBuildRoot"

  fRequireCommand cmake cmake
  fRequireCommand cc build-essential
  fRequireCommand c++ build-essential
  fRequireCommand tar tar
  fRequireCommand install coreutils
  fRequireCommand sed sed
  fRequireCommand ldd libc-bin
  fRequireCommand patchelf patchelf
  if [[ "$vBuildBackend" == cuda ]]; then
    vCudaEnabled=ON
    [[ "$vUseNccl" != ON ]] || vNcclEnabled=ON
    if [[ -n "${CUDACXX:-}" ]]; then
      vCudaCompiler="$(fResolveExecutable "$CUDACXX")"
    elif command -v nvcc >/dev/null 2>&1; then
      vCudaCompiler="$(command -v nvcc)"
    elif [[ -x /usr/local/cuda/bin/nvcc ]]; then
      vCudaCompiler=/usr/local/cuda/bin/nvcc
    fi
    [[ -n "$vCudaCompiler" ]] || fFail 'CUDA Toolkit not found: install it, set CUDACXX, or use --vulkan or --cpu.'
    vCudaCompiler="$(realpath -e -- "$vCudaCompiler")" || fFail 'Cannot resolve the CUDA compiler path.'
    vCudaVersion="$("$vCudaCompiler" --version | sed -nE 's/.*release ([0-9]+\.[0-9]+).*/\1/p')" || fFail 'Cannot read the CUDA version.'
    vCudaVersion="${vCudaVersion%%$'\n'*}"
    [[ -n "$vCudaVersion" ]] || fFail 'Cannot identify the CUDA Toolkit version.'
    printf 'Using existing CUDA Toolkit %s: %s\n' "$vCudaVersion" "$vCudaCompiler"
  fi
  if [[ "$vBuildBackend" == vulkan ]]; then
    vVulkanEnabled=ON
    # The shaders are compiled at build time; CMake also needs the Vulkan headers and SPIR-V headers.
    fRequireCommand glslc glslc
    printf 'Using the Vulkan backend (shader compiler: %s).\n' "$(command -v glslc)"
  fi
  if [[ "$vBuildUi" == ON ]]; then
    fRequireCommand node nodejs
    fRequireCommand npm npm
    fRequireCommand gzip gzip
    vNodeMajor="$(node -p 'process.versions.node.split(".").map(Number).slice(0, 2).join(" ")')" || fFail 'Cannot identify Node.js.'
    vNodeMinor="${vNodeMajor#* }"
    vNodeMajor="${vNodeMajor%% *}"
    [[ "$vNodeMajor" -eq 20 && "$vNodeMinor" -ge 19 || "$vNodeMajor" -eq 22 && "$vNodeMinor" -ge 12 || "$vNodeMajor" -gt 22 ]] || fFail 'The local interface requires Node.js 20.19+ or 22.12+. Install a supported version or use --no-ui.'
  fi

  if command -v ninja >/dev/null 2>&1; then
    aGeneratorArguments=(-G Ninja)
  fi
  aConfigureArguments=(
    -S "${vStagedSourceDirectory}/build"
    -B "$vCmakeDirectory"
    -DCMAKE_BUILD_TYPE=Release
    -DBUILD_SHARED_LIBS=OFF
    -DGGML_CCACHE=OFF
    -DGGML_NATIVE=ON
    "-DGGML_CUDA=${vCudaEnabled}"
    "-DGGML_CUDA_NCCL=${vNcclEnabled}"
    "-DGGML_VULKAN=${vVulkanEnabled}"
    "-DMGS_BUILD_UI=${vBuildUi}"
    "-DMGS_OPENSSL=${vBuildOpenSsl}"
  )
  [[ -z "$vCudaCompiler" ]] || aConfigureArguments+=("-DCMAKE_CUDA_COMPILER=${vCudaCompiler}")
  [[ -z "$vCudaArchitectures" ]] || aConfigureArguments+=("-DCMAKE_CUDA_ARCHITECTURES=${vCudaArchitectures}")

  # Local staging excludes stale generated files; remote sources stay in the same disposable workspace.
  printf 'Staging sources in %s.\n' "$vStagedSourceDirectory"
  install -d -m 0755 "$vStagedSourceDirectory" || fFail 'Cannot create the staged source directory.'
  if [[ -n "$vProjectDirectory" ]]; then
    printf 'Using local sources: %s\n' "$vProjectDirectory"
    tar -C "$vProjectDirectory" \
      --exclude='./.git' --exclude='./.agents' --exclude='./.claude' --exclude='./.codex' --exclude='./.aws' \
      --exclude='./_' --exclude='./build-*' --exclude='./cmake-build-*' \
      --exclude='*/node_modules' --exclude='*/.svelte-kit' --exclude='*/dist' \
      -cf - . | tar -C "$vStagedSourceDirectory" -xf - || fFail 'Cannot stage the project.'
  else
    fDownloadSources "$vStagedSourceDirectory"
  fi
  cd -- "$vStagedSourceDirectory" || fFail 'Cannot enter the staged source directory.'
  printf 'Configuring moe-gguf-server (%s) in %s.\n' "$vBuildBackend" "$vCmakeDirectory"
  if ! cmake "${aGeneratorArguments[@]}" "${aConfigureArguments[@]}"; then
    [[ "$vVulkanEnabled" != ON ]] || fFail 'CMake configuration failed. The Vulkan backend needs the Debian packages libvulkan-dev, spirv-headers and glslc.'
    fFail 'CMake configuration failed.'
  fi
  if [[ "$vNcclEnabled" == ON ]]; then
    vNcclIncludeDirectory="$(sed -n 's/^NCCL_INCLUDE_DIR:PATH=//p' "${vCmakeDirectory}/CMakeCache.txt")" || fFail 'Cannot read NCCL include configuration.'
    vNcclLibrary="$(sed -n 's/^NCCL_LIBRARY:FILEPATH=//p' "${vCmakeDirectory}/CMakeCache.txt")" || fFail 'Cannot read NCCL library configuration.'
    vNcclIncludeDirectory="${vNcclIncludeDirectory%%$'\n'*}"
    vNcclLibrary="${vNcclLibrary%%$'\n'*}"
    [[ -n "$vNcclIncludeDirectory" && "$vNcclIncludeDirectory" != *-NOTFOUND && -n "$vNcclLibrary" && "$vNcclLibrary" != *-NOTFOUND ]] || fFail 'NCCL was not detected; install libnccl2 and libnccl-dev or use --no-nccl.'
    printf 'NCCL enabled: %s (headers: %s).\n' "$vNcclLibrary" "$vNcclIncludeDirectory"
  fi
  cmake --build "$vCmakeDirectory" --config Release --target mgs --parallel "$vBuildJobs" || fFail 'Compilation failed.'
  vServerBinary=''
  for vCandidate in "${vCmakeDirectory}/bin/${cBinaryName}" "${vCmakeDirectory}/bin/Release/${cBinaryName}"; do
    if [[ -x "$vCandidate" ]]; then
      vServerBinary="$vCandidate"
      break
    fi
  done
  [[ -n "$vServerBinary" ]] || fFail "Compilation did not produce ${cBinaryName}."
  vInterpreter="$(patchelf --print-interpreter "$vServerBinary")" || fFail 'Cannot read the dynamic loader of the binary.'
  vInterpreterName="${vInterpreter##*/}"
  [[ "$vInterpreterName" =~ ^[A-Za-z0-9._+-]+$ ]] || fFail "Unexpected dynamic loader name: ${vInterpreterName}."
  # The launcher is static so that it needs no loader of its own and can resolve lib/mgs/ relative to itself.
  printf '%s\n' 'Building the relocatable launcher.'
  cc -std=c11 -O2 -Wall -Wextra -static "-DcLoaderName=\"${vInterpreterName}\"" \
    -o "${vBuildRoot}/launcher" "${vStagedSourceDirectory}/build/mgs-launcher.c" || fFail 'Cannot build the static launcher; install libc6-dev, which provides the static C library.'

  printf 'Collecting shared libraries in %s.\n' "$vBundleLibraryDirectory"
  install -d -m 0755 "$vBundleLibraryDirectory" || fFail 'Cannot create the library folder.'
  install -m 0755 -- "$vInterpreter" "${vBundleLibraryDirectory}/${vInterpreterName}" || fFail 'Cannot copy the dynamic loader.'
  fBundleDependencies "$vServerBinary" "$vBundleLibraryDirectory"
  vLibcPath="$(fListLibraries "$vServerBinary" | sed -nE 's/^libc\.so\.6\t(.+)$/\1/p')" || fFail 'Cannot locate the C library.'
  vLibcPath="${vLibcPath%%$'\n'*}"
  [[ -n "$vLibcPath" ]] || fFail 'Cannot locate the C library.'
  fBundleNameServiceModules "${vLibcPath%/*}" "$vBundleLibraryDirectory"
  fPinLibrarySearchPaths "$vBundleLibraryDirectory" "$vInterpreterName"

  # The server lives beside its libraries; $ORIGIN keeps every lookup relative to wherever lib/mgs/ is.
  install -m 0755 -- "$vServerBinary" "${vBundleLibraryDirectory}/${cBinaryName}" || fFail 'Cannot copy the server binary.'
  patchelf --force-rpath --set-rpath '$ORIGIN' "${vBundleLibraryDirectory}/${cBinaryName}" || fFail 'Cannot point the server binary to the bundled libraries.'
  if [[ -d "${vStagedSourceDirectory}/build/licenses" ]]; then
    cp -R -- "${vStagedSourceDirectory}/build/licenses" "${vBundleDirectory}/licenses" || fFail 'Cannot collect the licenses.'
  fi
  # The project's own license travels with the third-party notices.
  if [[ -f "${vStagedSourceDirectory}/LICENSE" ]]; then
    install -d -m 0755 "${vBundleDirectory}/licenses" || fFail 'Cannot create the license folder.'
    install -m 0644 -- "${vStagedSourceDirectory}/LICENSE" "${vBundleDirectory}/licenses/LICENSE" || fFail 'Cannot collect the project license.'
  fi

  # The ownership marker travels inside the library folder, which is replaced as a whole.
  printf '%s\n' 'Self-contained moe-gguf-server installation created by build/build-local.sh.' \
    > "${vBundleLibraryDirectory}/${cMarkerName}" || fFail 'Cannot write the installation marker.'

  # Serialize updates of this prefix without leaving a lock file behind.
  install -d -m 0755 "$vPrefix" "${vPrefix}/bin" "${vPrefix}/lib" "${vLicenseDirectory%/*}" || fFail 'Cannot create the prefix folders.'
  exec 9<"$vPrefix" || fFail 'Cannot open the installation prefix for locking.'
  flock -n 9 || fFail 'Another local build is already updating this installation prefix.'
  # Never replace an mgs installation that this script did not create.
  if [[ ! -f "${vLibraryDirectory}/${cMarkerName}" ]]; then
    for vEntry in "$vLauncherPath" "$vLibraryDirectory" "$vLicenseDirectory"; do
      [[ ! -e "$vEntry" && ! -L "$vEntry" ]] || fFail "${vEntry} already exists and was not created by this script; choose another --prefix."
    done
  fi

  # Copy the completed artifacts before renaming them on the destination filesystem.
  vNewLibraryDirectory="${vPrefix}/lib/.${cBinaryName}.new"
  rm -rf -- "$vNewLibraryDirectory" || fFail 'Cannot remove a previous partial library folder.'
  cp -R -- "$vBundleLibraryDirectory" "$vNewLibraryDirectory" || fFail 'Cannot copy the library folder.'
  vNewLauncher="${vPrefix}/bin/.${cBinaryName}.new"
  install -m 0755 -- "${vBuildRoot}/launcher" "$vNewLauncher" || fFail 'Cannot copy the launcher.'

  if [[ -d "${vBundleDirectory}/licenses" ]]; then
    vNewLicenseDirectory="${vLicenseDirectory%/*}/.licenses.new"
    rm -rf -- "$vNewLicenseDirectory" || fFail 'Cannot remove a previous partial license folder.'
    cp -R -- "${vBundleDirectory}/licenses" "$vNewLicenseDirectory" || fFail 'Cannot copy the licenses.'
  fi

  fReplaceDirectory "$vNewLibraryDirectory" "$vLibraryDirectory"
  vNewLibraryDirectory=''
  mv -f -T -- "$vNewLauncher" "$vLauncherPath" || fFail 'Cannot install the launcher.'
  vNewLauncher=''
  if [[ -n "$vNewLicenseDirectory" ]]; then
    fReplaceDirectory "$vNewLicenseDirectory" "$vLicenseDirectory"
    vNewLicenseDirectory=''
  fi

  fVerifyBundle "$vLibraryDirectory" "$vInterpreterName"
  "$vLauncherPath" --version || fFail 'The installed binary does not start.'

  printf '\nReady: %s\n' "$vLauncherPath"
  case "$vBuildBackend" in
    cuda) vSystemLibraries='only the NVIDIA driver is taken from the system' ;;
    vulkan) vSystemLibraries='only the Vulkan driver (ICD) and the libraries it loads are taken from the system' ;;
    *) vSystemLibraries='nothing is taken from the system' ;;
  esac
  printf 'Libraries: %s (%s).\n' "$vLibraryDirectory" "$vSystemLibraries"
  printf 'Licenses: %s\n' "$vLicenseDirectory"
  printf '%s\n' 'The prefix is relocatable: move bin/mgs and lib/mgs/ together, keeping their bin/ and lib/ levels.'
  case ":${PATH:-}:" in
    *":${vPrefix}/bin:"*) ;;
    *) printf 'Add %s/bin to PATH to run mgs by name (on Debian, ~/.profile adds ~/.local/bin at login once it exists).\n' "$vPrefix" ;;
  esac
  if [[ "$vBuildOpenSsl" == ON ]]; then
    printf 'Example: %s --model /path/model.gguf\n' "$vLauncherPath"
    printf '%s\n' 'The server defaults to HTTPS from port 11443 and HTTP redirection from port 11080.' \
      'Open the HTTPS URL printed at startup (normally https://127.0.0.1:11443/; API documentation: /api/doc/).'
  else
    printf 'Example: %s --http --model /path/model.gguf\n' "$vLauncherPath"
    printf '%s\n' 'This build has no HTTPS support; --http serves plain HTTP from port 11080.'
  fi
}

if ! fMain "$@" < /dev/null; then
  printf '%s\n' 'Local build failed.' >&2
  exit 1
fi
